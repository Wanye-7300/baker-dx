use std::collections;

use dioxus::prelude::*;

use futures::StreamExt;
use indexed_db_futures::prelude::*;
use indexed_db_futures::transaction::TransactionMode;
use indexed_db_futures::{database::Database, KeyRange};
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use web_sys::js_sys::{Object, Reflect};
use web_sys::wasm_bindgen::{JsCast, JsValue};

const MAX_SAFE_INTEGER: u64 = 9007199254740991;

static DB: GlobalSignal<Option<Database>> = Signal::global(|| None);

#[derive(Serialize, Deserialize, Clone, Debug)]
pub(crate) struct MessageWrapper {
    pub(crate) session_uuid: Uuid,
    pub(crate) message_id: u64,

    #[serde(flatten)]
    pub(crate) message: crate::Message,
}

pub(crate) async fn open_db() -> indexed_db_futures::Result<()> {
    let db: Database = Database::open("baker")
        .with_version(1u8)
        .with_on_upgrade_needed_fut(async move |event, db| {
            let old_version = event.old_version() as u64;
            let new_version = event.new_version().map(|v| v as u64);

            match (old_version, new_version) {
                (0, Some(1)) => {
                    db.create_object_store("messages")
                        .with_key_path(indexed_db_futures::KeyPath::Sequence(
                            vec!["session_uuid", "message_id"].into(),
                        ))
                        .build()?;

                    db.create_object_store("multimedia")
                        .with_key_path(indexed_db_futures::KeyPath::One("uuid"))
                        .build()?;
                }
                _ => {
                    panic!("Unknown Database Version");
                }
            }

            Ok(())
        })
        .await
        .unwrap();

    *DB.write() = Some(db);

    Ok(())
}

pub(crate) async fn put_messages(messages: Vec<MessageWrapper>) -> indexed_db_futures::Result<()> {
    let db = {
        let db = DB.read();
        db.as_ref().cloned().expect("Database not initialized")
    };

    let transaction = db
        .transaction("messages")
        .with_mode(TransactionMode::Readwrite)
        .build()?;

    let object_store = transaction.object_store("messages")?;

    for message in messages {
        object_store.put(message).serde()?.await?;
    }

    transaction.commit().await?;

    Ok(())
}

///
/// 插入一条消息
///
/// - 如果对应的 `message_id` 不存在消息，则直接在其上放消息
///
/// - 否则，给 `message_id` **及**之后的所有消息的 `id` + 1，然后再插入
///
/// ## 返回值
///
/// 如果需要把 `id` + 1，则返回 `true`
pub(crate) async fn insert_message(message: MessageWrapper) -> indexed_db_futures::Result<bool> {
    let db = {
        let db = DB.read();
        db.as_ref().cloned().expect("Database not initialized")
    };

    let transaction = db
        .transaction("messages")
        .with_mode(TransactionMode::Readwrite)
        .build()?;

    let obj_store = transaction.object_store("messages")?;

    let key_range = KeyRange::Only((message.session_uuid, message.message_id));

    let (need_to_update_index, messages) =
        if let Some(MessageWrapper { .. }) = obj_store.get(key_range).serde()?.await? {
            // 否则，给 `message_id` **及**之后的所有消息的 `id` + 1，然后再插入
            let key_range = KeyRange::Bound(
                (message.session_uuid, message.message_id),
                false,
                (message.session_uuid, MAX_SAFE_INTEGER),
                false,
            );

            if let Some(cursor) = obj_store.open_cursor().with_query(key_range).serde()?.await? {
                let key_range = KeyRange::Bound(
                    (message.session_uuid, message.message_id),
                    false,
                    (message.session_uuid, MAX_SAFE_INTEGER),
                    false,
                );

                let stream = cursor.stream_ser::<MessageWrapper>();
                let mut messages = stream.map(|x| x.unwrap()).collect::<Vec<MessageWrapper>>().await;
                messages.iter_mut().map(|x| x.message_id += 1).count();
                messages.push(message);

                obj_store.delete(key_range).serde()?.await?;

                (true, messages)
            } else {
                unreachable!()
            }
        } else {
            // 如果对应的 `message_id` 不存在消息，则直接在其上放消息
            (false, vec![message])
        };

    for message in messages {
        obj_store.put(message).serde()?.await?;
    }

    transaction.commit().await?;

    Ok(need_to_update_index)
}

pub(crate) async fn modify_message(message: MessageWrapper) -> indexed_db_futures::Result<()> {
    let db = {
        let db = DB.read();
        db.as_ref().cloned().expect("Database not initialized")
    };

    let transaction = db
        .transaction("messages")
        .with_mode(TransactionMode::Readwrite)
        .build()?;
    let object_store = transaction.object_store("messages")?;
    object_store.put(message).serde()?.await?;
    transaction.commit().await?;
    Ok(())
}

pub(crate) async fn get_messages(
    session_uuid: Uuid,
) -> indexed_db_futures::Result<collections::BTreeMap<u64, crate::Message>> {
    let db = {
        let db = DB.read();
        db.as_ref().cloned().expect("Database not initialized")
    };

    let transaction = db
        .transaction("messages")
        .with_mode(TransactionMode::Readonly)
        .build()?;

    let object_store = transaction.object_store("messages")?;

    let key_range = KeyRange::Bound((session_uuid, 0), false, (session_uuid, MAX_SAFE_INTEGER), false);

    let Some(cursor) = object_store.open_cursor().with_query(key_range).serde()?.await? else {
        return Ok(collections::BTreeMap::new());
    };

    let stream = cursor.stream_ser::<MessageWrapper>();
    let records = stream
        .map(|x| x.unwrap())
        .map(|x| (x.message_id, x.message))
        .collect::<collections::BTreeMap<u64, crate::Message>>()
        .await;

    Ok(records)
}

pub(crate) async fn delete_message(session_uuid: Uuid, message_id: u64) -> indexed_db_futures::Result<()> {
    let db = {
        let db = DB.read();
        db.as_ref().cloned().expect("Database not initialized")
    };

    let transaction = db
        .transaction(["messages", "multimedia"])
        .with_mode(TransactionMode::Readwrite)
        .build()?;

    let object_store = transaction.object_store("messages")?;
    let object_store_multimedia = transaction.object_store("multimedia")?;

    let key_range = KeyRange::Only((session_uuid, message_id));

    if let Some(message) = object_store.get(key_range.clone()).serde()?.await? {
        let message: MessageWrapper = message;

        if let crate::MessageType::Image(uuid) = message.message.content {
            object_store_multimedia.delete(KeyRange::Only(uuid)).serde()?.await?;
        }

        object_store.delete(key_range).serde()?.await?;
    }

    transaction.commit().await?;

    Ok(())
}

pub(crate) async fn delete_session_messages(session_uuid: Uuid) -> indexed_db_futures::Result<()> {
    let db = {
        let db = DB.read();
        db.as_ref().cloned().expect("Database not initialized")
    };

    let transaction = db
        .transaction(["messages", "multimedia"])
        .with_mode(TransactionMode::Readwrite)
        .build()?;

    let object_store = transaction.object_store("messages")?;
    let object_store_multimedia = transaction.object_store("multimedia")?;

    let key_range = KeyRange::Bound((session_uuid, 0), false, (session_uuid, MAX_SAFE_INTEGER), false);

    if let Some(cursor) = object_store
        .open_cursor()
        .with_query(key_range.clone())
        .serde()?
        .await?
    {
        let stream = cursor.stream_ser::<MessageWrapper>();
        let messages = stream
            .map(|x| x.unwrap())
            .map(|x| x.message)
            .collect::<Vec<crate::Message>>()
            .await;

        for message in messages {
            if let crate::MessageType::Image(uuid) = message.content {
                object_store_multimedia.delete(KeyRange::Only(uuid)).serde()?.await?;
            }
        }
    }

    object_store.delete(key_range).serde()?.await?;

    transaction.commit().await?;

    Ok(())
}

pub(crate) async fn save_multimedia(uuid: Uuid, blob: web_sys::Blob) -> indexed_db_futures::Result<()> {
    let db = {
        let db = DB.read();
        db.as_ref().cloned().expect("Database not initialized")
    };

    let transaction = db
        .transaction("multimedia")
        .with_mode(TransactionMode::Readwrite)
        .build()?;

    let obj_store = transaction.object_store("multimedia")?;

    let obj = Object::new();
    Reflect::set(&obj, &JsValue::from_str("uuid"), &JsValue::from_str(&uuid.to_string()))?;
    Reflect::set(&obj, &JsValue::from_str("blob"), &blob.into())?;

    obj_store.put(obj).build()?.await?;
    transaction.commit().await?;

    Ok(())
}

#[allow(unused)]
pub(crate) async fn remove_multimedia(uuid: Uuid) -> indexed_db_futures::Result<()> {
    let db = {
        let db = DB.read();
        db.as_ref().cloned().expect("Database not initialized")
    };

    let transaction = db
        .transaction("multimedia")
        .with_mode(TransactionMode::Readwrite)
        .build()?;

    let obj_store = transaction.object_store("multimedia")?;

    obj_store.delete(KeyRange::Only(uuid)).serde()?.await?;
    transaction.commit().await?;

    Ok(())
}

#[allow(unused)]
pub(crate) async fn get_multimedia(uuid: Uuid) -> indexed_db_futures::Result<Option<web_sys::Blob>> {
    let db = {
        let db = DB.read();
        db.as_ref().cloned().expect("Database not initialized")
    };

    let transaction = db
        .transaction("multimedia")
        .with_mode(TransactionMode::Readonly)
        .build()?;

    let obj_store = transaction.object_store("multimedia")?;

    let val: Option<JsValue> = obj_store.get(&JsValue::from_str(&uuid.to_string())).build()?.await?;

    Ok(val
        .and_then(|x| Reflect::get(&x, &JsValue::from_str("blob")).ok())
        .and_then(|x| x.dyn_into::<web_sys::Blob>().ok()))
}
