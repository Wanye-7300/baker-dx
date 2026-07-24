use std::collections;

use dioxus::prelude::*;

use futures::StreamExt;
use indexed_db_futures::prelude::*;
use indexed_db_futures::transaction::TransactionMode;
use indexed_db_futures::{database::Database, KeyRange};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

const MAX_SAFE_INTEGER: u64 = 9007199254740991;

static DB: GlobalSignal<Option<Database>> = Signal::global(|| None);

#[derive(Serialize, Deserialize)]
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
        .transaction("messages")
        .with_mode(TransactionMode::Readwrite)
        .build()?;

    let obj_store = transaction.object_store("messages")?;

    let key_range = KeyRange::Only((session_uuid, message_id));

    obj_store.delete(key_range).serde()?.await?;

    transaction.commit().await?;

    Ok(())
}
