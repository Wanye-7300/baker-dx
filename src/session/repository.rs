use std::collections;
use std::sync::Arc;

use super::model::*;
use crate::operator::model::*;
use crate::shared::database;
use crate::shared::utils;

use dioxus::signals::ReadableExt;
use dioxus::signals::Signal;
use dioxus::signals::WritableExt;
use futures::lock::Mutex;
use uuid::Uuid;

const SESSION_KEY: &str = "session";

#[derive(Debug)]
pub(crate) struct MessageRepository {
    session_uuid: Option<Uuid>,
    messages: Option<collections::BTreeMap<u64, Message>>,
    mutex: Arc<Mutex<()>>,
}

impl MessageRepository {
    pub(crate) fn new() -> MessageRepository {
        MessageRepository {
            session_uuid: None,
            messages: None,
            mutex: Arc::new(Mutex::new(())),
        }
    }

    pub(crate) fn current_session(&self) -> Option<Uuid> {
        self.session_uuid
    }

    pub(crate) async fn select(mut repository: Signal<Self>, session_uuid: Uuid) -> anyhow::Result<()> {
        let messages = database::get_messages(session_uuid)
            .await
            .map_err(|err| anyhow::anyhow!("IDB: Cannot get messages: {err}"))?;

        {
            let mut repository = repository.write();
            repository.session_uuid = Some(session_uuid);
            repository.messages = Some(messages);
        }

        Ok(())
    }

    pub(crate) async fn push(mut repository: Signal<Self>, message: Message) -> anyhow::Result<()> {
        let lock = {
            let repository = repository.read();
            repository.mutex.clone()
        };

        let _guard = lock.lock().await;

        let (session_uuid, next_index) = {
            let repository = repository.read();

            let session_uuid = repository
                .session_uuid
                .ok_or_else(|| anyhow::anyhow!("MessagesRepository.session_uuid is `None`"))?;

            if repository.messages.is_none() {
                anyhow::bail!("MessagesRepository.messages is `None`");
            }

            let next_index = repository
                .messages
                .as_ref()
                .unwrap()
                .last_key_value()
                .map(|x| *x.0 + 1)
                .unwrap_or(0);

            (session_uuid, next_index)
        };

        let wrapper = database::MessageWrapper {
            session_uuid,
            message_id: next_index,
            message: message.clone(),
        };

        database::put_messages(vec![wrapper])
            .await
            .map_err(|err| anyhow::anyhow!("IDB: Error occurred when pushing: {err}"))?;

        {
            let mut repository = repository.write();
            if repository.current_session() == Some(session_uuid) {
                repository.messages.as_mut().unwrap().insert(next_index, message);
            }
        }

        Ok(())
    }

    pub(crate) async fn insert(mut repository: Signal<Self>, message: Message, message_id: u64) -> anyhow::Result<()> {
        let lock = {
            let repository = repository.read();
            repository.mutex.clone()
        };

        let _guard = lock.lock().await;

        let session_uuid = {
            let repository = repository.read();

            let session_uuid = repository
                .session_uuid
                .ok_or_else(|| anyhow::anyhow!("MessagesRepository.session_uuid is `None`"))?;

            if repository.messages.is_none() {
                anyhow::bail!("MessagesRepository.messages is `None`");
            }

            session_uuid
        };

        let wrapper = database::MessageWrapper {
            session_uuid,
            message_id,
            message: message.clone(),
        };

        let need_to_move_messages = database::insert_message(wrapper)
            .await
            .map_err(|err| anyhow::anyhow!("IDB: Error occurred when inserting: {err}"))?;

        {
            let mut repository = repository.write();

            let messages = repository.messages.as_mut().unwrap();

            if need_to_move_messages {
                let split_off = messages.split_off(&message_id);
                for (idx, msg) in split_off {
                    messages.insert(idx + 1, msg);
                }
                messages.insert(message_id, message);
            } else {
                messages.insert(message_id, message);
            }
        }

        Ok(())
    }

    pub(crate) async fn delete(mut repository: Signal<Self>, message_id: u64) -> anyhow::Result<()> {
        let lock = {
            let repository = repository.read();
            repository.mutex.clone()
        };

        let _guard = lock.lock().await;

        let session_uuid = {
            let repository = repository.read();

            let session_uuid = repository
                .session_uuid
                .ok_or_else(|| anyhow::anyhow!("MessagesRepository.session_uuid is `None`"))?;

            if repository.messages.is_none() {
                anyhow::bail!("MessagesRepository.messages is `None`");
            }

            if !repository.messages.as_ref().unwrap().contains_key(&message_id) {
                anyhow::bail!("The given message_id is not exist in MessagesRepository.messages");
            }

            session_uuid
        };

        database::delete_message(session_uuid, message_id)
            .await
            .map_err(|err| anyhow::anyhow!("IDB: Error occurred when deleting: {err}"))?;

        {
            let mut repository = repository.write();
            repository.messages.as_mut().unwrap().remove(&message_id);
        }

        Ok(())
    }

    pub(crate) async fn modify(mut repository: Signal<Self>, message_id: u64, message: Message) -> anyhow::Result<()> {
        let lock = {
            let repository = repository.read();
            repository.mutex.clone()
        };

        let _guard = lock.lock().await;

        let session_uuid = {
            let repository = repository.read();

            let session_uuid = repository
                .session_uuid
                .ok_or_else(|| anyhow::anyhow!("MessagesRepository.session_uuid is `None`"))?;

            if repository.messages.is_none() {
                anyhow::bail!("MessagesRepository.messages is `None`");
            }

            if !repository.messages.as_ref().unwrap().contains_key(&message_id) {
                anyhow::bail!("The given message_id is not exist in MessagesRepository.messages");
            }

            session_uuid
        };

        let wrapper = database::MessageWrapper {
            session_uuid,
            message_id,
            message: message.clone(),
        };

        database::modify_message(wrapper)
            .await
            .map_err(|err| anyhow::anyhow!("IDB: Error occurred when modifying: {err}"))?;

        {
            let mut repository = repository.write();
            repository.messages.as_mut().unwrap().insert(message_id, message);
        }

        Ok(())
    }

    pub(crate) async fn append_reaction(
        mut repository: Signal<Self>,
        message_id: u64,
        reaction: Reaction,
    ) -> anyhow::Result<()> {
        let lock = {
            let repository = repository.read();
            repository.mutex.clone()
        };

        let _guard = lock.lock().await;

        let (session_uuid, mut message) = {
            let repository = repository.read();

            let session_uuid = repository
                .session_uuid
                .ok_or_else(|| anyhow::anyhow!("MessagesRepository.session_uuid is `None`"))?;

            let messages = repository
                .messages
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("MessagesRepository.messages is `None`"))?;

            let Some(message) = messages.get(&message_id).cloned() else {
                anyhow::bail!("The given message_id is not exist in MessagesRepository.messages");
            };

            (session_uuid, message)
        };

        message.append_reaction(reaction);

        let wrapper = database::MessageWrapper {
            session_uuid,
            message_id,
            message: message.clone(),
        };

        database::modify_message(wrapper)
            .await
            .map_err(|err| anyhow::anyhow!("IDB: Error occurred when modifying: {err}"))?;

        {
            let mut repository = repository.write();
            repository.messages.as_mut().unwrap().insert(message_id, message);
        }

        Ok(())
    }

    pub(crate) fn iterator(&self) -> anyhow::Result<impl Iterator<Item = (u64, &Message)>> {
        if let Some(messages) = &self.messages {
            Ok(messages.iter().map(|(k, v)| (*k, v)))
        } else {
            anyhow::bail!("Message.Repository is `None`");
        }
    }

    pub(crate) fn clear(&mut self) {
        self.messages = None;
        self.session_uuid = None;
    }
}

#[derive(Clone, Debug)]
pub(crate) struct SessionRepository {
    sessions: Vec<(Uuid, Session)>,
}

impl SessionRepository {
    fn save(&self) -> anyhow::Result<()> {
        utils::set_item(SESSION_KEY, &self.sessions)
    }

    pub(crate) fn from_local_storage_or_default() -> anyhow::Result<SessionRepository> {
        let sessions = utils::get_item_or_default(SESSION_KEY, Vec::<(Uuid, Session)>::new)?;

        Ok(SessionRepository { sessions })
    }

    pub(crate) fn get(&self, session_uuid: &Uuid) -> anyhow::Result<&Session> {
        self.sessions
            .iter()
            .find(|(uuid, _)| uuid == session_uuid)
            .map(|(_, session)| session)
            .ok_or(anyhow::anyhow!(
                "Uuid `{session_uuid}` is not found in SessionRepository.sessions"
            ))
    }

    pub(crate) fn get_mut(&mut self, session_uuid: &Uuid) -> anyhow::Result<&mut Session> {
        self.sessions
            .iter_mut()
            .find(|(uuid, _)| uuid == session_uuid)
            .map(|(_, session)| session)
            .ok_or(anyhow::anyhow!(
                "Uuid `{session_uuid}` is not found in SessionRepository.sessions"
            ))
    }

    pub(crate) fn push_session(&mut self, session: Session) -> anyhow::Result<Uuid> {
        let session_uuid = Uuid::new_v4();

        self.sessions.push((session_uuid, session));
        self.save()?;

        Ok(session_uuid)
    }

    pub(crate) async fn delete_session(mut repository: Signal<Self>, session_uuid: Uuid) -> anyhow::Result<()> {
        {
            let mut repository = repository.write();

            if repository.sessions.iter().position(|x| x.0 == session_uuid).is_none() {
                anyhow::bail!("Uuid `{session_uuid}` is not found in SessionRepository.sessions");
            }

            repository.sessions.retain(|x| x.0 != session_uuid);
            repository.save()?;
        }

        database::delete_session_messages(session_uuid)
            .await
            .map_err(|err| anyhow::anyhow!("IDB: Failed to delete session: {err}"))?;

        Ok(())
    }

    pub(crate) fn deactivate_operator_helper(
        &mut self,
        operator_uuid: Uuid,
        operators: &[(Uuid, Operator)],
    ) -> anyhow::Result<()> {
        for (_, session) in &mut self.sessions {
            session.deactivate_operator_helper(operator_uuid, operators);
        }

        self.save()?;

        Ok(())
    }

    pub(crate) fn iterator(&self) -> impl Iterator<Item = (Uuid, &Session)> {
        self.sessions.iter().rev().map(|(uuid, session)| (*uuid, session))
    }
}
