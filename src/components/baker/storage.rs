use crate::components::baker::models::{
    AppState, BackgroundSettings, ChatHeadStyle, Contact, Message, MessageKind, Operator,
    UserProfile,
};
use serde::Deserialize;
use std::collections::HashMap;
use uuid::Uuid;

#[cfg(target_arch = "wasm32")]
use gloo_storage::{LocalStorage, Storage};

#[cfg(not(target_arch = "wasm32"))]
use std::fs;

#[derive(Deserialize)]
struct LegacyContact {
    pub id: usize,
    pub unread_count: usize,
    #[serde(default)]
    pub chat_head_style: ChatHeadStyle,
}

#[derive(Deserialize)]
struct LegacyMessage {
    pub id: usize,
    pub sender_id: usize,
    pub content: String,
    pub timestamp: String,
    #[serde(default)]
    pub animate: bool,
}

#[derive(Deserialize)]
struct LegacyOperator {
    pub id: usize,
    pub name: String,
    pub avatar_url: String,
}

#[derive(Deserialize)]
struct LegacyUserProfile {
    pub name: String,
    pub avatar_url: String,
}

#[derive(Deserialize)]
struct LegacyAppState {
    pub user_profile: LegacyUserProfile,
    pub contacts: Vec<LegacyContact>,
    pub messages: HashMap<usize, Vec<LegacyMessage>>,
    pub operators: Vec<LegacyOperator>,
    #[serde(default)]
    pub background: BackgroundSettings,
}

fn migrate_legacy_state(legacy: LegacyAppState) -> AppState {
    let mut id_map: HashMap<usize, String> = HashMap::new();
    let user_id = Uuid::new_v4().to_string();

    let operators = legacy
        .operators
        .into_iter()
        .map(|op| {
            let new_id = Uuid::new_v4().to_string();
            id_map.insert(op.id, new_id.clone());
            Operator {
                id: new_id,
                name: op.name,
                avatar_url: op.avatar_url,
            }
        })
        .collect::<Vec<_>>();
    let operator_map = operators
        .iter()
        .map(|op| (op.id.clone(), (op.name.clone(), op.avatar_url.clone())))
        .collect::<HashMap<_, _>>();

    let contacts = legacy
        .contacts
        .into_iter()
        .map(|contact| {
            let new_id = id_map.get(&contact.id).cloned().unwrap_or_else(|| {
                let new_id = Uuid::new_v4().to_string();
                id_map.insert(contact.id, new_id.clone());
                new_id
            });
            let (name, avatar) = operator_map
                .get(&new_id)
                .cloned()
                .unwrap_or_else(|| ("".to_string(), "".to_string()));
            Contact {
                id: new_id.clone(),
                unread_count: contact.unread_count,
                chat_head_style: contact.chat_head_style,
                name,
                avatar_url: avatar,
                participant_ids: vec![new_id],
                is_group: false,
            }
        })
        .collect::<Vec<_>>();

    let mut messages: HashMap<String, Vec<Message>> = HashMap::new();
    for (legacy_contact_id, list) in legacy.messages {
        let contact_id = id_map.get(&legacy_contact_id).cloned().unwrap_or_else(|| {
            let new_id = Uuid::new_v4().to_string();
            id_map.insert(legacy_contact_id, new_id.clone());
            new_id
        });
        let converted = list
            .into_iter()
            .map(|msg| {
                let LegacyMessage {
                    id: _legacy_id,
                    sender_id,
                    content,
                    timestamp: _timestamp,
                    animate,
                } = msg;
                let sender_id = if sender_id == 0 {
                    user_id.clone()
                } else {
                    id_map.get(&sender_id).cloned().unwrap_or_else(|| {
                        let new_id = Uuid::new_v4().to_string();
                        id_map.insert(sender_id, new_id.clone());
                        new_id
                    })
                };
                Message {
                    id: Uuid::new_v4().to_string(),
                    sender_id,
                    content,
                    kind: MessageKind::Normal,
                    animate,
                }
            })
            .collect::<Vec<_>>();
        messages.insert(contact_id, converted);
    }

    AppState {
        user_profile: UserProfile {
            id: user_id,
            name: legacy.user_profile.name,
            avatar_url: legacy.user_profile.avatar_url,
        },
        contacts,
        messages,
        operators,
        background: legacy.background,
        update_snooze_date: None,
    }
}

pub fn load_state() -> AppState {
    #[cfg(target_arch = "wasm32")]
    {
        if let Ok(state) = LocalStorage::get("baker_dx_state") {
            return state;
        }
        if let Ok(raw) = LocalStorage::get::<String>("baker_dx_state") {
            if let Ok(state) = serde_json::from_str::<AppState>(&raw) {
                return state;
            }
            if let Ok(legacy) = serde_json::from_str::<LegacyAppState>(&raw) {
                return migrate_legacy_state(legacy);
            }
        }
        AppState::default()
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        if let Ok(content) = fs::read_to_string("baker_dx_state.json") {
            if let Ok(state) = serde_json::from_str(&content) {
                return state;
            }
            if let Ok(legacy) = serde_json::from_str::<LegacyAppState>(&content) {
                return migrate_legacy_state(legacy);
            }
        }
        AppState::default()
    }
}

pub fn save_state(state: &AppState) {
    #[cfg(target_arch = "wasm32")]
    {
        let _ = LocalStorage::set("baker_dx_state", state);
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        if let Ok(content) = serde_json::to_string_pretty(state) {
            let _ = fs::write("baker_dx_state.json", content);
        }
    }
}
