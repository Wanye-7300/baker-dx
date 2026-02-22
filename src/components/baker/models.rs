use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

fn new_uuid() -> String {
    Uuid::new_v4().to_string()
}

#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct Contact {
    pub id: String,
    pub unread_count: usize,
    #[serde(default)]
    pub chat_head_style: ChatHeadStyle,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub avatar_url: String,
    #[serde(default)]
    pub participant_ids: Vec<String>,
    #[serde(default)]
    pub is_group: bool,
}

#[derive(Clone, PartialEq, Debug, Serialize, Deserialize, Default)]
pub enum MessageKind {
    #[default]
    Normal,
    Status,
}

#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct Message {
    pub id: String,
    pub sender_id: String,
    pub content: String,
    #[serde(default)]
    pub kind: MessageKind,
    #[serde(default)]
    pub animate: bool,
}

#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct Operator {
    pub id: String,
    pub name: String,
    pub avatar_url: String,
}

#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct UserProfile {
    #[serde(default = "new_uuid")]
    pub id: String,
    pub name: String,
    pub avatar_url: String,
}

#[derive(Clone, PartialEq, Debug, Serialize, Deserialize, Default)]
pub enum ChatHeadStyle {
    #[default]
    Default,
    Alt,
}

#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub enum BackgroundMode {
    DotDark,
    DotLight,
    CustomColor,
    CustomImage,
}

#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct BackgroundSettings {
    pub mode: BackgroundMode,
    pub custom_color: String,
    pub custom_image: String,
}

impl Default for BackgroundSettings {
    fn default() -> Self {
        Self {
            mode: BackgroundMode::DotDark,
            custom_color: "#1a1a1a".to_string(),
            custom_image: "".to_string(),
        }
    }
}

impl Default for UserProfile {
    fn default() -> Self {
        Self {
            id: new_uuid(),
            name: "Me".to_string(),
            avatar_url: "".to_string(),
        }
    }
}

#[derive(Clone, PartialEq, Debug, Serialize, Deserialize, Default)]
pub struct AppState {
    pub user_profile: UserProfile,
    pub contacts: Vec<Contact>,
    pub messages: HashMap<String, Vec<Message>>,
    pub operators: Vec<Operator>,
    #[serde(default)]
    pub background: BackgroundSettings,
    #[serde(default)]
    pub update_snooze_date: Option<String>,
}
