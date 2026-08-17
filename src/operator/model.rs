use crate::shared::assets::{CHARACTERS_AVATARS, ICON_ROUND_SNS_ENDFIELD_GROUP_A};

use dioxus::prelude::*;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// 表示干员头像。
#[derive(Clone, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[repr(u8)]
pub(crate) enum Avatar {
    #[default]
    None = 0x00,
    Preset(String),
    Uploaded(Uuid),
}

impl Avatar {
    pub(crate) fn to_asset_operator(&self) -> Asset {
        match &self {
            Avatar::None => CHARACTERS_AVATARS["none"],
            Avatar::Preset(preset) => CHARACTERS_AVATARS[preset.as_str()],
            Avatar::Uploaded(_uuid) => todo!(),
        }
    }

    pub(crate) fn to_asset_session(&self) -> Asset {
        match &self {
            Avatar::None => ICON_ROUND_SNS_ENDFIELD_GROUP_A,
            Avatar::Preset(preset) => CHARACTERS_AVATARS[preset.as_str()],
            Avatar::Uploaded(_uuid) => todo!(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct Operator {
    name: String,
    avatar: Avatar,
    #[serde(default = "default_operator_active")]
    activity: bool,
}

impl Operator {
    pub(crate) fn new(name: String, avatar: Avatar) -> Operator {
        Operator {
            name,
            avatar,
            activity: true,
        }
    }

    pub(crate) fn name(&self) -> &String {
        &self.name
    }

    pub(crate) fn rename(&mut self, name: String) {
        self.name = name;
    }

    pub(crate) fn get_avatar_originally(&self) -> Avatar {
        self.avatar.clone()
    }

    pub(crate) fn activity(&self) -> bool {
        self.activity
    }

    #[allow(unused)]
    pub(crate) fn activate(&mut self) {
        self.activity = true;
    }

    pub(crate) fn deactivate(&mut self) {
        self.activity = false;
    }
}

fn default_operator_active() -> bool {
    true
}

#[derive(Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Debug, Default)]
#[serde(tag = "st", content = "sc")]
pub(crate) enum Sender {
    #[default]
    #[serde(rename = "end")]
    Endministrator,

    #[serde(rename = "o")]
    Others(Uuid),

    /// 分隔符等
    #[serde(rename = "n")]
    None,
}

impl Sender {
    pub(crate) fn from_optional_uuid(uuid: Option<Uuid>) -> Self {
        match uuid {
            Some(uuid) => Self::Others(uuid),
            None => Self::Endministrator,
        }
    }

    pub(crate) fn avatar_should_on_left(&self) -> bool {
        matches!(self, Self::Others(_))
    }
}
