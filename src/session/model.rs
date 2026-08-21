use crate::operator::model::*;
use crate::shared::assets;

use dioxus::prelude::*;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "t", content = "c")]
pub(crate) enum MessageType {
    #[serde(rename = "a")]
    Text(String),

    #[serde(rename = "b")]
    Image(Uuid),

    #[serde(rename = "c")]
    HorizontalBreak,

    #[serde(rename = "d")]
    State(String),

    #[serde(rename = "e")]
    StateWithHorizontalLine(String),

    #[serde(rename = "f")]
    Sticker(assets::stickers::Stickers),
}

impl MessageType {
    pub(crate) fn is_text_or_image(&self) -> bool {
        matches!(self, MessageType::Text(_))
            || matches!(self, MessageType::Image(_))
            || matches!(self, MessageType::Sticker(_))
    }
}

pub(crate) type Reaction = (assets::Emoji, Vec<Option<Uuid>>);

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct Message {
    sender: Sender,
    #[serde(rename = "c")]
    content: MessageType,
    #[serde(skip_serializing)]
    #[serde(default)]
    animation: bool,
    #[serde(default = "Vec::new")]
    #[serde(rename = "r")]
    reactions: Vec<Reaction>,
}

impl Message {
    /// 新建一条消息（用于发送时）。
    pub(crate) fn new(sender: Sender, content: MessageType) -> Message {
        Message {
            sender,
            content,
            animation: true,
            reactions: vec![],
        }
    }

    pub(crate) fn sender(&self) -> &Sender {
        &self.sender
    }

    pub(crate) fn content(&self) -> &MessageType {
        &self.content
    }

    pub(crate) fn animation(&self) -> bool {
        self.animation
    }

    pub(crate) fn set_animation(&mut self, animation: bool) {
        self.animation = animation;
    }

    pub(crate) fn reactions(&self) -> &Vec<Reaction> {
        &self.reactions
    }

    /// 为消息添加一个 Reaction。
    ///
    /// 注意，当已经有这个 Reaction，仅把不在名单内的干员加进去。
    pub(crate) fn append_reaction(&mut self, reaction: Reaction) {
        if let Some(index) = self.reactions.iter().position(|x| x.0 == reaction.0) {
            let ids_unlisted = reaction
                .1
                .iter()
                .filter(|x| !self.reactions[index].1.contains(x))
                .collect::<Vec<_>>();
            self.reactions[index].1.extend(ids_unlisted);
        } else {
            self.reactions.push(reaction);
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct Session {
    session_name: String,
    avatar: Avatar,
    participants_ids: Vec<Uuid>,
}

impl Session {
    pub(crate) fn new(session_name: String, avatar: Avatar, participants_ids: Vec<Uuid>) -> Session {
        Session {
            session_name,
            avatar,
            participants_ids,
        }
    }

    pub(crate) fn refresh_avatar(&mut self, operators: &[(Uuid, Operator)]) {
        self.avatar = match self.participants_ids.as_slice() {
            [participant_id] => operators
                .iter()
                .find(|x| x.0 == *participant_id)
                .map(|x| &x.1)
                .filter(|operator| operator.activity())
                .map(|operator| operator.get_avatar_originally())
                .unwrap_or_default(),
            _ => Avatar::None,
        };
    }

    pub(crate) fn session_name(&self) -> &String {
        &self.session_name
    }

    pub(crate) fn rename(&mut self, new_name: String) {
        self.session_name = new_name;
    }

    pub(crate) fn avatar(&self) -> Asset {
        self.avatar.to_asset_session()
    }

    pub(crate) fn participants_ids(&self) -> &Vec<Uuid> {
        &self.participants_ids
    }

    pub(crate) fn set_participants_ids(&mut self, ids: Vec<Uuid>) {
        self.participants_ids = ids;
    }

    pub(crate) fn deactivate_operator_helper(&mut self, operator_uuid: Uuid, operators: &[(Uuid, Operator)]) {
        self.participants_ids.retain(|x| *x != operator_uuid);
        self.refresh_avatar(operators);
    }
}
