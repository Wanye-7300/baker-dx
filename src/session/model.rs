use std::str::FromStr;

use crate::session::repository::MessageRepository;
use crate::shared::assets::{
    self, ALL_MISSION_ICON_GRAY, CHAR_MISSION_ICON_GRAY, FAC_MISSION_ICON_GRAY, MAIN_MISSION_ICON_GRAY,
    MISC_MISSION_ICON_GRAY,
};
use crate::{operator::model::*, shared::assets::ACTIVITY_MISSION_ICON_GRAY};

use dioxus::prelude::*;
use serde::{Deserialize, Serialize};
use strum::{EnumString, VariantNames};
use uuid::Uuid;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, EnumString, VariantNames, strum::Display)]
#[repr(u8)]
pub(crate) enum TaskImportance {
    Critical,
    Important,
    Minor,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, EnumString, VariantNames, strum::Display)]
#[repr(u8)]
pub(crate) enum TaskType {
    Activity,
    All,
    Character,

    /// 谁知道这代表什么？
    Fac,
    Main,
    Misc,
}

impl TaskType {
    pub(crate) fn as_asset(&self) -> Asset {
        match self {
            TaskType::Activity => ACTIVITY_MISSION_ICON_GRAY,
            TaskType::All => ALL_MISSION_ICON_GRAY,
            TaskType::Character => CHAR_MISSION_ICON_GRAY,
            TaskType::Fac => FAC_MISSION_ICON_GRAY,
            TaskType::Main => MAIN_MISSION_ICON_GRAY,
            TaskType::Misc => MISC_MISSION_ICON_GRAY,
        }
    }
}

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

    #[serde(rename = "g")]
    Task {
        title: String,
        location: String,
        task_importance: TaskImportance,
        task_type: TaskType,
        completed: bool,
    },
}

impl MessageType {
    pub(crate) fn is_text_or_image(&self) -> bool {
        matches!(self, MessageType::Text(_))
            || matches!(self, MessageType::Image(_))
            || matches!(self, MessageType::Sticker(_))
    }

    pub(crate) unsafe fn as_text_mut_unchecked(&mut self) -> &mut String {
        match self {
            MessageType::Text(text) => text,
            _ => unsafe { std::hint::unreachable_unchecked() },
        }
    }

    pub(crate) unsafe fn as_task_mut_unchecked(
        &mut self,
    ) -> (&mut String, &mut String, &mut TaskImportance, &mut TaskType, &mut bool) {
        match self {
            MessageType::Task {
                title,
                location,
                task_importance,
                task_type,
                completed,
            } => (title, location, task_importance, task_type, completed),
            _ => unsafe { std::hint::unreachable_unchecked() },
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct Reaction {
    #[serde(rename = "u")]
    uuid: Uuid,
    #[serde(rename = "e")]
    emoji: assets::Emoji,
    #[serde(rename = "s")]
    senders: Vec<Option<Uuid>>,
    #[serde(skip_serializing)]
    #[serde(default)]
    animation: bool,
}

impl Reaction {
    pub(crate) fn new(emoji: assets::Emoji, senders: Vec<Option<Uuid>>) -> Reaction {
        Reaction {
            uuid: Uuid::new_v4(),
            emoji,
            senders,
            animation: false,
        }
    }

    pub(crate) fn with_animation(mut self) -> Reaction {
        self.animation = true;
        self
    }

    pub(crate) fn uuid(&self) -> Uuid {
        self.uuid
    }

    pub(crate) fn emoji(&self) -> assets::Emoji {
        self.emoji
    }

    pub(crate) fn senders(&self) -> &Vec<Option<Uuid>> {
        &self.senders
    }

    #[allow(unused)]
    pub(crate) fn clear_senders(&mut self) -> Vec<Option<Uuid>> {
        std::mem::take(&mut self.senders)
    }

    #[allow(unused)]
    pub(crate) fn push_sender(&mut self, sender: Option<Uuid>) {
        self.senders.push(sender);
    }

    pub(crate) fn animation(&self) -> bool {
        self.animation
    }

    pub(crate) fn set_animation(&mut self, animation: bool) {
        self.animation = animation
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct Message {
    sender: Sender,
    #[serde(rename = "c")]
    content: MessageType,
    #[serde(skip_serializing)]
    #[serde(default)]
    animation: bool,
    #[serde(skip_serializing)]
    #[serde(default)]
    input_animation: bool,
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
            input_animation: false,
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

    pub(crate) fn input_animation(&self) -> bool {
        self.input_animation
    }

    pub(crate) fn set_input_animation(&mut self, input_animation: bool) {
        self.input_animation = input_animation
    }

    pub(crate) fn reactions(&self) -> &Vec<Reaction> {
        &self.reactions
    }

    /// 为消息添加一个 Reaction。
    ///
    /// 注意，当已经有这个 Reaction，仅把不在名单内的干员加进去。
    pub(crate) fn append_reaction(&mut self, reaction: Reaction) {
        if let Some(index) = self.reactions.iter().position(|x| x.emoji == reaction.emoji) {
            let ids_unlisted = reaction
                .senders
                .iter()
                .filter(|x| !self.reactions[index].senders.contains(x))
                .collect::<Vec<_>>();
            self.reactions[index].senders.extend(ids_unlisted);
        } else {
            self.reactions.push(reaction);
        }
    }

    pub(crate) fn clear_reaction(&mut self) -> Vec<Reaction> {
        std::mem::take(&mut self.reactions)
    }

    pub(crate) fn get_settings_vm(
        &self,
        repository: Signal<MessageRepository>,
        session_uuid: Uuid,
        message_id: u64,
    ) -> crate::shared::setting::SettingViewModel {
        use crate::shared::setting::*;

        SettingViewModel::new(
            format!("属性 {}:{}", session_uuid, message_id),
            match self.content() {
                MessageType::Task {
                    title,
                    location,
                    task_importance,
                    task_type,
                    completed,
                } => SettingItemPage::new()
                    .with_child(SettingItem::new(
                        "标题".to_string(),
                        None,
                        SettingItemType::Str {
                            value: title.to_string(),
                        },
                        Some(EventHandler::new(move |val| {
                            let mut message = repository.read().get(message_id).unwrap().clone();

                            async move {
                                match val {
                                    SettingItemValue::Str(value) => {
                                        *unsafe { message.content.as_task_mut_unchecked() }.0 = value;
                                        MessageRepository::modify(repository, message_id, message)
                                            .await
                                            .unwrap();
                                    }
                                    _ => unsafe {
                                        std::hint::unreachable_unchecked();
                                    },
                                }
                            }
                        })),
                    ))
                    .with_child(SettingItem::new(
                        "地点".to_string(),
                        None,
                        SettingItemType::Str {
                            value: location.to_string(),
                        },
                        Some(EventHandler::new(move |val| {
                            let mut message = repository.read().get(message_id).unwrap().clone();

                            async move {
                                match val {
                                    SettingItemValue::Str(value) => {
                                        *unsafe { message.content.as_task_mut_unchecked() }.1 = value;
                                        MessageRepository::modify(repository, message_id, message)
                                            .await
                                            .unwrap();
                                    }
                                    _ => unsafe {
                                        std::hint::unreachable_unchecked();
                                    },
                                }
                            }
                        })),
                    ))
                    .with_child(SettingItem::new(
                        "重要性".to_string(),
                        None,
                        SettingItemType::Selection {
                            selections: TaskImportance::VARIANTS.iter().map(|x| (*x).to_owned()).collect(),
                            value: task_importance.to_string(),
                        },
                        Some(EventHandler::new(move |val| {
                            let mut message = repository.read().get(message_id).unwrap().clone();

                            async move {
                                match val {
                                    SettingItemValue::Selection(value) => {
                                        *unsafe { message.content.as_task_mut_unchecked() }.2 =
                                            TaskImportance::from_str(&value).unwrap();
                                        MessageRepository::modify(repository, message_id, message)
                                            .await
                                            .unwrap();
                                    }
                                    _ => unsafe {
                                        std::hint::unreachable_unchecked();
                                    },
                                }
                            }
                        })),
                    ))
                    .with_child(SettingItem::new(
                        "任务类型".to_string(),
                        Some("控制任务的图标。缩写稍微有点晦涩难懂，作者没搞明白是什么意思。".to_string()),
                        SettingItemType::Selection {
                            selections: TaskType::VARIANTS.iter().map(|x| (*x).to_owned()).collect(),
                            value: task_type.to_string(),
                        },
                        Some(EventHandler::new(move |val| {
                            let mut message = repository.read().get(message_id).unwrap().clone();

                            async move {
                                match val {
                                    SettingItemValue::Selection(value) => {
                                        *unsafe { message.content.as_task_mut_unchecked() }.3 =
                                            TaskType::from_str(&value).unwrap();
                                        MessageRepository::modify(repository, message_id, message)
                                            .await
                                            .unwrap();
                                    }
                                    _ => unsafe {
                                        std::hint::unreachable_unchecked();
                                    },
                                }
                            }
                        })),
                    ))
                    .with_child(SettingItem::new(
                        "已完成".to_string(),
                        None,
                        SettingItemType::Bool { value: *completed },
                        Some(EventHandler::new(move |val| {
                            let mut message = repository.read().get(message_id).unwrap().clone();

                            async move {
                                match val {
                                    SettingItemValue::Bool(value) => {
                                        *unsafe { message.content.as_task_mut_unchecked() }.4 = value;
                                        MessageRepository::modify(repository, message_id, message)
                                            .await
                                            .unwrap();
                                    }
                                    _ => unsafe {
                                        std::hint::unreachable_unchecked();
                                    },
                                }
                            }
                        })),
                    )),
                _ => SettingItemPage::new().with_child(SettingItem::new(
                    "没有设置".to_string(),
                    Some("如果要修改内容，请使用**修改模式**。".to_string()),
                    SettingItemType::Empty,
                    None,
                )),
            },
            false,
        )
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
