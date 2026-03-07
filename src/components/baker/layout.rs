use crate::components::baker::chat_area::{ChatArea, PendingTyping, ReplayTypingPhase};
use crate::components::baker::modals::{
    NewChatModal, NewChatSelection, OpsSelection, ProfileModal, ReplayIntervalMode, ReplaySettings,
    ReplaySettingsModal, TutorialModal, UpdateAvailableModal,
};
use crate::components::baker::models::{
    BackgroundMode, ChatHeadStyle, Contact, Message, MessageKind, MessageReaction,
};
use crate::components::baker::settings::SettingsPage;
use crate::components::baker::sidebar::Sidebar;
use chrono::Utc;
use dioxus::prelude::*;
#[cfg(target_arch = "wasm32")]
use gloo_timers::future::TimeoutFuture;
use reqwest::Client;
use serde::Deserialize;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Duration;
#[cfg(not(target_arch = "wasm32"))]
use tokio::time::sleep;
use uuid::Uuid;

const MESSAGE_SOUND: Asset = asset!("/assets/sound/message.mp3");
const MESSAGE_SELF_SOUND: Asset = asset!("/assets/sound/message-self.mp3");

fn play_message_sound(is_self: bool) {
    let sound_src = if is_self {
        MESSAGE_SELF_SOUND.to_string()
    } else {
        MESSAGE_SOUND.to_string()
    };
    spawn(async move {
        let script = format!(
            r#"
            const audio = new Audio("{sound_src}");
            audio.volume = 0.5;
            audio.play();
        "#
        );
        let _ = document::eval(&script).await;
    });
}

#[derive(Clone, PartialEq)]
struct UpdateInfo {
    version: String,
    url: String,
}

#[derive(Deserialize)]
pub(super) struct RepoConfig {
    pub(super) owner: String,
    pub(super) repo: String,
}

#[derive(Deserialize)]
struct ReleaseResponse {
    tag_name: String,
    html_url: String,
}

#[derive(Clone, PartialEq)]
struct ReplayContext {
    contact_id: String,
    prev_sender_id: Option<String>,
}

#[cfg(target_arch = "wasm32")]
async fn sleep_ms(ms: u64) {
    TimeoutFuture::new(ms.min(u32::MAX as u64) as u32).await;
}

#[cfg(not(target_arch = "wasm32"))]
async fn sleep_ms(ms: u64) {
    sleep(Duration::from_millis(ms)).await;
}

fn schedule_animate_off_in_state(
    mut app_state: Signal<crate::components::baker::models::AppState>,
    contact_id: String,
    msg_id: String,
) {
    spawn(async move {
        sleep_ms(220).await;
        if let Some(msgs) = app_state.write().messages.get_mut(&contact_id) {
            if let Some(msg) = msgs.iter_mut().find(|m| m.id == msg_id) {
                msg.animate = false;
            }
        }
    });
}

fn schedule_animate_off_in_list(mut list: Signal<Vec<Message>>, msg_id: String) {
    spawn(async move {
        sleep_ms(220).await;
        list.with_mut(|msgs| {
            if let Some(msg) = msgs.iter_mut().find(|m| m.id == msg_id) {
                msg.animate = false;
            }
        });
    });
}

fn schedule_reaction_animate_off_in_state(
    mut app_state: Signal<crate::components::baker::models::AppState>,
    contact_id: String,
    msg_id: String,
) {
    spawn(async move {
        sleep_ms(220).await;
        if let Some(msgs) = app_state.write().messages.get_mut(&contact_id) {
            if let Some(msg) = msgs.iter_mut().find(|m| m.id == msg_id) {
                msg.animate_reactions = false;
            }
        }
    });
}

fn schedule_reaction_animate_off_in_list(mut list: Signal<Vec<Message>>, msg_id: String) {
    spawn(async move {
        sleep_ms(220).await;
        list.with_mut(|msgs| {
            if let Some(msg) = msgs.iter_mut().find(|m| m.id == msg_id) {
                msg.animate_reactions = false;
            }
        });
    });
}

pub fn is_remote_newer(local: &str, remote: &str) -> anyhow::Result<bool> {
    use semver::Version;

    let local_version = Version::parse(local)?;
    let remote_version = Version::parse(remote)?;

    Ok(remote_version > local_version)
}

pub(super) fn load_repo_config() -> Option<RepoConfig> {
    let raw = include_str!("../../../github-list-releases-parameters.json");
    serde_json::from_str(raw).ok()
}

async fn fetch_latest_release() -> Option<ReleaseResponse> {
    let config = load_repo_config()?;
    let url = format!(
        "https://api.github.com/repos/{}/{}/releases/latest",
        config.owner, config.repo
    );
    let client = Client::new();
    let response = client
        .get(url)
        .header("Accept", "application/vnd.github+json")
        .header("X-GitHub-Api-Version", "2022-11-28")
        .send()
        .await
        .ok()?;
    if !response.status().is_success() {
        return None;
    }
    let body = response.text().await.ok()?;
    serde_json::from_str(&body).ok()
}

#[cfg(target_arch = "wasm32")]
async fn open_url(url: String) {
    let url_json = serde_json::to_string(&url).unwrap_or_else(|_| "\"\"".to_string());
    let script = format!("window.open({url_json}, '_blank');");
    let _ = document::eval(&script).await;
}

#[cfg(not(target_arch = "wasm32"))]
async fn open_url(url: String) {
    let _ = webbrowser::open(&url);
}

#[component]
pub fn BakerLayout() -> Element {
    let mut app_state = use_context::<Signal<crate::components::baker::models::AppState>>();

    let mut show_new_chat = use_signal(|| false);
    let mut show_profile = use_signal(|| false);
    let mut show_tutorial = use_signal(|| false);

    let mut selected_contact_id = use_signal(|| Option::<String>::None);
    let mut menu_close_token = use_signal(|| 0usize);
    let mut replay_request_msg_id = use_signal(|| Option::<String>::None);
    let mut replay_active = use_signal(|| Option::<ReplayContext>::None);
    let mut replay_messages = use_signal(Vec::<Message>::new);
    let mut replay_token = use_signal(|| 0usize);
    let mut replay_pending = use_signal(|| Option::<PendingTyping>::None);
    let mut update_info = use_signal(|| Option::<UpdateInfo>::None);
    let mut update_checked = use_signal(|| false);

    let navigator = use_navigator();

    use_effect(move || {
        if update_checked() {
            return;
        }
        update_checked.set(true);
        let mut update_info = update_info;
        let app_state = app_state;
        spawn(async move {
            let latest = match fetch_latest_release().await {
                Some(release) => release,
                None => return,
            };
            let local_version = env!("CARGO_PKG_VERSION");
            match is_remote_newer(local_version, &latest.tag_name) {
                // 版本已经是新的了
                Ok(false) => return,
                // 无法比较
                Err(err) => {
                    error!("Failed to compare versions: {}", err);
                    return;
                }
                Ok(true) => {}
            }
            let today = Utc::now().date_naive().format("%Y-%m-%d").to_string();
            let snooze_date = app_state.read().update_snooze_date.clone();
            if snooze_date.as_deref() == Some(today.as_str()) {
                return;
            }
            update_info.set(Some(UpdateInfo {
                version: latest.tag_name,
                url: latest.html_url,
            }));
        });
    });

    let mut add_message = move |sender_id: String, content: String, kind: MessageKind| {
        let current_contact_id = match selected_contact_id() {
            Some(id) => id,
            None => return,
        };

        let is_self = sender_id == app_state.read().user_profile.id;
        let new_id = {
            let mut state = app_state.write();
            let messages = state
                .messages
                .entry(current_contact_id.clone())
                .or_default();
            let new_id = Uuid::new_v4().to_string();

            messages.push(Message {
                id: new_id.clone(),
                sender_id,
                content,
                kind,
                animate: true,
                animate_reactions: false,
                reactions: Vec::new(),
            });
            new_id
        };
        play_message_sound(is_self);
        schedule_animate_off_in_state(app_state, current_contact_id, new_id);
    };

    let handle_send = move |content: String| {
        let user_id = app_state.read().user_profile.id.clone();
        add_message(user_id, content, MessageKind::Normal);
    };

    let mut handle_send_other = move |sender_id: String, content: String| {
        add_message(sender_id, content, MessageKind::Normal);
    };

    let handle_send_status = move |content: String| {
        let user_id = app_state.read().user_profile.id.clone();
        add_message(user_id, content, MessageKind::Status);
    };

    let handle_send_image = move |data_url: String| {
        let user_id = app_state.read().user_profile.id.clone();
        add_message(user_id, data_url, MessageKind::Image);
    };
    let mut handle_send_image_other = move |sender_id: String, data_url: String| {
        add_message(sender_id, data_url, MessageKind::Image);
    };

    let handle_send_sticker = move |sticker_src: String| {
        let user_id = app_state.read().user_profile.id.clone();
        add_message(user_id, sticker_src, MessageKind::Sticker);
    };

    let mut handle_send_sticker_other = move |sender_id: String, sticker_src: String| {
        add_message(sender_id, sticker_src, MessageKind::Sticker);
    };
    let handle_add_sticker = move |sticker_src: String| {
        let trimmed = sticker_src.trim();
        if trimmed.is_empty() {
            return;
        }
        let mut state = app_state.write();
        if !state.stickers.iter().any(|s| s == trimmed) {
            state.stickers.push(trimmed.to_string());
        }
    };

    let selected_contact = {
        let selected_id = selected_contact_id();
        app_state
            .read()
            .contacts
            .iter()
            .find(|c| {
                if let Some(id) = selected_id.as_ref() {
                    id == &c.id
                } else {
                    false
                }
            })
            .cloned()
    };

    // Derived signals for Sidebar
    let contacts = use_memo(move || app_state.read().contacts.clone());
    let stickers = use_memo(move || app_state.read().stickers.clone());

    // Derived signals for ChatArea
    let messages = use_memo(move || {
        if let Some(id) = selected_contact_id() {
            if let Some(replay) = replay_active() {
                if replay.contact_id == id {
                    return replay_messages();
                }
            }
            app_state
                .read()
                .messages
                .get(&id)
                .cloned()
                .unwrap_or_default()
        } else {
            Vec::new()
        }
    });

    use_effect(move || {
        let current = selected_contact_id();
        if let Some(id) = current {
            let mut state = app_state.write();
            if let Some(msgs) = state.messages.get_mut(&id) {
                for msg in msgs.iter_mut() {
                    msg.animate = false;
                }
            }
        }
    });

    // Derived signals for Modals
    let operators = use_signal(move || app_state.read().operators.clone());
    // Sync operators back to state when modified in modal (workaround for SettingsModal signature)
    use_effect(move || {
        let current_ops = operators.read();
        if *current_ops != app_state.read().operators {
            app_state.write().operators = current_ops.clone();
        }
    });
    let background = use_signal(move || app_state.read().background.clone());
    use_effect(move || {
        let current_background = background.read();
        if *current_background != app_state.read().background {
            app_state.write().background = current_background.clone();
        }
    });

    let add_contact = move |selection: NewChatSelection| {
        let mut state = app_state.write();
        match selection {
            NewChatSelection::Single(op) => {
                let op_id = op.id.clone();
                if !state.contacts.iter().any(|c| c.id == op_id) {
                    state.contacts.push(Contact {
                        id: op_id.clone(),
                        unread_count: 0,
                        chat_head_style: ChatHeadStyle::Default,
                        name: op.name.clone(),
                        avatar_url: op.avatar_url.clone(),
                        participant_ids: vec![op_id.clone()],
                        is_group: false,
                    });
                }
                selected_contact_id.set(Some(op_id));
                show_new_chat.set(false);
            }
            NewChatSelection::Group {
                name,
                avatar_url,
                member_ids,
            } => {
                let group_id = Uuid::new_v4().to_string();
                state.contacts.push(Contact {
                    id: group_id.clone(),
                    unread_count: 0,
                    chat_head_style: ChatHeadStyle::Default,
                    name,
                    avatar_url,
                    participant_ids: member_ids,
                    is_group: true,
                });
                selected_contact_id.set(Some(group_id));
                show_new_chat.set(false);
            }
        }
    };

    // Helper to update user profile
    let update_profile = move |(name, avatar): (String, String)| {
        let mut state = app_state.write();
        state.user_profile.name = name;
        state.user_profile.avatar_url = avatar;
        show_profile.set(false);
    };

    // Helper to delete message
    let delete_message = move |msg_id: String| {
        if let Some(contact_id) = selected_contact_id() {
            let mut state = app_state.write();
            if let Some(msgs) = state.messages.get_mut(&contact_id) {
                msgs.retain(|m| m.id != msg_id);
            }
        }
    };

    // Helper to edit message
    let edit_message = move |(msg_id, new_content): (String, String)| {
        if let Some(contact_id) = selected_contact_id() {
            let mut state = app_state.write();
            if let Some(msgs) = state.messages.get_mut(&contact_id) {
                if let Some(msg) = msgs.iter_mut().find(|m| m.id == msg_id) {
                    msg.content = new_content;
                }
            }
        }
    };

    let add_reaction = move |(msg_id, reaction): (String, String)| {
        let reaction = reaction.trim().to_string();
        if reaction.is_empty() {
            return;
        }
        let sender_id = app_state.read().user_profile.id.clone();
        if let Some(contact_id) = selected_contact_id() {
            let mut should_animate = false;
            let msg_id_value = msg_id.clone();
            {
                let mut state = app_state.write();
                if let Some(msgs) = state.messages.get_mut(&contact_id) {
                    if let Some(msg) = msgs.iter_mut().find(|m| m.id == msg_id) {
                        msg.reactions.push(MessageReaction {
                            content: reaction,
                            sender_id,
                        });
                        msg.animate_reactions = true;
                        should_animate = true;
                    }
                }
            }
            if should_animate {
                schedule_reaction_animate_off_in_state(app_state, contact_id, msg_id_value);
            }
        }
    };

    let delete_reaction = move |msg_id: String| {
        if let Some(contact_id) = selected_contact_id() {
            let user_id = app_state.read().user_profile.id.clone();
            let mut state = app_state.write();
            if let Some(msgs) = state.messages.get_mut(&contact_id) {
                if let Some(msg) = msgs.iter_mut().find(|m| m.id == msg_id) {
                    // 只删除当前用户的反应，保留其他人的
                    msg.reactions
                        .retain(|reaction| reaction.sender_id != user_id);
                }
            }
        }
    };

    let insert_message =
        move |(before_id, content, sender_id_opt): (String, String, Option<String>)| {
            if let Some(contact_id) = selected_contact_id() {
                let sender_id = match sender_id_opt {
                    // 我方
                    None => app_state.read().user_profile.id.clone(),
                    // 指定发送者（单聊对方或群组选定成员）
                    Some(id) => id,
                };
                let new_id = {
                    let mut state = app_state.write();
                    let messages = state.messages.entry(contact_id.clone()).or_default();
                    let new_id = Uuid::new_v4().to_string();
                    let insert_index = messages
                        .iter()
                        .position(|m| m.id == before_id)
                        .unwrap_or(messages.len());

                    messages.insert(
                        insert_index,
                        Message {
                            id: new_id.clone(),
                            sender_id,
                            content,
                            kind: MessageKind::Normal,
                            animate: true,
                            animate_reactions: false,
                            reactions: Vec::new(),
                        },
                    );
                    new_id
                };

                schedule_animate_off_in_state(app_state, contact_id, new_id);
            }
        };

    let update_chat_head_style = move |style: ChatHeadStyle| {
        if let Some(contact_id) = selected_contact_id() {
            let mut state = app_state.write();
            if let Some(contact) = state.contacts.iter_mut().find(|c| c.id == contact_id) {
                contact.chat_head_style = style;
            }
        }
    };

    let mut cancel_replay = {
        let mut replay_active = replay_active;
        let mut replay_messages = replay_messages;
        let mut replay_token = replay_token;
        let mut replay_pending = replay_pending;
        move || {
            replay_token.set(replay_token() + 1);
            replay_active.set(None);
            replay_messages.set(Vec::new());
            replay_pending.set(None);
        }
    };

    let mut clear_messages = {
        let mut cancel_replay = cancel_replay;
        let mut app_state = app_state;
        let selected_contact_id = selected_contact_id;
        move || {
            if let Some(contact_id) = selected_contact_id() {
                let mut state = app_state.write();
                state.messages.insert(contact_id, Vec::new());
                cancel_replay();
            }
        }
    };

    let mut clear_chat = {
        let mut cancel_replay = cancel_replay;
        let mut app_state = app_state;
        let mut selected_contact_id = selected_contact_id;
        move || {
            if let Some(contact_id) = selected_contact_id() {
                let mut state = app_state.write();
                state.messages.remove(&contact_id);
                state.contacts.retain(|c| c.id != contact_id);
                selected_contact_id.set(None);
                cancel_replay();
            }
        }
    };

    let set_group_ops_list = {
        let mut app_state = app_state;
        let selected_contact_id = selected_contact_id;
        move |ops_selection: OpsSelection| {
            if let Some(contact_id) = selected_contact_id() {
                let mut state = app_state.write();
                let contact = state.contacts.iter_mut().find(|x| x.id == contact_id);

                if contact.is_none() {
                    error!("Failed to find contact {}", contact_id);
                    return;
                }

                let contact = contact.unwrap();

                contact.participant_ids = ops_selection.ops;
                contact.name = ops_selection.name;
                contact.avatar_url = ops_selection.avatar_url;
            }
        }
    };

    use_effect(move || {
        let current = selected_contact_id();
        if let Some(replay) = replay_active() {
            if Some(replay.contact_id) != current {
                replay_token.set(replay_token() + 1);
                replay_active.set(None);
                replay_messages.set(Vec::new());
                replay_pending.set(None);
            }
        }
    });

    let need_to_scroll_down = use_signal(|| false);

    let mut start_replay = {
        let mut replay_messages = replay_messages;
        let mut replay_active = replay_active;
        let mut replay_token = replay_token;
        let mut replay_pending = replay_pending;
        let app_state = app_state;
        let selected_contact_id = selected_contact_id;
        let mut need_to_scroll_down = need_to_scroll_down;
        move |start_msg_id: String, settings: ReplaySettings| {
            let contact_id = match selected_contact_id() {
                Some(id) => id,
                None => return,
            };
            let all_messages = app_state
                .read()
                .messages
                .get(&contact_id)
                .cloned()
                .unwrap_or_default();
            let start_index = all_messages
                .iter()
                .position(|m| m.id == start_msg_id)
                .unwrap_or(0);
            let prev_sender_id = if start_index > 0 {
                Some(all_messages[start_index - 1].sender_id.clone())
            } else {
                None
            };

            let token = replay_token() + 1;
            replay_token.set(token);
            replay_messages.set(Vec::new());
            replay_pending.set(None);
            replay_active.set(Some(ReplayContext {
                contact_id,
                prev_sender_id,
            }));

            let mut replay_messages_async = replay_messages;
            let replay_token_async = replay_token;
            let mut replay_pending_async = replay_pending;
            let settings_clone = settings.clone();
            let user_id = app_state.read().user_profile.id.clone();
            spawn(async move {
                for msg in all_messages.into_iter().skip(start_index) {
                    if replay_token_async() != token {
                        break;
                    }
                    if settings_clone.gap_ms > 0 {
                        sleep_ms(settings_clone.gap_ms).await;
                    }
                    if matches!(msg.kind, MessageKind::Status) {
                        replay_messages_async.with_mut(|list| {
                            list.push(Message {
                                animate: true,
                                animate_reactions: false,
                                ..msg.clone()
                            });
                        });
                        let is_self = msg.sender_id == user_id;
                        play_message_sound(is_self);
                        schedule_animate_off_in_list(replay_messages_async, msg.id);
                        need_to_scroll_down.set(true);
                        continue;
                    }
                    let typing_ms = match msg.kind {
                        MessageKind::Image | MessageKind::Sticker => settings_clone.fixed_ms,
                        _ => match settings_clone.mode {
                            ReplayIntervalMode::Fixed => settings_clone.fixed_ms,
                            ReplayIntervalMode::PerChar => {
                                let len = msg.content.chars().count() as u64;
                                len.saturating_mul(settings_clone.per_char_ms)
                            }
                        },
                    };
                    let is_other = msg.sender_id != user_id;
                    if is_other {
                        let msg_id = msg.id.clone();
                        replay_messages_async.with_mut(|list| {
                            list.push(Message {
                                animate: true,
                                animate_reactions: false,
                                reactions: Vec::new(),
                                ..msg.clone()
                            });
                        });
                        replay_pending_async.set(Some(PendingTyping {
                            id: msg_id.clone(),
                            phase: ReplayTypingPhase::Typing,
                        }));
                        schedule_animate_off_in_list(replay_messages_async, msg_id.clone());
                        if typing_ms > 0 {
                            sleep_ms(typing_ms).await;
                        }
                        replay_pending_async.set(Some(PendingTyping {
                            id: msg_id,
                            phase: ReplayTypingPhase::Reveal,
                        }));
                        need_to_scroll_down.set(true);
                        play_message_sound(false);
                        sleep_ms(200).await;
                        replay_pending_async.set(None);
                        if !msg.reactions.is_empty() {
                            if settings_clone.gap_ms > 0 {
                                sleep_ms(settings_clone.gap_ms).await;
                            }
                            let reactions = msg.reactions.clone();
                            let msg_id = msg.id.clone();
                            replay_messages_async.with_mut(|list| {
                                if let Some(item) = list.iter_mut().find(|m| m.id == msg_id) {
                                    item.reactions = reactions;
                                    item.animate_reactions = true;
                                }
                            });
                            schedule_reaction_animate_off_in_list(replay_messages_async, msg_id);
                        }
                    } else {
                        replay_pending_async.set(None);
                        if typing_ms > 0 {
                            sleep_ms(typing_ms).await;
                        }
                        replay_messages_async.with_mut(|list| {
                            list.push(Message {
                                animate: true,
                                animate_reactions: false,
                                reactions: Vec::new(),
                                ..msg.clone()
                            });
                        });
                        need_to_scroll_down.set(true);
                        play_message_sound(true);
                        schedule_animate_off_in_list(replay_messages_async, msg.id.clone());
                        if !msg.reactions.is_empty() {
                            if settings_clone.gap_ms > 0 {
                                sleep_ms(settings_clone.gap_ms).await;
                            }
                            let reactions = msg.reactions.clone();
                            let msg_id = msg.id.clone();
                            replay_messages_async.with_mut(|list| {
                                if let Some(item) = list.iter_mut().find(|m| m.id == msg_id) {
                                    item.reactions = reactions;
                                    item.animate_reactions = true;
                                }
                            });
                            schedule_reaction_animate_off_in_list(replay_messages_async, msg_id);
                        }
                    }
                }
                replay_pending_async.set(None);
            });
        }
    };

    let user_profile = app_state.read().user_profile.clone();
    let hide_tutorial = app_state.read().hide_tutorial;
    let replay_pending_for_contact = use_memo(move || {
        if let Some(replay) = replay_active() {
            if let Some(selected_id) = selected_contact_id() {
                if replay.contact_id == selected_id {
                    return replay_pending();
                }
            }
        }
        None
    });
    let background_style = use_memo(move || {
        let bg = app_state.read().background.clone();
        match bg.mode {
            BackgroundMode::DotDark => {
                "background-color: #1a1a1a; background-image: radial-gradient(#2a2a2a 1px, transparent 1px); background-size: 20px 20px;".to_string()
            }
            BackgroundMode::DotLight => {
                "background-color: #f2f2f2; background-image: radial-gradient(#d0d0d0 1px, transparent 1px); background-size: 20px 20px;".to_string()
            }
            BackgroundMode::CustomColor => format!("background-color: {};", bg.custom_color),
            BackgroundMode::CustomImage => {
                if bg.custom_image.is_empty() {
                    format!("background-color: {};", bg.custom_color)
                } else {
                    format!("background-image: url({}); background-size: cover; background-position: center; background-repeat: no-repeat; background-color: #1a1a1a;", bg.custom_image)
                }
            }
        }
    });

    rsx! {
        div {
            class: "w-full h-screen bg-cover bg-center flex flex-col overflow-hidden text-sans",
            style: "{background_style}",
            onclick: move |_| menu_close_token.set(menu_close_token() + 1),

            // Modals
            if show_new_chat() {
                NewChatModal {
                    operators,
                    on_close: move |_| show_new_chat.set(false),
                    on_select: add_contact,
                }
            }

            if show_profile() {
                ProfileModal {
                    current_name: user_profile.name.clone(),
                    current_avatar: user_profile.avatar_url.clone(),
                    on_close: move |_| show_profile.set(false),
                    on_save: update_profile,
                }
            }
            if show_tutorial() {
                TutorialModal {
                    on_close: move |_| show_tutorial.set(false),
                    on_confirm: move |dont_show| {
                        if dont_show {
                            app_state.write().hide_tutorial = true;
                        }
                        show_tutorial.set(false);
                    },
                }
            }

            if replay_request_msg_id().is_some() {
                ReplaySettingsModal {
                    on_close: move |_| replay_request_msg_id.set(None),
                    on_start: move |settings| {
                        if let Some(msg_id) = replay_request_msg_id() {
                            start_replay(msg_id, settings);
                        }
                        replay_request_msg_id.set(None);
                    },
                }
            }
            if let Some(info) = update_info() {
                UpdateAvailableModal {
                    latest_version: info.version.clone(),
                    release_url: info.url.clone(),
                    on_close: move |_| update_info.set(None),
                    on_skip_today: move |_| {
                        let today = Utc::now().date_naive().format("%Y-%m-%d").to_string();
                        app_state.write().update_snooze_date = Some(today);
                        update_info.set(None);
                    },
                    on_update_now: move |url| {
                        update_info.set(None);
                        spawn(async move {
                            open_url(url).await;
                        });
                    },
                }
            }

            // 顶部导航栏
            div { class: "h-16 flex items-center px-8 justify-between shrink-0 z-10",
                div { class: "flex items-center gap-4",
                    span {
                        class: "text-white text-base font-bold cursor-pointer select-none hover:text-gray-300 transition-colors",
                        ondoubleclick: move |_| {
                            navigator.push(Route::SettingsPage {});
                        },
                        "//BAKER/会话消息"
                    }
                    if !hide_tutorial {
                        a {
                            class: "text-blue-300 text-sm underline hover:text-blue-200 transition-colors",
                            href: "#",
                            onclick: move |evt| {
                                evt.prevent_default();
                                show_tutorial.set(true);
                            },
                            "点击这里看教程！！"
                        }
                    }
                }

                // Profile Button
                div {
                    class: "flex items-center gap-3 cursor-pointer hover:bg-white/5 p-2 rounded-lg transition-colors",
                    onclick: move |_| show_profile.set(true),
                    span { class: "text-gray-300 text-sm", "{user_profile.name}" }
                    div { class: "w-8 h-8 rounded bg-gray-600 overflow-hidden border border-gray-500",
                        if !user_profile.avatar_url.is_empty() {
                            img {
                                src: "{user_profile.avatar_url}",
                                class: "w-full h-full object-cover",
                            }
                        } else {
                            div { class: "w-full h-full flex items-center justify-center text-white text-xs font-bold",
                                "{user_profile.name.chars().next().unwrap_or('?')}"
                            }
                        }
                    }
                }
            }

            // 主内容区
            div { class: "flex-1 flex overflow-hidden p-8 gap-8 min-h-0",

                // 左侧栏
                Sidebar {
                    contacts,
                    operators,
                    selected_contact_id,
                    on_add_click: move |_| {
                        cancel_replay();
                        show_new_chat.set(true);
                    },
                    need_to_scroll_down,
                }

                // 右侧聊天区
                if let Some(contact) = selected_contact {
                    {
                        let replay_prev_sender_id = replay_active()
                            .and_then(|replay| {
                                if replay.contact_id == contact.id {
                                    replay.prev_sender_id
                                } else {
                                    None
                                }
                            });
                        let is_replaying = replay_active()
                            .map(|replay| replay.contact_id == contact.id)
                            .unwrap_or(false);
                        let force_first_avatar = is_replaying;
                        rsx! {
                            ChatArea {
                                contact,
                                operators,
                                messages,
                                user_profile,
                                menu_close_token,
                                first_prev_sender_id: replay_prev_sender_id,
                                force_first_avatar,
                                pending_typing: replay_pending_for_contact,
                                need_to_scroll_down,
                                on_send_message: handle_send,
                                on_send_other_message: move |(sender_id, text)| {
                                    handle_send_other(sender_id, text);
                                },
                                on_send_status: handle_send_status,
                                on_send_image: handle_send_image,
                                on_send_image_other: move |(sender_id, data_url)| {
                                    handle_send_image_other(sender_id, data_url);
                                },
                                on_send_sticker: handle_send_sticker,
                                on_send_sticker_other: move |(sender_id, sticker)| {
                                    handle_send_sticker_other(sender_id, sticker);
                                },
                                stickers,
                                on_add_sticker: handle_add_sticker,
                                on_delete_message: delete_message,
                                on_edit_message: edit_message,
                                on_add_reaction: add_reaction,
                                on_delete_reaction: delete_reaction,
                                on_insert_message: insert_message,
                                on_start_replay: move |msg_id| replay_request_msg_id.set(Some(msg_id)),
                                is_replaying,
                                on_exit_replay: move |_| cancel_replay(),
                                on_update_chat_head_style: update_chat_head_style,
                                on_clear_messages: move |_| clear_messages(),
                                on_clear_chat: move |_| clear_chat(),
                                on_set_group_ops_list: set_group_ops_list,
                            }
                        }
                    }
                } else {
                    div { class: "flex-1 flex items-center justify-center text-gray-500 bg-gray-900/50 border border-gray-600 backdrop-blur-sm",
                        "请选择一个会话"
                    }
                }
            }
        }
    }
}

#[derive(Debug, Clone, Routable, PartialEq)]
#[rustfmt::skip]
pub enum Route {
    #[route("/")]
    BakerLayout {},
    #[route("/settings")]
    SettingsPage {},
}
