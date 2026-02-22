use crate::components::baker::chat_area::{ChatArea, PendingTyping, ReplayTypingPhase};
use crate::components::baker::modals::{
    NewChatModal, NewChatSelection, ProfileModal, ReplayIntervalMode, ReplaySettings,
    ReplaySettingsModal, SettingsModal, UpdateAvailableModal,
};
use crate::components::baker::models::{
    BackgroundMode, ChatHeadStyle, Contact, Message, MessageKind,
};
use crate::components::baker::sidebar::Sidebar;
use crate::components::baker::storage::{load_state, save_state};
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

#[derive(Clone, PartialEq)]
struct UpdateInfo {
    version: String,
    url: String,
}

#[derive(Deserialize)]
struct RepoConfig {
    owner: String,
    repo: String,
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

fn parse_version(input: &str) -> Option<Vec<u64>> {
    let trimmed = input.trim();
    let without_prefix = trimmed.strip_prefix('v').unwrap_or(trimmed);
    let base = without_prefix.split('-').next().unwrap_or(without_prefix);
    let mut parts = Vec::new();
    for part in base.split('.') {
        let value = part.parse::<u64>().ok()?;
        parts.push(value);
    }
    if parts.is_empty() {
        None
    } else {
        Some(parts)
    }
}

fn is_remote_newer(local: &str, remote: &str) -> bool {
    let local_parts = match parse_version(local) {
        Some(parts) => parts,
        None => return false,
    };
    let remote_parts = match parse_version(remote) {
        Some(parts) => parts,
        None => return false,
    };
    let max_len = local_parts.len().max(remote_parts.len());
    for i in 0..max_len {
        let a = *local_parts.get(i).unwrap_or(&0);
        let b = *remote_parts.get(i).unwrap_or(&0);
        if b > a {
            return true;
        }
        if b < a {
            return false;
        }
    }
    false
}

fn load_repo_config() -> Option<RepoConfig> {
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
        .header("User-Agent", "baker-dx")
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
async fn open_url(_url: String) {}

#[component]
pub fn BakerLayout() -> Element {
    // Load initial state
    let mut app_state = use_signal(load_state);

    // Persist state on change
    use_effect(move || {
        save_state(&app_state.read());
    });

    let mut show_settings = use_signal(|| false);
    let mut show_new_chat = use_signal(|| false);
    let mut show_profile = use_signal(|| false);

    let mut selected_contact_id = use_signal(|| Option::<String>::None);
    let mut menu_close_token = use_signal(|| 0usize);
    let mut replay_request_msg_id = use_signal(|| Option::<String>::None);
    let mut replay_active = use_signal(|| Option::<ReplayContext>::None);
    let mut replay_messages = use_signal(Vec::<Message>::new);
    let mut replay_token = use_signal(|| 0usize);
    let mut replay_pending = use_signal(|| Option::<PendingTyping>::None);
    let mut update_info = use_signal(|| Option::<UpdateInfo>::None);
    let mut update_checked = use_signal(|| false);

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
            if !is_remote_newer(local_version, &latest.tag_name) {
                return;
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

        let new_id = {
            let mut state = app_state.write();
            let messages = state
                .messages
                .entry(current_contact_id.clone())
                .or_insert(Vec::new());
            let new_id = Uuid::new_v4().to_string();

            messages.push(Message {
                id: new_id.clone(),
                sender_id,
                content,
                kind,
                animate: true,
            });
            new_id
        };

        let mut app_state_for_anim = app_state;
        spawn(async move {
            sleep_ms(220).await;
            if let Some(msgs) = app_state_for_anim
                .write()
                .messages
                .get_mut(&current_contact_id)
            {
                if let Some(msg) = msgs.iter_mut().find(|m| m.id == new_id) {
                    msg.animate = false;
                }
            }
        });
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

    let insert_message = move |(before_id, content, is_self): (String, String, bool)| {
        if let Some(contact_id) = selected_contact_id() {
            let sender_id = if is_self {
                app_state.read().user_profile.id.clone()
            } else {
                app_state
                    .read()
                    .contacts
                    .iter()
                    .find(|c| c.id == contact_id)
                    .and_then(|c| c.participant_ids.first().cloned())
                    .unwrap_or(contact_id.clone())
            };
            let new_id = {
                let mut state = app_state.write();
                let messages = state
                    .messages
                    .entry(contact_id.clone())
                    .or_insert(Vec::new());
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
                    },
                );
                new_id
            };

            let mut app_state_for_anim = app_state;
            spawn(async move {
                sleep_ms(220).await;
                if let Some(msgs) = app_state_for_anim.write().messages.get_mut(&contact_id) {
                    if let Some(msg) = msgs.iter_mut().find(|m| m.id == new_id) {
                        msg.animate = false;
                    }
                }
            });
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

    let mut start_replay = {
        let mut replay_messages = replay_messages;
        let mut replay_active = replay_active;
        let mut replay_token = replay_token;
        let mut replay_pending = replay_pending;
        let app_state = app_state;
        let selected_contact_id = selected_contact_id;
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
                                ..msg.clone()
                            });
                        });
                        let mut replay_messages_anim = replay_messages_async;
                        let msg_id = msg.id;
                        spawn(async move {
                            sleep_ms(220).await;
                            replay_messages_anim.with_mut(|list| {
                                if let Some(item) = list.iter_mut().find(|m| m.id == msg_id) {
                                    item.animate = false;
                                }
                            });
                        });
                        continue;
                    }
                    let typing_ms = match settings_clone.mode {
                        ReplayIntervalMode::Fixed => settings_clone.fixed_ms,
                        ReplayIntervalMode::PerChar => {
                            let len = msg.content.chars().count() as u64;
                            len.saturating_mul(settings_clone.per_char_ms)
                        }
                    };
                    let is_other = msg.sender_id != user_id;
                    if is_other {
                        let msg_id = msg.id.clone();
                        replay_messages_async.with_mut(|list| {
                            list.push(Message {
                                animate: true,
                                ..msg.clone()
                            });
                        });
                        replay_pending_async.set(Some(PendingTyping {
                            id: msg_id.clone(),
                            phase: ReplayTypingPhase::Typing,
                        }));
                        let mut replay_messages_anim = replay_messages_async;
                        let msg_id_anim = msg_id.clone();
                        spawn(async move {
                            sleep_ms(220).await;
                            replay_messages_anim.with_mut(|list| {
                                if let Some(item) = list.iter_mut().find(|m| m.id == msg_id_anim) {
                                    item.animate = false;
                                }
                            });
                        });
                        if typing_ms > 0 {
                            sleep_ms(typing_ms).await;
                        }
                        replay_pending_async.set(Some(PendingTyping {
                            id: msg_id,
                            phase: ReplayTypingPhase::Reveal,
                        }));
                        sleep_ms(200).await;
                        replay_pending_async.set(None);
                    } else {
                        replay_pending_async.set(None);
                        if typing_ms > 0 {
                            sleep_ms(typing_ms).await;
                        }
                        replay_messages_async.with_mut(|list| {
                            list.push(Message {
                                animate: true,
                                ..msg.clone()
                            });
                        });
                        let mut replay_messages_anim = replay_messages_async;
                        let msg_id = msg.id;
                        spawn(async move {
                            sleep_ms(220).await;
                            replay_messages_anim.with_mut(|list| {
                                if let Some(item) = list.iter_mut().find(|m| m.id == msg_id) {
                                    item.animate = false;
                                }
                            });
                        });
                    }
                }
                replay_pending_async.set(None);
            });
        }
    };

    let user_profile = app_state.read().user_profile.clone();
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
    let background_style = {
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
    };

    rsx! {
        div {
            class: "w-full h-screen bg-cover bg-center flex flex-col overflow-hidden text-sans",
            style: "{background_style}",
            onclick: move |_| menu_close_token.set(menu_close_token() + 1),

            // Modals
            if show_settings() {
                SettingsModal {
                    operators,
                    background,
                    on_close: move |_| show_settings.set(false),
                }
            }

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
                        ondoubleclick: move |_| show_settings.set(true),
                        "//BAKER/好友沟通"
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
                        let force_first_avatar = replay_active()
                            .map(|replay| replay.contact_id == contact.id)
                            .unwrap_or(false);
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
                                on_send_message: handle_send,
                                    on_send_other_message: move |(sender_id, text)| {
                                        handle_send_other(sender_id, text);
                                    },
                                on_send_status: handle_send_status,
                                on_delete_message: delete_message,
                                on_edit_message: edit_message,
                                on_insert_message: insert_message,
                                on_start_replay: move |msg_id| replay_request_msg_id.set(Some(msg_id)),
                                on_update_chat_head_style: update_chat_head_style,
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
