use crate::components::baker::input_bar::InputBar;
use crate::components::baker::modals::{
    EditMessageModal, InsertMessageModal, OpsSelection, PickSenderModal, ReactionModal,
    SetGroupOpsListModal,
};
use crate::components::baker::models::{
    ChatHeadStyle, Contact, Message, MessageKind, Operator, UserProfile,
};
use dioxus::prelude::*;
use std::collections::HashMap;
use std::rc::Rc;

const CHAT_HEAD_LEFT: Asset = asset!("/assets/images/chat_head_left.png");
const CHAT_HEAD_MID: Asset = asset!("/assets/images/chat_head_mid.png");
const CHAT_HEAD_RIGHT: Asset = asset!("/assets/images/chat_head_right.png");
const CHAT_HEAD_LEFT_2: Asset = asset!("/assets/images/chat_head_left_2.png");
const CHAT_HEAD_MID_2: Asset = asset!("/assets/images/chat_head_mid_2.png");
const CHAT_HEAD_RIGHT_2: Asset = asset!("/assets/images/chat_head_right_2.png");

fn menu_style(x: i32, y: i32, width: i32, height: i32) -> String {
    format!(
        "left: clamp(8px, {x}px, calc(100vw - {width}px - 8px)); top: clamp(8px, {y}px, calc(100vh - {height}px - 8px));"
    )
}

#[derive(Clone, PartialEq)]
pub enum ReplayTypingPhase {
    Typing,
    Reveal,
}

#[derive(Clone, PartialEq)]
pub struct PendingTyping {
    pub id: String,
    pub phase: ReplayTypingPhase,
}

#[component]
pub fn ChatArea(
    contact: Contact,
    operators: ReadSignal<Vec<Operator>>,
    messages: ReadSignal<Vec<Message>>,
    user_profile: UserProfile,
    menu_close_token: ReadSignal<usize>,
    first_prev_sender_id: Option<String>,
    force_first_avatar: bool,
    pending_typing: ReadSignal<Option<PendingTyping>>,
    need_to_scroll_down: Signal<bool>,
    on_send_message: EventHandler<String>,
    on_send_other_message: EventHandler<(String, String)>,
    on_send_status: EventHandler<String>,
    on_send_image: EventHandler<String>,
    on_send_sticker: EventHandler<String>,
    on_send_sticker_other: EventHandler<(String, String)>,
    stickers: ReadSignal<Vec<String>>,
    on_add_sticker: EventHandler<String>,
    on_delete_message: EventHandler<String>,
    on_edit_message: EventHandler<(String, String)>,
    on_add_reaction: EventHandler<(String, String)>,
    on_delete_reaction: EventHandler<String>,
    on_insert_message: EventHandler<(String, String, Option<String>)>,
    on_start_replay: EventHandler<String>,
    is_replaying: bool,
    on_exit_replay: EventHandler<()>,
    on_update_chat_head_style: EventHandler<ChatHeadStyle>,
    on_clear_messages: EventHandler<()>,
    on_clear_chat: EventHandler<()>,
    on_set_group_ops_list: EventHandler<OpsSelection>,
) -> Element {
    let messages_list = messages.read().clone();
    let mut context_menu = use_signal(|| Option::<(i32, i32, String)>::None);
    let mut editing_msg_id = use_signal(|| Option::<String>::None);
    let mut insert_before_id = use_signal(|| Option::<String>::None);
    let mut reaction_msg_id = use_signal(|| Option::<String>::None);
    let mut header_menu_open = use_signal(|| false);
    let mut show_pick_sender = use_signal(|| false);
    let mut pick_sender_text = use_signal(|| "".to_string());
    let mut pick_sender_sticker = use_signal(|| Option::<String>::None);
    let mut clear_input_token = use_signal(|| 0usize);
    let mut sticker_menu_state = use_signal(|| Option::<(i32, i32)>::None);
    let mut show_set_group_ops_list = use_signal(|| false);

    use_effect(move || {
        menu_close_token.read();
        context_menu.set(None);
        header_menu_open.set(false);
        sticker_menu_state.set(None);
    });

    // 监听消息列表变化，自动滚动到底部
    use_effect(move || {
        messages.read();
        pending_typing.read();
        if !need_to_scroll_down() {
            return;
        }
        spawn(async move {
            let eval = document::eval(
                r#"
                requestAnimationFrame(() => {
                    const container = document.getElementById('chat-scroll-container');
                    if (!container) return;
                    const lastBubble = container.querySelector('.bubble-wrap:last-of-type');
                    if (lastBubble) {
                        lastBubble.getBoundingClientRect();
                    }
                    void container.offsetHeight;
                    container.scrollTo({
                        top: container.scrollHeight,
                        behavior: 'instant'
                    });
                });
            "#,
            );
            let _ = eval.await;
        });
    });

    let mut handle_delete = move |id: String| {
        on_delete_message.call(id);
        context_menu.set(None);
        need_to_scroll_down.set(false);
    };
    let handle_edit_save = move |new_content: String| {
        if let Some(id) = editing_msg_id() {
            on_edit_message.call((id, new_content));
        }
        editing_msg_id.set(None);
        need_to_scroll_down.set(false);
    };
    let handle_insert_save = move |(content, sender_id): (String, Option<String>)| {
        if let Some(before_id) = insert_before_id() {
            on_insert_message.call((before_id, content, sender_id));
        }
        insert_before_id.set(None);
        need_to_scroll_down.set(false);
    };
    let handle_reaction_save = move |reaction: String| {
        if let Some(id) = reaction_msg_id() {
            on_add_reaction.call((id, reaction));
        }
        reaction_msg_id.set(None);
        need_to_scroll_down.set(false);
    };
    let mut handle_delete_reaction = move |id: String| {
        on_delete_reaction.call(id);
        context_menu.set(None);
        need_to_scroll_down.set(false);
    };
    let user_id = user_profile.id.clone();
    let user_profile = Rc::new(user_profile);
    let first_prev_sender_id = first_prev_sender_id.clone();
    let pending_typing_state = pending_typing.read().clone();
    let operators_list = operators.read().clone();
    let header_name = if !contact.name.is_empty() {
        contact.name.clone()
    } else {
        operators_list
            .iter()
            .find(|op| op.id == contact.id)
            .map(|op| op.name.clone())
            .unwrap_or_else(|| "未命名会话".to_string())
    };
    let operators_map: HashMap<&str, &Operator> = operators_list
        .iter()
        .map(|op| (op.id.as_str(), op))
        .collect();
    let resolve_sender = |sender_id: &str| -> (String, String) {
        if sender_id == user_id {
            return (user_profile.name.clone(), user_profile.avatar_url.clone());
        }
        if let Some(op) = operators_map.get(sender_id) {
            return (op.name.clone(), op.avatar_url.clone());
        }
        ("".to_string(), "".to_string())
    };
    let selectable_members = contact
        .participant_ids
        .iter()
        .filter_map(|id| operators_list.iter().find(|op| op.id == *id).cloned())
        .collect::<Vec<_>>();
    let context_menu_value = context_menu();
    let context_menu_view = context_menu_value.as_ref().map(|(x, y, msg_id)| {
        let show_reaction = messages_list
            .iter()
            .find(|m| m.id == *msg_id)
            .map(|m| !matches!(m.kind, MessageKind::Status))
            .unwrap_or(false);
        let show_delete_reaction = messages_list
            .iter()
            .find(|m| m.id == *msg_id)
            .map(|m| !m.reactions.is_empty())
            .unwrap_or(false);
        let x = *x;
        let y = *y;
        let msg_id = msg_id.clone();
        rsx! {
            div {
                class: "fixed z-[100] bg-[#2b2b2b] border border-gray-600 rounded shadow-xl py-1 w-32",
                style: "{menu_style(x, y, 128, 256)}",
                onclick: |e| e.stop_propagation(),
                div {
                    class: "px-4 py-2 hover:bg-[#3a3a3a] cursor-pointer text-white text-sm transition-colors",
                    onclick: {
                        let msg_id = msg_id.clone();
                        move |_| {
                            editing_msg_id.set(Some(msg_id.clone()));
                            context_menu.set(None);
                        }
                    },
                    "修改消息"
                }
                if show_reaction {
                    div {
                        class: "px-4 py-2 hover:bg-[#3a3a3a] cursor-pointer text-white text-sm transition-colors",
                        onclick: {
                            let msg_id = msg_id.clone();
                            move |_| {
                                reaction_msg_id.set(Some(msg_id.clone()));
                                context_menu.set(None);
                            }
                        },
                        "反应…"
                    }
                }
                if show_delete_reaction {
                    div {
                        class: "px-4 py-2 hover:bg-[#3a3a3a] cursor-pointer text-white text-sm transition-colors",
                        onclick: {
                            let msg_id = msg_id.clone();
                            move |_| handle_delete_reaction(msg_id.clone())
                        },
                        "删除反应"
                    }
                }
                div {
                    class: "px-4 py-2 hover:bg-[#3a3a3a] cursor-pointer text-white text-sm transition-colors",
                    onclick: {
                        let msg_id = msg_id.clone();
                        move |_| {
                            insert_before_id.set(Some(msg_id.clone()));
                            context_menu.set(None);
                        }
                    },
                    "在此前插入…"
                }
                div {
                    class: "px-4 py-2 hover:bg-[#3a3a3a] cursor-pointer text-white text-sm transition-colors",
                    onclick: {
                        let msg_id = msg_id.clone();
                        move |_| {
                            on_start_replay.call(msg_id.clone());
                            context_menu.set(None);
                        }
                    },
                    "从此开始回放…"
                }
                div {
                    class: "px-4 py-2 hover:bg-[#3a3a3a] cursor-pointer text-red-400 text-sm transition-colors",
                    onclick: {
                        let msg_id = msg_id.clone();
                        move |_| handle_delete(msg_id.clone())
                    },
                    "删除消息"
                }
            }
        }
    });
    enum ChatRow {
        Status {
            id: String,
            content: String,
        },
        Message {
            msg: Message,
            is_self: bool,
            show_avatar: bool,
            item_margin: String,
            msg_id: String,
            pending_phase: Option<ReplayTypingPhase>,
            sender_name: String,
            sender_avatar: String,
            reaction_labels: Vec<String>,
        },
    }
    let mut message_rows = Vec::new();
    let mut last_sender_id: Option<String> = None;
    for msg in messages_list.iter() {
        if matches!(msg.kind, MessageKind::Status) {
            message_rows.push(ChatRow::Status {
                id: msg.id.clone(),
                content: msg.content.clone(),
            });
            continue;
        }
        let is_self = msg.sender_id == user_id;
        let show_avatar = if last_sender_id.is_none() {
            if force_first_avatar {
                true
            } else {
                match first_prev_sender_id.as_ref() {
                    Some(prev_sender_id) => prev_sender_id != &msg.sender_id,
                    None => true,
                }
            }
        } else {
            last_sender_id
                .as_ref()
                .map(|sender_id| sender_id != &msg.sender_id)
                .unwrap_or(true)
        };
        let item_margin = if show_avatar { "mt-4" } else { "mt-1" }.to_string();
        let pending_phase = pending_typing_state.as_ref().and_then(|pending| {
            if pending.id == msg.id {
                Some(pending.phase.clone())
            } else {
                None
            }
        });
        let (sender_name, sender_avatar) = resolve_sender(&msg.sender_id);
        // 将相同表情内容的反应用计数形式合并显示，例如 "😀 x3"
        let mut reaction_counts: Vec<(String, u32)> = Vec::new();
        for reaction in &msg.reactions {
            if let Some((_, count)) = reaction_counts
                .iter_mut()
                .find(|(content, _)| content == &reaction.content)
            {
                *count += 1;
            } else {
                reaction_counts.push((reaction.content.clone(), 1));
            }
        }
        let reaction_labels = reaction_counts
            .into_iter()
            .map(|(content, count)| {
                if count > 1 {
                    format!("{content} x{count}")
                } else {
                    content
                }
            })
            .collect::<Vec<_>>();
        message_rows.push(ChatRow::Message {
            msg: msg.clone(),
            is_self,
            show_avatar,
            item_margin,
            msg_id: msg.id.clone(),
            pending_phase,
            sender_name,
            sender_avatar,
            reaction_labels,
        });
        last_sender_id = Some(msg.sender_id.clone());
    }

    rsx! {
        div {
            class: "flex-1 flex flex-col h-full relative min-h-0",
            onclick: move |_| {
                context_menu.set(None);
                header_menu_open.set(false);
            },

            if let Some(editing_id) = editing_msg_id() {
                if let Some(msg) = messages.read().iter().find(|m| m.id == editing_id) {
                    EditMessageModal {
                        initial_content: msg.content.clone(),
                        on_close: move |_| editing_msg_id.set(None),
                        on_save: handle_edit_save,
                    }
                }
            }
            if insert_before_id().is_some() {
                InsertMessageModal {
                    members: selectable_members.clone(),
                    on_close: move |_| insert_before_id.set(None),
                    on_save: handle_insert_save,
                }
            }
            if reaction_msg_id().is_some() {
                ReactionModal {
                    on_close: move |_| reaction_msg_id.set(None),
                    on_save: handle_reaction_save,
                }
            }
            if show_pick_sender() {
                PickSenderModal {
                    members: selectable_members.clone(),
                    on_close: move |_| show_pick_sender.set(false),
                    on_send: move |sender_id| {
                        if let Some(sticker) = pick_sender_sticker() {
                            on_send_sticker_other.call((sender_id, sticker));
                            pick_sender_sticker.set(None);
                            show_pick_sender.set(false);
                            return;
                        }
                        let text = pick_sender_text();
                        if !text.trim().is_empty() {
                            on_send_other_message.call((sender_id, text));
                            pick_sender_text.set("".to_string());
                            clear_input_token.set(clear_input_token() + 1);
                        }
                        show_pick_sender.set(false);
                    },
                }
            }
            if show_set_group_ops_list() {
                SetGroupOpsListModal {
                    on_close: move |_| show_set_group_ops_list.set(false),
                    on_select: move |ops_selection| {
                        on_set_group_ops_list.call(ops_selection);
                        show_set_group_ops_list.set(false);
                    },
                    selected_contact_id: Some(contact.id.clone()),
                }
            }

            {context_menu_view}

            div { class: "h-14 flex items-stretch shrink-0 mb-1 relative",
                {
                    let (left, mid, right) = match contact.chat_head_style {
                        ChatHeadStyle::Default => (CHAT_HEAD_LEFT, CHAT_HEAD_MID, CHAT_HEAD_RIGHT),
                        ChatHeadStyle::Alt => (CHAT_HEAD_LEFT_2, CHAT_HEAD_MID_2, CHAT_HEAD_RIGHT_2),
                    };
                    rsx! {
                        img {
                            src: "{left}",
                            class: "h-full w-auto object-cover select-none pointer-events-none",
                        }
                        div {
                            class: "flex-1 h-full",
                            style: "background-image: url({mid}); background-size: 100% 100%;",
                        }
                        img {
                            src: "{right}",
                            class: "h-full w-auto object-cover select-none pointer-events-none",
                        }
                    }
                }

                div { class: "absolute inset-0 flex items-center justify-between px-6",
                    div { class: "flex items-center gap-2 ml-2 z-30",
                        span { class: "text-white font-bold text-lg", "{header_name}" }
                    }

                    div { class: "flex items-center justify-end h-full relative",
                        div {
                            class: "w-8 h-8 rounded-full flex items-center justify-center cursor-pointer transition-opacity hover:opacity-80 mr-0",
                            style: "background-color: rgb(68, 67, 67);",
                            onclick: move |e| {
                                e.stop_propagation();
                                header_menu_open.set(!header_menu_open());
                            },

                            div { class: "flex gap-0.5",
                                div {
                                    class: "w-1 h-1 rounded-full",
                                    style: "background-color: rgb(255, 253, 253);",
                                }
                                div {
                                    class: "w-1 h-1 rounded-full",
                                    style: "background-color: rgb(255, 253, 253);",
                                }
                                div {
                                    class: "w-1 h-1 rounded-full",
                                    style: "background-color: rgb(255, 253, 253);",
                                }
                            }
                        }
                        if header_menu_open() {
                            div {
                                class: "absolute right-0 top-10 z-50 w-32 bg-[#2b2b2b] border border-gray-600 rounded shadow-xl py-1",
                                onclick: |e| e.stop_propagation(),
                                div {
                                    class: "px-4 py-2 hover:bg-[#3a3a3a] cursor-pointer text-white text-sm transition-colors",
                                    onclick: move |_| {
                                        on_update_chat_head_style.call(ChatHeadStyle::Default);
                                        header_menu_open.set(false);
                                    },
                                    "会话头样式 1"
                                }
                                div {
                                    class: "px-4 py-2 hover:bg-[#3a3a3a] cursor-pointer text-white text-sm transition-colors",
                                    onclick: move |_| {
                                        on_update_chat_head_style.call(ChatHeadStyle::Alt);
                                        header_menu_open.set(false);
                                    },
                                    "会话头样式 2"
                                }
                                div { class: "h-px bg-gray-600 my-1" }
                                div {
                                    class: "px-4 py-2 hover:bg-[#3a3a3a] cursor-pointer text-white text-sm transition-colors",
                                    onclick: move |_| {
                                        on_clear_messages.call(());
                                        header_menu_open.set(false);
                                    },
                                    "清除聊天内容"
                                }
                                div {
                                    class: "px-4 py-2 hover:bg-[#3a3a3a] cursor-pointer text-white text-sm transition-colors",
                                    onclick: move |_| {
                                        on_clear_chat.call(());
                                        header_menu_open.set(false);
                                    },
                                    "清除聊天内容和会话"
                                }
                                if is_replaying {
                                    div {
                                        class: "px-4 py-2 hover:bg-[#3a3a3a] cursor-pointer text-white text-sm transition-colors",
                                        onclick: move |_| {
                                            on_exit_replay.call(());
                                            header_menu_open.set(false);
                                        },
                                        "退出回放"
                                    }
                                }
                                {
                                    if contact.is_group {
                                        rsx! {
                                            div {
                                                class: "px-4 py-2 hover:bg-[#3a3a3a] cursor-pointer text-white text-sm transition-colors",
                                                onclick: move |_| {
                                                    show_set_group_ops_list.set(true);
                                                    header_menu_open.set(false);
                                                },
                                                "更改群组干员名单……"
                                            }
                                        }
                                    } else {
                                        rsx! {}
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // 聊天主体 (包含消息列表和输入框)
            div { class: "flex-1 flex flex-col relative bg-transparent rounded-b-xl min-h-0",
                // 边框绘制
                // 统一的边框容器 (左、右、下边框及圆角)
                div {
                    class: "absolute inset-0 rounded-b-xl border-l-[1.5px] border-r-[1.5px] border-b-[1.5px] border-[rgb(202,201,201)] pointer-events-none z-20",
                    style: "top: 0;", // 确保顶部与容器对齐
                }
                // 1. 顶部左侧长横线
                div { class: "absolute top-0 left-0 right-[264px] h-[1.5px] bg-[rgb(202,201,201)] z-30 pointer-events-none" }
                // 2. 顶部右侧短横线
                div { class: "absolute top-0 right-0 w-8 h-[1.5px] bg-[rgb(202,201,201)] z-30 pointer-events-none" }
                // 3. 凹陷部分 (使用 SVG 绘制，保证平滑的斜角)
                div { class: "absolute top-0 right-8 w-[232px] h-[10px] z-30 pointer-events-none",
                    svg {
                        width: "100%",
                        height: "100%",
                        view_box: "0 0 232 10",
                        preserve_aspect_ratio: "none",
                        path {
                            d: "M0,0 L16,6 L216,6 L232,0",
                            fill: "none",
                            stroke: "rgb(202, 201, 201)",
                            stroke_width: "1.5",
                            vector_effect: "non-scaling-stroke",
                        }
                    }
                }

                // 彩色装饰条 (位于凹陷内部)
                div { class: "absolute top-[0px] right-[44px] flex gap-2 h-[2px] z-30 pointer-events-none",
                    div {
                        class: "w-16 h-[2px] shadow-[0_0_8px_rgb(226,2,226)]",
                        style: "background-color: rgb(226, 2, 226); clip-path: polygon(0 0, 100% 0, 100% 100%, 8px 100%);",
                    }
                    div {
                        class: "w-16 h-[2px] shadow-[0_0_8px_rgb(243,241,0)]",
                        style: "background-color: rgb(243, 241, 0);",
                    }
                    div {
                        class: "w-16 h-[2px] shadow-[0_0_8px_rgb(1,241,241)]",
                        style: "background-color: rgb(1, 241, 241); clip-path: polygon(0 0, 100% 0, calc(100% - 8px) 100%, 0 100%);",
                    }
                }

                // 消息列表区域
                div {
                    id: "chat-scroll-container",
                    class: "flex-1 overflow-y-auto p-6 mr-3 custom-scrollbar flex flex-col relative z-10",

                    for row in message_rows {
                        match row {
                            ChatRow::Status { id, content } => rsx! {
                                div {
                                    class: "text-center text-gray-500 text-xs my-2 font-mono cursor-context-menu",
                                    key: "{id}",
                                    oncontextmenu: move |evt| {
                                        evt.prevent_default();
                                        context_menu
                                            .set(
                                                Some((
                                                    evt.client_coordinates().x as i32,
                                                    evt.client_coordinates().y as i32,
                                                    id.clone(),
                                                )),
                                            );
                                    },
                                    "{content}"
                                }
                            },
                            ChatRow::Message {
                                msg,
                                is_self,
                                show_avatar,
                                item_margin,
                                msg_id,
                                pending_phase,
                                sender_name,
                                sender_avatar,
                                reaction_labels,
                            } => {
                                let row_key = msg_id.clone();
                                rsx! {
                                    div { class: "{item_margin}", key: "{row_key}",
                                        MessageBubble {
                                            message: msg,
                                            is_self,
                                            show_avatar,
                                            show_sender_name: contact.is_group && show_avatar,
                                            sender_name,
                                            sender_avatar,
                                            pending_phase,
                                            reaction_labels,
                                            user_profile: user_profile.clone(),
                                            on_context_menu: move |evt: MouseEvent| {
                                                context_menu
                                                    .set(
                                                        Some((
                                                            evt.client_coordinates().x as i32,
                                                            evt.client_coordinates().y as i32,
                                                            msg_id.clone(),
                                                        )),
                                                    );
                                            },
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                div { class: "h-[2px] mx-3 mt-0 mb-1.5 bg-[rgb(71,71,71)] z-10" }
                div {
                    class: "mx-[1.5px] mb-[1.5px] rounded-b-[10px] flex flex-col",
                    style: if sticker_menu_state().is_some() { "background-color: black; border-radius: 8px;" } else { "background-color: rgb(50, 50, 50); border-radius: 0 0 10px 10px;" },
                    div { class: "p-4",
                        InputBar {
                            on_send: move |text| on_send_message.call(text),
                            on_send_other: move |text| {
                                if contact.is_group {
                                    pick_sender_text.set(text);
                                    show_pick_sender.set(true);
                                } else {
                                    on_send_other_message.call((contact.id.clone(), text));
                                    clear_input_token.set(clear_input_token() + 1);
                                }
                            },
                            is_group: contact.is_group,
                            on_send_status: move |text| on_send_status.call(text),
                            on_send_image: move |data_url| on_send_image.call(data_url),
                            on_send_sticker: move |(sticker_src, is_ctrl)| {
                                if is_ctrl {
                                    pick_sender_sticker.set(Some(sticker_src));
                                    show_pick_sender.set(true);
                                } else {
                                    on_send_sticker.call(sticker_src);
                                }
                            },
                            stickers,
                            on_add_sticker,
                            menu_close_token,
                            sticker_menu: sticker_menu_state,
                            clear_text_token: clear_input_token,
                            need_to_scroll_down,
                        }
                    }
                }
            }
        }
    }
}

#[component]
fn MessageBubble(
    message: Message,
    is_self: bool,
    show_avatar: bool,
    show_sender_name: bool,
    sender_name: String,
    sender_avatar: String,
    pending_phase: Option<ReplayTypingPhase>,
    reaction_labels: Vec<String>,
    user_profile: Rc<UserProfile>,
    on_context_menu: EventHandler<MouseEvent>,
) -> Element {
    let align_class = if is_self { "items-end" } else { "items-start" };
    let anim_origin = if is_self {
        "transform-origin: calc(100% + 8px) 0;"
    } else {
        "transform-origin: -8px 0;"
    };

    let (bg_color, text_color, show_grid) = if is_self {
        ("rgb(243, 242, 242)", "text-black", true)
    } else {
        ("rgb(69, 69, 69)", "text-white", false)
    };

    let bubble_style = if show_grid {
        format!(
            "
            background-color: {bg_color};
            background-image: 
                linear-gradient(rgb(239, 237, 237) 1px, transparent 1px),
                linear-gradient(90deg, rgb(239, 237, 237) 1px, transparent 1px);
            background-size: 4px 4px;
            overflow-wrap: anywhere;
            word-break: break-word;
        "
        )
    } else {
        format!(
            "background-color: {bg_color}; background-image: none; overflow-wrap: anywhere; word-break: break-word;"
        )
    };
    let image_bubble_style =
        "background: transparent; background-image: none; box-shadow: none; padding: 0;";
    let is_sticker = matches!(message.kind, MessageKind::Sticker);

    let is_image = matches!(message.kind, MessageKind::Image);
    let base_bubble_class =
        "relative px-3 py-2 text-base font-medium shadow-sm break-words whitespace-pre-wrap leading-relaxed text-left";
    let bubble_class = if is_image && !is_sticker {
        "relative text-base font-medium leading-relaxed text-left"
    } else {
        base_bubble_class
    };
    let typing_bubble_class =
        "relative px-3 py-2 text-base font-medium shadow-sm leading-relaxed text-left min-h-[42px] flex items-center";

    let bubble_radius_class = if is_self {
        "rounded-2xl rounded-tr-none"
    } else {
        "rounded-2xl rounded-tl-none"
    };

    let bubble_anim_style = if message.animate
        || matches!(
            pending_phase,
            Some(ReplayTypingPhase::Reveal) | Some(ReplayTypingPhase::Typing)
        ) {
        format!("animation: bubbleExpand 0.15s cubic-bezier(1,0,1,0.2) forwards; {anim_origin}")
    } else {
        anim_origin.to_string()
    };
    let text_anim_style = if message.animate
        || matches!(
            pending_phase,
            Some(ReplayTypingPhase::Reveal) | Some(ReplayTypingPhase::Typing)
        ) {
        if is_self {
            "opacity: 0; animation: textFadeIn 0.2s ease-out 0.15s forwards;"
        } else {
            "opacity: 0; animation: textFadeIn 0.1s ease-out 0.12s forwards;"
        }
    } else {
        "opacity: 1;"
    };

    let avatar_slot_class = if is_self {
        "absolute right-0 top-0"
    } else {
        "absolute left-0 top-0"
    };
    let row_align_class = if is_self {
        "flex justify-end"
    } else {
        "flex justify-start"
    };
    let bubble_wrap_class = if is_self {
        "relative group bubble-wrap mt-1 cursor-context-menu inline-block max-w-[60%] min-w-0 mr-[68px]"
            .to_string()
    } else {
        "relative group bubble-wrap mt-1 cursor-context-menu inline-block max-w-[60%] min-w-0 ml-[68px]"
            .to_string()
    };
    let reaction_wrap_class = "mt-2 flex gap-1 items-center flex-wrap";
    let reaction_align_class = if is_self {
        "justify-end"
    } else {
        "justify-start"
    };
    let name_wrap_class = if is_self {
        "w-full flex justify-end pr-[68px]"
    } else {
        "w-full flex justify-start pl-[68px]"
    };

    rsx! {
        div { class: "flex flex-col {align_class} gap-0 w-full max-w-full",
            if show_sender_name {
                div { class: "{name_wrap_class}",
                    div {
                        class: "text-sm mb-1",
                        style: "color: rgb(167, 167, 167);",
                        if is_self {
                            "{user_profile.name}"
                        } else {
                            "{sender_name}"
                        }
                    }
                }
            }
            div { class: "relative w-full min-w-0 {row_align_class}",
                if show_avatar {
                    div { class: "w-14 h-14 bg-gray-600 border border-gray-500 shrink-0 flex items-center justify-center text-xs text-gray-300 rounded-sm overflow-hidden {avatar_slot_class}",
                        if is_self {
                            if !user_profile.avatar_url.is_empty() {
                                img {
                                    src: "{user_profile.avatar_url}",
                                    class: "w-full h-full object-cover",
                                }
                            } else {
                                "Me"
                            }
                        } else if !sender_avatar.is_empty() {
                            img {
                                src: "{sender_avatar}",
                                class: "w-full h-full object-cover",
                            }
                        } else {
                            span { class: "text-2xl font-bold",
                                "{sender_name.chars().next().unwrap_or('?')}"
                            }
                        }
                    }
                }

                div {
                    class: "{bubble_wrap_class}",
                    style: "{bubble_anim_style}",
                    oncontextmenu: move |evt| {
                        evt.prevent_default();
                        on_context_menu.call(evt);
                    },
                    if !is_image {
                        if !is_self {
                            div { class: "absolute top-0 -left-[8px] w-[9px] h-[20px] overflow-hidden",
                                svg {
                                    view_box: "0 0 9 20",
                                    width: "100%",
                                    height: "100%",
                                    preserve_aspect_ratio: "none",
                                    path {
                                        d: "M9,0 L0,0 Q9,0 9,20 Z",
                                        fill: "{bg_color}",
                                    }
                                }
                            }
                        } else {
                            div { class: "absolute top-0 -right-[8px] w-[9px] h-[20px] overflow-hidden",
                                svg {
                                    view_box: "0 0 9 20",
                                    width: "100%",
                                    height: "100%",
                                    preserve_aspect_ratio: "none",
                                    path {
                                        d: "M0,0 L9,0 Q0,0 0,20 Z",
                                        fill: "{bg_color}",
                                    }
                                }
                            }
                        }
                    }

                    if matches!(pending_phase, Some(ReplayTypingPhase::Typing)) {
                        div {
                            class: "{typing_bubble_class} {bubble_radius_class} {text_color}",
                            style: "{bubble_style}",
                            div {
                                class: "typing-dots",
                                style: "{text_anim_style}",
                                div { class: "typing-dot typing-dot-1" }
                                div { class: "typing-dot typing-dot-2" }
                                div { class: "typing-dot typing-dot-3" }
                            }
                        }
                    } else {
                        div {
                            class: "{bubble_class} {bubble_radius_class} {text_color}",
                            style: if is_image { "{image_bubble_style}" } else { "{bubble_style}" },
                            if is_image && !is_sticker {
                                img {
                                    src: "{message.content}",
                                    class: "max-w-[320px] object-contain rounded",
                                }
                            } else if is_sticker {
                                img {
                                    src: "{message.content}",
                                    class: "max-w-[200px] object-contain",
                                    style: "{text_anim_style}",
                                }
                            } else {
                                div { style: "{text_anim_style}", "{message.content}" }
                            }
                            if !reaction_labels.is_empty() {
                                div { class: "{reaction_wrap_class} {reaction_align_class}",
                                    for label in reaction_labels.iter().cloned() {
                                        span {
                                            class: "px-2 py-0.5 text-base rounded-full text-gray-200",
                                            style: if message.animate_reactions { "background-color: rgb(60, 60, 60); opacity: 0; animation: textFadeIn 0.2s ease-out 0.05s forwards;" } else { "background-color: rgb(60, 60, 60); opacity: 1;" },
                                            "{label}"
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
