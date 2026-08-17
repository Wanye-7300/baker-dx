use std::iter;

use crate::operator::model::*;
use crate::operator::view_model::*;
use crate::panic_try;
use crate::session::model::*;
use crate::session::repository::*;
use crate::session::view_model::input_view_model::*;
use crate::session::view_model::session_view_model::*;
use crate::shared::assets;
use crate::shared::assets::ICON_ROUND_CHR_0003_ENDMINF;
use crate::shared::assets::icons;
use crate::shared::assets::stickers;
use crate::shared::database;
use crate::ui::components::*;
use crate::ui::selector;
use crate::view_try;

use dioxus::{prelude::*, web::WebFileExt};
use strum::VariantArray;
use uuid::Uuid;

type ActionParameter = (
    u64,
    MessageType,
    Vec<(crate::shared::assets::Emoji, Vec<Option<Uuid>>)>,
    bool,
);
type ProcessedMessage = (u64, MessageType, Vec<(assets::Emoji, Vec<Option<Uuid>>)>, bool);

#[component]
pub(crate) fn SessionUI() -> Element {
    let session_view_model = use_context::<SessionViewModel>();
    let session_ui_view_model = use_context::<SessionUIViewModel>();
    let input_view_model = use_context::<InputViewModel>();

    let sessions = session_view_model.sessions;
    let current_session = session_view_model.message_repository.read().current_session();
    let message_repository = session_view_model.message_repository;

    let with_more_menu_open = session_ui_view_model.with_more_menu_open;
    let with_stickers_menu_open = session_ui_view_model.with_stickers_menu_open;
    let mut with_message_actions_menu_open = session_ui_view_model.with_message_actions_menu_open;
    let mut with_reaction_menu_open = session_ui_view_model.with_reaction_menu_open;
    let mut need_to_scroll_down = session_ui_view_model.need_to_scroll_down;

    let mut input_area_message_type = input_view_model.input_area_message_type;
    let mut input_area_text = input_view_model.input_area_text;
    let mut input_area_mode = input_view_model.input_area_mode;

    let submit = move |sender: Sender| {
        if input_area_message_type() == InputAreaMessageType::Text && input_area_text.is_empty() {
            return;
        }

        let message = Message::new(
            sender,
            match input_area_message_type() {
                InputAreaMessageType::Text => MessageType::Text(input_area_text()),
                InputAreaMessageType::Image(uuid) => MessageType::Image(uuid),
                InputAreaMessageType::HorizontalBreak => MessageType::HorizontalBreak,
                InputAreaMessageType::State => MessageType::State(input_area_text()),
                InputAreaMessageType::StateWithHorizontalLine => {
                    MessageType::StateWithHorizontalLine(input_area_text())
                }
                InputAreaMessageType::Sticker(sticker) => MessageType::Sticker(sticker),
            },
        );

        spawn(async move {
            match input_area_mode() {
                InputAreaMode::Normal => {
                    panic_try!(MessageRepository::push(message_repository, message).await);
                }
                InputAreaMode::Insert { id } => {
                    panic_try!(MessageRepository::insert(message_repository, message, id).await);
                    input_area_mode.set(InputAreaMode::Normal);
                }
                InputAreaMode::Modify { id } => {
                    panic_try!(MessageRepository::modify(message_repository, id, message).await);
                    input_area_mode.set(InputAreaMode::Normal);
                }
            }
        });

        input_area_text.set(String::new());
        input_area_message_type.set(InputAreaMessageType::Text);
        *need_to_scroll_down.write() = true;
    };

    if current_session.is_some() {
        // TODO: 虽然 current_session.read().unwrap() 正常情况下是保证正确的 —— 但是谁知道呢？SessionMainContent 与
        // InputArea 同
        let uuid = current_session.unwrap();
        let current_session_name = sessions.read();
        let current_session_name = view_try!(current_session_name.get(&uuid)).session_name();

        rsx! {
            div { id: "session", class: "flex flex-column",
                div { class: "flex flex-column", id: "session-header",
                    span { {current_session_name.to_string()} }
                }
                div { id: "session-main", class: "flex flex-column",
                    SessionMainContent {}
                    InputArea { on_submit: submit }
                    if with_more_menu_open() {
                        MoreMenu { on_submit: submit }
                    }
                    if with_stickers_menu_open() {
                        StickersMenu { on_submit: submit }
                    }
                }
                img {
                    id: "session-decorate",
                    src: crate::DECO_SNS_TWEET_DECORATE_10,
                }

                if let Some(Action(_session_uuid, message_id, x, y)) = with_message_actions_menu_open() {
                    Menu {
                        groups: vec![
                            MenuGroup {
                                title: Some(String::from("对消息进行操作")),
                                items: vec![
                                    MenuItem {
                                        icon: Some(icons::DELETE_48DP_000000_FILL0_WGHT400_GRAD0_OPSZ48),
                                        label: String::from("删除"),
                                        on_click: EventHandler::new(move |_| async move {
                                            panic_try!(
                                                MessageRepository::delete(message_repository, message_id).
                                                await
                                            );
                                            input_area_mode.set(InputAreaMode::Normal);
                                        }),
                                    },
                                    MenuItem {
                                        icon: Some(
                                            icons::ADD_REACTION_48DP_000000_FILL0_WGHT400_GRAD0_OPSZ48,
                                        ),
                                        label: String::from("添加Reaction…"),
                                        on_click: EventHandler::new(move |_| {
                                            with_reaction_menu_open.set(with_message_actions_menu_open());
                                        }),
                                    },
                                    MenuItem {
                                        icon: Some(
                                            icons::ARROW_INSERT_48DP_000000_FILL0_WGHT400_GRAD0_OPSZ48,
                                        ),
                                        label: String::from("在此前插入消息…"),
                                        on_click: EventHandler::new(move |_| {
                                            input_area_mode
                                                .set(InputAreaMode::Insert {
                                                    id: message_id,
                                                });
                                        }),
                                    },
                                    MenuItem {
                                        icon: Some(icons::EDIT_48DP_000000_FILL0_WGHT400_GRAD0_OPSZ48),
                                        label: String::from("修改消息…"),
                                        on_click: EventHandler::new(move |_| {
                                            input_area_mode
                                                .set(InputAreaMode::Modify {
                                                    id: message_id,
                                                });
                                        }),
                                    },
                                ],
                            },
                        ],
                        on_close: move |_| {
                            with_message_actions_menu_open.set(None);
                        },
                        x,
                        y,
                    }
                }

                if let Some(Action(session_uuid, message_id, x, y)) = with_reaction_menu_open() {
                    ReactionMenu {
                        on_confirm: move |(participants_ids_selected, emoji): (Vec<Option<Uuid>>, _)| async move {
                            panic_try!(
                                MessageRepository::append_reaction(message_repository, message_id, (emoji,
                                participants_ids_selected),). await
                            );
                        },
                        session_uuid,
                        on_close: move |_| {
                            with_reaction_menu_open.set(None);
                        },
                        x,
                        y,
                    }
                }
            }
        }
    } else {
        rsx! {
            div { id: "session-not-chosen", class: "flex", "请选择一个会话" }
        }
    }
}

#[component]
fn SessionMainContent() -> Element {
    let session_view_model = use_context::<SessionViewModel>();
    let session_ui_view_model = use_context::<SessionUIViewModel>();
    let operator_view_model = use_context::<OperatorViewModel>();

    let message_repository = session_view_model.message_repository;

    let mut need_to_scroll_down = session_ui_view_model.need_to_scroll_down;

    // use_resource(move || async move {
    //     let current_session_uuid = current_session.unwrap();

    //     need_to_scroll_down.set(true);
    // });

    use_effect(move || {
        if !*need_to_scroll_down.read() {
            return;
        }

        spawn(async {
            // TODO: 使用 MountedData
            let _ = document::eval(
                "\n\
            let element = document.querySelector('#session-main-content');\n\
            element.scroll(0, element.scrollHeight);",
            )
            .await;
        });

        *need_to_scroll_down.write() = false;
    });

    let mut messages = vec![];

    {
        let m = message_repository.read();
        let m = view_try!(m.iterator());

        let mut iter = m.peekable();

        let mut temporary = vec![]; // 用于判断一组消息是不是一个人发的，然后塞进 messages
        let mut sender_now = iter.peek().map(|x| x.1.sender());

        loop {
            let peek = iter.peek();
            if peek.is_some_and(|x| Some(x.1.sender()) == sender_now && x.1.content().is_text_or_image()) {
                temporary.push((
                    peek.unwrap().0,
                    peek.unwrap().1.content().clone(),
                    peek.unwrap().1.reactions().clone(),
                    peek.unwrap().1.animation(),
                ));
            } else {
                if !temporary.is_empty() {
                    messages.push((
                        sender_now.is_some_and(|x| x.avatar_should_on_left()),
                        match sender_now {
                            Some(Sender::Others(uuid)) => {
                                view_try!(operator_view_model.operator_repository.read().get(*uuid))
                                    .get_avatar_originally()
                                    .to_asset_operator()
                            }
                            _ => ICON_ROUND_CHR_0003_ENDMINF,
                        },
                        temporary,
                    ));
                }
                if peek.is_none() {
                    break;
                }
                temporary = vec![(
                    peek.unwrap().0,
                    peek.unwrap().1.content().clone(),
                    peek.unwrap().1.reactions().clone(),
                    peek.unwrap().1.animation(),
                )];
                sender_now = peek.map(|x| x.1.sender());
            }
            iter.next();
        }
    }

    rsx! {
        div { id: "session-main-content",
            for (avatar_on_left , avatar , messages) in messages {
                MessageRow { avatar_on_left, avatar, messages }
            }
        }
    }
}

#[component]
fn InputArea(on_submit: EventHandler<Sender>) -> Element {
    let session_view_model = use_context::<SessionViewModel>();
    let session_ui_view_model = use_context::<SessionUIViewModel>();
    let input_view_model = use_context::<InputViewModel>();
    let operator_view_model = use_context::<OperatorViewModel>();

    let sessions = session_view_model.sessions;
    let current_session = session_view_model.message_repository.read().current_session().unwrap();

    let mut with_sender_selector_open = session_ui_view_model.with_sender_selector_open;
    let mut with_more_menu_open = session_ui_view_model.with_more_menu_open;
    let mut with_stickers_menu_open = session_ui_view_model.with_stickers_menu_open;

    let mut input_area_message_type = input_view_model.input_area_message_type;
    let mut input_area_text = input_view_model.input_area_text;
    let mut input_area_mode = input_view_model.input_area_mode;

    let on_submit_click = move |evt: Event<MouseData>| match evt.modifiers().ctrl() {
        true => with_sender_selector_open.set(true),
        false => on_submit.call(Sender::Endministrator),
    };

    let input_area_style = if with_more_menu_open() || with_stickers_menu_open() {
        "input-area-with-menu flex flex-row"
    } else {
        "flex flex-row"
    };

    use_effect(move || {
        session_view_model.message_repository.read();
        with_more_menu_open.set(false);
    });

    rsx! {
        div { id: "input-area", class: input_area_style.to_string(),
            div { id: "input-area-input",
                // input {
                //     id: "input-area-input-input",
                //     oninput: move |evt| { input_area_text.set(evt.value()) },
                //     onkeypress: move |evt: Event<KeyboardData>| {
                //         if evt.code() == Code::Enter {
                //             match evt.modifiers().ctrl() {
                //                 true => with_sender_selector_open.set(true),
                //                 false => on_submit.call(Sender::Endministrator),
                //             }
                //         }
                //     },
                //     r#type: "text",
                //     value: input_area_text,
                // }
                textarea {
                    id: "input-area-input-input",
                    oninput: move |evt| { input_area_text.set(evt.value()) },
                    onkeydown: move |evt: Event<KeyboardData>| {
                        if evt.code() == Code::Enter {
                            if evt.modifiers().shift() {
                                return;
                            }
                            evt.stop_propagation();
                            evt.prevent_default();
                            match evt.modifiers().ctrl() {
                                true => with_sender_selector_open.set(true),
                                false => on_submit.call(Sender::Endministrator),
                            }
                        }
                    },
                    value: input_area_text,
                }
            }
            button { id: "input-area-submit", onclick: on_submit_click }
            button {
                id: "input-area-stickers",
                class: if with_stickers_menu_open() { "input-area-stickers-selected input-area-button" } else { "input-area-button" },
                onclick: move |_| {
                    with_stickers_menu_open.set(!with_stickers_menu_open());
                    with_more_menu_open.set(false);
                },
            }
            button {
                id: "input-area-more",
                class: if with_more_menu_open() { "input-area-more-selected input-area-button" } else { "input-area-button" },
                onclick: move |_| {
                    with_more_menu_open.set(!with_more_menu_open());
                    with_stickers_menu_open.set(false);
                },
            }

            if with_sender_selector_open() {
                selector::Selector {
                    kv: view_try!(sessions.read().get(& current_session))
                        .participants_ids()
                        .iter()
                        .filter_map(|x| {
                            operator_view_model
                                .operator_repository
                                .read()
                                .get(*x)
                                .ok()
                                .filter(|operator| operator.activity())
                                .map(|operator| (Some(*x), operator.name().clone()))
                        })
                        .chain(iter::once((None, "管理员".to_owned())))
                        .collect(),
                    title: "选择发送者",
                    message_type_selector: true,
                    func: move |(message_type, sender): (Option<InputAreaMessageType>, Sender)| {
                        if let Some(message_type) = message_type {
                            input_area_message_type.set(message_type);
                        }
                        on_submit.call(sender);
                        with_sender_selector_open.set(false);
                    },
                    on_close: move |_| {
                        with_sender_selector_open.set(false);
                    },
                }
            }

            if let InputAreaMode::Insert { .. } = *input_area_mode.read() {
                div { id: "insert-mode-wrapper",
                    span { "插入模式" }
                    button {
                        onclick: move |_| {
                            input_area_mode.set(InputAreaMode::Normal);
                        },
                        "×"
                    }
                }
            } else if let InputAreaMode::Modify { .. } = *input_area_mode.read() {
                div { id: "insert-mode-wrapper", class: "modify-mode",
                    span { "修改模式" }
                    button {
                        onclick: move |_| {
                            input_area_mode.set(InputAreaMode::Normal);
                        },
                        "×"
                    }
                }
            }
        }
    }
}

#[component]
fn MoreMenu(on_submit: EventHandler<Sender>) -> Element {
    let session_view_model = use_context::<SessionViewModel>();
    let session_ui_view_model = use_context::<SessionUIViewModel>();
    let input_view_model = use_context::<InputViewModel>();
    let operator_view_model = use_context::<OperatorViewModel>();

    let mut sessions = session_view_model.sessions;
    let current_session = session_view_model.message_repository.read().current_session().unwrap();
    let mut message_repository = session_view_model.message_repository;
    let session_name = sessions.read().get(&current_session).unwrap().session_name().clone();

    let mut with_sender_selector_open = session_ui_view_model.with_sender_selector_open;
    let mut with_more_menu_open = session_ui_view_model.with_more_menu_open;

    let mut input_area_message_type = input_view_model.input_area_message_type;

    let participants_ids: Signal<fnv::FnvHashSet<Uuid>> = use_signal(|| {
        sessions
            .read()
            .get(&current_session)
            .unwrap()
            .participants_ids()
            .iter()
            .copied()
            .collect()
    });

    use_effect(move || {
        let mut participant_ids = participants_ids.read().iter().copied().collect::<Vec<_>>();
        participant_ids.sort_unstable();

        let operators = operator_view_model.operator_repository.read();
        if let Ok(session) = sessions.write().get_mut(&current_session) {
            session.set_participants_ids(participant_ids);
            session.refresh_avatar(operators.operators());
        }
    });

    let onchange = move |evt: Event<FormData>| {
        if let Some(file_data) = evt.files().first() {
            let file = file_data.get_web_file().unwrap();

            spawn(async move {
                let uuid = Uuid::new_v4();
                database::save_multimedia(uuid, file.into()).await.unwrap();
                input_area_message_type.set(InputAreaMessageType::Image(uuid));
                // on_submit.call(None);
                with_sender_selector_open.set(true);
                with_more_menu_open.set(false);
            });
        }
    };

    rsx! {
        InputAreaMenu {
            label { class: "more-menu-upload-button",
                "发送图片"
                input {
                    r#type: "file",
                    accept: "image/*",
                    hidden: true,
                    onchange,
                }
            }
            hr {}
            h3 { "会话设置" }
            label { "会话名" }
            input {
                r#type: "text",
                value: session_name.to_string(),
                onchange: move |evt| {
                    sessions.write().get_mut(&current_session).unwrap().rename(evt.value());
                },
                {session_name.to_string()}
            }
            div { class: "more-menu-actions",
                span {
                    onclick: move |_| async move {
                        message_repository.write().clear();
                        panic_try!(SessionRepository::delete_session(sessions, current_session). await);
                    },
                    "删除此会话（消息会永久消失！）"
                }
            }
            h3 { "干员管理" }
            crate::ui::ParticipantsSelection { participants_ids }
        }
    }
}

#[component]
fn StickersMenu(on_submit: EventHandler<Sender>) -> Element {
    let session_ui_view_model = use_context::<SessionUIViewModel>();
    let input_view_model = use_context::<InputViewModel>();

    let mut with_sender_selector_open = session_ui_view_model.with_sender_selector_open;
    let mut with_stickers_menu_open = session_ui_view_model.with_stickers_menu_open;

    let mut input_area_message_type = input_view_model.input_area_message_type;

    rsx! {
        InputAreaMenu {
            div { class: "stickers-menu",
                for variant in stickers::Stickers::VARIANTS {
                    img {
                        onclick: move |_| {
                            input_area_message_type.set(InputAreaMessageType::Sticker(*variant));
                            with_sender_selector_open.set(true);
                            with_stickers_menu_open.set(false);
                        },
                        class: "stickers-menu-sticker",
                        key: "{*variant:?}",
                        title: "{*variant:?}",
                        src: Asset::from(*variant),
                    }
                }
            }
        }
    }
}

#[component]
fn MessageRow(avatar_on_left: bool, avatar: Asset, messages: Vec<ProcessedMessage>) -> Element {
    let session_view_model = use_context::<SessionViewModel>();
    let session_ui_view_model = use_context::<SessionUIViewModel>();

    let current_session = session_view_model.message_repository.read().current_session().unwrap();
    let mut with_reaction_menu_open = session_ui_view_model.with_reaction_menu_open;

    if messages.is_empty() {
        return rsx! {};
    }

    let message_id = messages[0].0;

    let oncontextmenu = move |evt: Event<MouseData>| {
        evt.prevent_default();
        let position = evt.client_coordinates();
        with_reaction_menu_open.set(Some(Action(current_session, message_id, position.x, position.y)));
    };

    match &messages[0].1 {
        MessageType::HorizontalBreak => rsx! {
            div { class: "horizontal-break",
                span { oncontextmenu }
                img { class: "hb-deco1", src: crate::DECO_SNS_TWEET_DECORATE_11 }
                img { class: "hb-deco2", src: crate::LINE_SNS_TWEET_DECORATE }
            }
        },
        MessageType::State(txt) => rsx! {
            div { class: "state", oncontextmenu,
                span { {txt.to_string()} }
            }
        },
        MessageType::StateWithHorizontalLine(txt) => rsx! {
            div { class: "state-with-hl", oncontextmenu,
                span {}
                span { {txt.to_string()} }
                span {}
            }
        },
        MessageType::Text(_) | MessageType::Image(_) | MessageType::Sticker(_) => {
            let avatar_left_class = if avatar_on_left {
                "message-row-avatar message-row-avatar-background"
            } else {
                "message-row-avatar"
            };

            let avatar_right_class = if !avatar_on_left {
                "message-row-avatar message-row-avatar-background"
            } else {
                "message-row-avatar"
            };

            rsx! {
                div { class: "flex flex-row",
                    // 左侧头像
                    div { class: avatar_left_class,
                        if avatar_on_left {
                            img { src: avatar }
                        }
                    }
                    // 中间消息
                    div { class: if avatar_on_left { "message-row-content message-row-content-left" } else { "message-row-content message-row-content-right" },
                        for message in messages {
                            MessageBubble {
                                key: "{message.0}",
                                avatar_on_left,
                                message,
                            }
                        }
                    }
                    // 右侧头像
                    div { class: avatar_right_class,
                        if !avatar_on_left {
                            img { src: avatar }
                        }
                    }
                }
            }
        }
    }
}

#[component]
fn MessageBubble(avatar_on_left: bool, message: ActionParameter) -> Element {
    let session_view_model = use_context::<SessionViewModel>();
    let session_ui_view_model = use_context::<SessionUIViewModel>();
    let operator_view_model = use_context::<OperatorViewModel>();

    let current_session = session_view_model.message_repository.read().current_session().unwrap();
    let mut with_message_actions_menu_open = session_ui_view_model.with_message_actions_menu_open;

    let message_id = message.0;

    let oncontextmenu = move |evt: Event<MouseData>| {
        evt.prevent_default();
        let position = evt.client_coordinates();
        with_message_actions_menu_open.set(Some(Action(current_session, message_id, position.x, position.y)));
    };

    let bubble_class = if avatar_on_left {
        if message.3 {
            "message-bubble-others message-bubble-animate-left"
        } else {
            "message-bubble-others"
        }
    } else {
        if message.3 {
            "message-bubble-self message-bubble-animate-right"
        } else {
            "message-bubble-self"
        }
    };

    rsx! {
        div { class: "message-bubble-wrapper", oncontextmenu,
            match message.1 {
                MessageType::Text(txt) => rsx! {
                    RichText { class: bubble_class, text: txt,
                        if !message.2.is_empty() {
                            div { class: "message-reaction-wrapper",
                                for reaction in message.2 {
                                    span { class: if message.3 { "message-reaction message-reaction-animation" } else { "message-reaction" },
                                        img { src: Asset::from(reaction.0) }
                                        span {
                                            {
                                                reaction
                                                    .1
                                                    .iter()
                                                    .map(|x| match x {
                                                        Some(x) => {
                                                            operator_view_model
                                                                .operator_repository
                                                                .read()
                                                                .get(*x)
                                                                .unwrap()
                                                                .name()
                                                                .clone()
                                                        }
                                                        None => String::from("管理员"),
                                                    })
                                                    .collect::<Vec<String>>()
                                                    .join("、")
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
                MessageType::Image(uuid) => rsx! {
                    span { class: "{bubble_class} message-bubble-image",
                        crate::ui::Image { uuid }
                    }
                },
                MessageType::Sticker(sticker) => rsx! {
                    div { class: "{bubble_class} message-bubble-sticker",
                        img { src: Asset::from(sticker) }
                    }
                },
                _ => unreachable!(),
            }
        }
    }
}
