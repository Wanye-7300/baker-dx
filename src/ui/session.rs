use std::iter;

use dioxus::{prelude::*, web::WebFileExt};
use uuid::Uuid;

use crate::Sender;

pub(crate) mod selector;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub(crate) enum InputAreaMessageType {
    #[default]
    Text,
    Image(Uuid),
    HorizontalBreak,
    State,
    StateWithHorizontalLine,
}

impl TryFrom<&str> for InputAreaMessageType {
    type Error = anyhow::Error;

    fn try_from(value: &str) -> std::prelude::v1::Result<Self, Self::Error> {
        match value {
            "a" => Ok(Self::Text),
            "c" => Ok(Self::HorizontalBreak),
            "d" => Ok(Self::State),
            "e" => Ok(Self::StateWithHorizontalLine),
            _ => Err(anyhow::anyhow!("Unknown Parameter")),
        }
    }
}

#[component]
pub(crate) fn Image(
    uuid: Uuid,
    #[props(extends = GlobalAttributes, extends = img)] attributes: Vec<Attribute>,
) -> Element {
    let mut img_src = use_signal(String::new);
    let img_uuid = use_signal(|| uuid);

    use_effect(move || {
        img_uuid.read();

        spawn(async move {
            let blob = crate::database::get_multimedia(img_uuid()).await.unwrap().unwrap();
            let url = web_sys::Url::create_object_url_with_blob(&blob).unwrap();

            img_src.set(url);
        });
    });

    rsx! {
        img { src: img_src, ..attributes }
    }
}

#[component]
pub(super) fn SessionUI() -> Element {
    let mut baker_state = use_context::<crate::BakerState>();
    let sessions = baker_state.sessions;
    let current_session = baker_state.current_session;

    let with_sender_selector_message_type = use_signal(|| true);
    let with_sender_selector_open = use_signal(|| false);
    let with_more_menu_open = use_signal(|| false);

    let mut input_area_message_type = use_signal(InputAreaMessageType::default);
    let mut input_area_text = use_signal(String::new);

    let submit = move |sender: Sender| {
        if input_area_message_type() == InputAreaMessageType::Text && input_area_text.is_empty() {
            return;
        }

        let mut sessions = baker_state.sessions;
        let current_session = baker_state.current_session.unwrap();

        let insert_id = sessions.read().get(&current_session).unwrap().id;

        let message = crate::Message {
            sender,
            content: match input_area_message_type() {
                InputAreaMessageType::Text => crate::MessageType::Text(input_area_text()),
                InputAreaMessageType::Image(uuid) => crate::MessageType::Image(uuid),
                InputAreaMessageType::HorizontalBreak => crate::MessageType::HorizontalBreak,
                InputAreaMessageType::State => crate::MessageType::State(input_area_text()),
                InputAreaMessageType::StateWithHorizontalLine => {
                    crate::MessageType::StateWithHorizontalLine(input_area_text())
                }
            },
            animation: true,
        };

        let message_wrapper = crate::database::MessageWrapper {
            session_uuid: current_session,
            message_id: insert_id,
            message: message.clone(),
        };

        let mode = *baker_state.input_area_mode.read();

        spawn(async move {
            match mode {
                crate::InputAreaMode::Normal => {
                    sessions.write().get_mut(&current_session).unwrap().id += 1;
                    crate::database::put_messages(vec![message_wrapper]).await.unwrap();
                }
                crate::InputAreaMode::Insert { id } => {
                    let mut message_wrapper = message_wrapper;
                    message_wrapper.message_id = id;
                    let need_to_update = crate::database::insert_message(message_wrapper).await.unwrap();

                    let mut sessions = baker_state.sessions;
                    if need_to_update {
                        sessions.write().get_mut(&current_session).unwrap().id += 1;
                    }
                }
                crate::InputAreaMode::Modify { id } => {
                    let mut message_wrapper = message_wrapper;
                    message_wrapper.message_id = id;
                    crate::database::modify_message(message_wrapper).await.unwrap();
                }
            }
        });

        let mut messages = baker_state.messages.write();
        let messages = messages.as_mut().unwrap();

        for (_, v) in messages.iter_mut().rev() {
            // TODO: 优化
            v.animation = false;
        }

        match mode {
            crate::InputAreaMode::Normal => {
                messages.insert(insert_id, message);
            }
            crate::InputAreaMode::Insert { id } => {
                if messages.contains_key(&id) {
                    let others = messages.split_off(&id);
                    messages.insert(id, message);
                    for (k, v) in others {
                        messages.insert(k + 1, v);
                    }
                } else {
                    messages.insert(id, message);
                }
                baker_state.input_area_mode.set(crate::InputAreaMode::Normal);
            }
            crate::InputAreaMode::Modify { id } => {
                messages.remove(&id);
                messages.insert(id, message);
                baker_state.input_area_mode.set(crate::InputAreaMode::Normal);
            }
        }

        input_area_text.set(String::new());
        input_area_message_type.set(InputAreaMessageType::Text);
        *baker_state.need_to_scroll_down.write() = true;
    };

    if current_session.read().is_some() {
        // TODO: 虽然 current_session.read().unwrap() 正常情况下是保证正确的 —— 但是谁知道呢？SessionMainContent 与
        // InputArea 同
        let uuid = current_session.read().unwrap();
        let current_session_name = sessions.read()[&uuid].session_name.clone();

        rsx! {
            div { id: "session", class: "flex flex-column",
                div { class: "flex flex-column", id: "session-header",
                    span { {current_session_name} }
                }
                div { id: "session-main", class: "flex flex-column",
                    SessionMainContent {}
                    InputArea {
                        input_area_message_type,
                        input_area_text,
                        with_more_menu_open,
                        with_sender_selector_message_type,
                        with_sender_selector_open,
                        on_submit: submit,
                    }
                    if with_more_menu_open() {
                        MoreMenu {
                            current_session,
                            with_more_menu_open,
                            with_sender_selector_open,
                            input_area_message_type,
                            on_submit: submit,
                        }
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
    let mut baker_state = use_context::<crate::BakerState>();

    use_resource(move || async move {
        let current_session_uuid = baker_state.current_session.read().unwrap();
        let messages = crate::database::get_messages(current_session_uuid).await.unwrap();
        baker_state.messages.set(Some(messages));

        baker_state.need_to_scroll_down.set(true);
    });

    use_effect(move || {
        if !*baker_state.need_to_scroll_down.read() {
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

        *baker_state.need_to_scroll_down.write() = false;
    });

    let mut messages = vec![];

    if let Some(m) = baker_state.messages.read().as_ref() {
        let mut iter = m.iter().peekable();

        let mut temporary = vec![]; // 用于判断一组消息是不是一个人发的，然后塞进 messages
        let mut sender_now = iter.peek().map(|x| x.1.sender);

        loop {
            let peek = iter.peek();
            if peek.is_some_and(|x| Some(x.1.sender) == sender_now && x.1.content.is_text_or_image()) {
                temporary.push((
                    *peek.unwrap().0,
                    peek.unwrap().1.content.clone(),
                    peek.unwrap().1.animation,
                ));
            } else {
                if !temporary.is_empty() {
                    messages.push((
                        sender_now.is_some_and(|x| x.avatar_should_on_left()),
                        match sender_now {
                            Some(Sender::Others(uuid)) => {
                                crate::ui::assets::get_avatar(&baker_state.operators.read()[&uuid].avatar)
                            }
                            _ => crate::ui::assets::get_avatar("endministratorf"),
                        },
                        temporary,
                    ));
                }
                if peek.is_none() {
                    break;
                }
                temporary = vec![(
                    *peek.unwrap().0,
                    peek.unwrap().1.content.clone(),
                    peek.unwrap().1.animation,
                )];
                sender_now = peek.map(|x| x.1.sender);
            }
            iter.next();
        }
    }

    rsx! {
        div { id: "session-main-content",
            for (avatar_on_left , avatar , messages) in messages {
                MessageRow {
                    avatar_on_left,
                    avatar,
                    messages,
                    on_delete_message: move |(session_uuid, message_id)| {
                        spawn(async move {
                            crate::database::delete_message(session_uuid, message_id).await.unwrap();
                        });
                    },
                }
            }
        }
    }
}

#[component]
fn InputArea(
    input_area_message_type: Signal<InputAreaMessageType>,
    input_area_text: Signal<String>,
    with_more_menu_open: Signal<bool>,
    with_sender_selector_message_type: Signal<bool>,
    with_sender_selector_open: Signal<bool>,
    on_submit: EventHandler<crate::Sender>,
) -> Element {
    let mut baker_state = use_context::<crate::BakerState>();
    let participants_ids_count = baker_state
        .sessions
        .get(&baker_state.current_session.unwrap())
        .unwrap()
        .participants_ids
        .len();
    let first_participant = *baker_state
        .sessions
        .get(&baker_state.current_session.unwrap())
        .unwrap()
        .participants_ids
        .first()
        .unwrap();

    let on_submit_click = move |evt: Event<MouseData>| match evt.modifiers().ctrl() {
        true => match participants_ids_count {
            1 => on_submit.call(Sender::Others(first_participant)),
            _ => with_sender_selector_open.set(true),
        },
        false => on_submit.call(Sender::Endministrator),
    };

    let input_area_style = if with_more_menu_open() {
        "input-area-with-menu flex flex-row"
    } else {
        "flex flex-row"
    };

    use_effect(move || {
        baker_state.current_session.read();
        with_more_menu_open.set(false);
    });

    rsx! {
        div { id: "input-area", class: input_area_style.to_string(),
            div { id: "input-area-input",
                input {
                    id: "input-area-input-input",
                    oninput: move |evt| { input_area_text.set(evt.value()) },
                    onkeypress: move |evt: Event<KeyboardData>| {
                        if evt.code() == Code::Enter {
                            match evt.modifiers().ctrl() {
                                true => {
                                    match participants_ids_count {
                                        1 => on_submit.call(Sender::Others(first_participant)),
                                        _ => with_sender_selector_open.set(true),
                                    }
                                }
                                false => on_submit.call(Sender::Endministrator),
                            }
                        }
                    },
                    r#type: "text",
                    value: input_area_text,
                }
            }
            button { id: "input-area-submit", onclick: on_submit_click }
            button { id: "input-area-stickers", class: "input-area-button" }
            button {
                id: "input-area-more",
                class: if with_more_menu_open() { "input-area-more-selected input-area-button" } else { "input-area-button" },
                onclick: move |_| {
                    if with_more_menu_open() {
                        with_more_menu_open.set(false);
                    } else {
                        with_more_menu_open.set(true);
                    }
                },
            }

            if with_sender_selector_open() {
                selector::Selector {
                    kv: baker_state
                        .sessions
                        .get(&baker_state.current_session.unwrap())
                        .unwrap()
                        .participants_ids
                        .iter()
                        .map(|x| (Some(*x), baker_state.operators.get(x).unwrap().name.clone()))
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

            if let crate::InputAreaMode::Insert { .. } = *baker_state.input_area_mode.read() {
                div { id: "insert-mode-wrapper",
                    span { "插入模式" }
                    button {
                        onclick: move |_| {
                            baker_state.input_area_mode.set(crate::InputAreaMode::Normal);
                        },
                        "×"
                    }
                }
            } else if let crate::InputAreaMode::Modify { .. } = *baker_state.input_area_mode.read() {
                div { id: "insert-mode-wrapper", class: "modify-mode",
                    span { "修改模式" }
                    button {
                        onclick: move |_| {
                            baker_state.input_area_mode.set(crate::InputAreaMode::Normal);
                        },
                        "×"
                    }
                }
            }
        }
    }
}

#[component]
fn MoreMenu(
    current_session: Signal<Option<Uuid>>,
    with_more_menu_open: Signal<bool>,
    with_sender_selector_open: Signal<bool>,
    input_area_message_type: Signal<InputAreaMessageType>,
    on_submit: EventHandler<Sender>,
) -> Element {
    let mut baker_state = use_context::<crate::BakerState>();
    let session_id = current_session.unwrap();
    let session_name = baker_state.sessions.get(&session_id).unwrap().session_name.clone();
    let participants_ids = use_signal(|| {
        baker_state
            .sessions
            .get(&session_id)
            .unwrap()
            .participants_ids
            .iter()
            .cloned()
            .collect()
    });

    let onchange = move |evt: Event<FormData>| {
        if let Some(file_data) = evt.files().first() {
            let file = file_data.get_web_file().unwrap();

            spawn(async move {
                let uuid = Uuid::new_v4();
                crate::database::save_multimedia(uuid, file.into()).await.unwrap();
                input_area_message_type.set(InputAreaMessageType::Image(uuid));
                // on_submit.call(None);
                with_sender_selector_open.set(true);
                with_more_menu_open.set(false);
            });
        }
    };

    let mut animation_end = use_signal(|| false);

    rsx! {
        div {
            id: "more-menu-wrapper",
            onanimationend: move |_| {
                animation_end.set(true);
            },
            if animation_end() {
                div { id: "more-menu", class: "menu",
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
                            baker_state.sessions.write().get_mut(&session_id).unwrap().session_name = evt
                                .value();
                        },
                        {session_name.to_string()}
                    }
                    div { class: "more-menu-actions",
                        span {
                            onclick: move |_| {
                                baker_state.sessions.write().retain(|k, _| { *k != session_id });
                                baker_state.current_session.set(None);
                                wasm_bindgen_futures::spawn_local(async move {
                                    crate::database::delete_session_messages(session_id).await.unwrap();
                                });
                            },
                            "删除此会话（消息会永久消失！）"
                        }
                    }
                    h3 { "干员管理" }
                    super::ParticipantsSelection { participants_ids }
                }
            }
        }
    }
}

#[component]
fn MessageRow(
    avatar_on_left: bool,
    avatar: Asset,
    messages: Vec<(u64, crate::MessageType, bool)>,
    on_delete_message: EventHandler<(Uuid, u64)>,
) -> Element {
    if messages.is_empty() {
        return rsx! {};
    }

    let mut baker_state = use_context::<crate::BakerState>();
    let message_id = messages[0].0;

    // 用于分隔符等的 Actions 菜单
    let action = move || {
        let delete_messages = move |_| {
            let session_uuid = baker_state.current_session.unwrap();
            let message_id = message_id;

            on_delete_message.call((session_uuid, message_id));

            let messages = baker_state.messages.as_mut();
            messages.unwrap().remove(&message_id);

            baker_state.input_area_mode.set(crate::InputAreaMode::Normal);
        };

        let on_prepare_to_modify = move |_| {
            baker_state
                .input_area_mode
                .set(crate::InputAreaMode::Modify { id: message_id });
        };

        let on_prepare_to_insert = move |_| {
            baker_state
                .input_area_mode
                .set(crate::InputAreaMode::Insert { id: message_id });
        };

        rsx! {
            span { class: "special-actions",
                span { onclick: delete_messages, "⤷ 删除" }
                span { onclick: on_prepare_to_modify, "修改" }
                span { onclick: on_prepare_to_insert, "在此前插入消息" }
            }
        }
    };

    match &messages[0].1 {
        crate::MessageType::HorizontalBreak => rsx! {
            div { class: "horizontal-break",
                span {}
                {action()}
            }
        },
        crate::MessageType::State(txt) => rsx! {
            div { class: "state",
                span { {txt.to_string()} }
                {action()}
            }
        },
        crate::MessageType::StateWithHorizontalLine(txt) => rsx! {
            div { class: "state-with-hl",
                span {}
                span { {txt.to_string()} }
                span {}
                {action()}
            }
        },
        crate::MessageType::Text(_) | crate::MessageType::Image(_) => {
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
                                on_delete_message,
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
fn MessageBubble(
    avatar_on_left: bool,
    message: (u64, crate::MessageType, bool),
    on_delete_message: EventHandler<(Uuid, u64)>,
) -> Element {
    let mut baker_state = use_context::<crate::BakerState>();

    let message_id = message.0;

    let delete_messages = move |_| {
        let session_uuid = baker_state.current_session.unwrap();
        let message_id = message_id;

        on_delete_message.call((session_uuid, message_id));

        let messages = baker_state.messages.as_mut();
        messages.unwrap().remove(&message_id);

        baker_state.input_area_mode.set(crate::InputAreaMode::Normal);
    };

    let on_prepare_to_modify = move |_| {
        baker_state
            .input_area_mode
            .set(crate::InputAreaMode::Modify { id: message.0 });
    };

    let on_prepare_to_insert = move |_| {
        baker_state
            .input_area_mode
            .set(crate::InputAreaMode::Insert { id: message.0 });
    };

    // TODO: 添加 修改 和 插入消息的功能
    let message_actions_left = rsx! {
        span { class: "actions actions-left",
            span { onclick: delete_messages, "删除" }
            span { onclick: on_prepare_to_modify, "修改" }
            span { onclick: on_prepare_to_insert, "在此前插入消息" }
        }
    };

    let message_actions_right = rsx! {
        span { class: "actions",
            span { onclick: on_prepare_to_modify, "修改" }
            span { onclick: on_prepare_to_insert, "在此前插入消息" }
            span { onclick: delete_messages, "删除" }
        }
    };

    let bubble_class = if avatar_on_left {
        if message.2 {
            "message-bubble-others message-bubble-animate-left"
        } else {
            "message-bubble-others"
        }
    } else {
        if message.2 {
            "message-bubble-self message-bubble-animate-right"
        } else {
            "message-bubble-self"
        }
    };

    rsx! {
        div { class: "message-bubble-wrapper",
            if !avatar_on_left {
                {message_actions_right}
            }
            match message.1 {
                crate::MessageType::Text(txt) => rsx! {
                    span { class: bubble_class, {txt} }
                },
                crate::MessageType::Image(uuid) => rsx! {
                    span { class: "{bubble_class} message-bubble-image",
                        Image { uuid }
                    }
                },
                _ => unreachable!(),
            }

            if avatar_on_left {
                {message_actions_left}
            }
        }
    }
}
