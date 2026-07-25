use dioxus::{prelude::*, web::WebFileExt};
use uuid::Uuid;

pub(crate) mod selector;

#[component]
pub(super) fn SessionUI() -> Element {
    let baker_state = use_context::<crate::BakerState>();
    let sessions = baker_state.sessions;
    let current_session = baker_state.current_session;

    let with_more_menu_open = use_signal(|| false);

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
                    InputArea { with_more_menu_open }
                    if with_more_menu_open() {
                        MoreMenu { current_session }
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
        let mut sender_uuid_now = iter.peek().and_then(|x| x.1.sender);

        loop {
            let peek = iter.peek();
            if peek.is_some_and(|x| x.1.sender == sender_uuid_now) {
                temporary.push((
                    *peek.unwrap().0,
                    peek.unwrap().1.content.clone(),
                    peek.unwrap().1.animation,
                ));
            } else {
                if !temporary.is_empty() {
                    messages.push((
                        sender_uuid_now.is_some(),
                        match sender_uuid_now {
                            Some(uuid) => crate::ui::assets::get_avatar(&baker_state.operators.read()[&uuid].avatar),
                            None => crate::ui::assets::get_avatar("endministratorf"),
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
                sender_uuid_now = peek.and_then(|x| x.1.sender);
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
fn InputArea(with_more_menu_open: Signal<bool>) -> Element {
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

    let mut value = use_signal(String::new);

    let mut with_sender_selector_open = use_signal(|| false);

    let mut submit = move |sender_uuid: Option<Uuid>| {
        if value.is_empty() {
            return;
        }

        let mut sessions = baker_state.sessions;
        let current_session = baker_state.current_session.unwrap();

        let insert_id = sessions.read().get(&current_session).unwrap().id;

        let message = crate::Message {
            sender: sender_uuid,
            content: crate::MessageType::Text(value()),
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
                }
                baker_state.input_area_mode.set(crate::InputAreaMode::Normal);
            }
            crate::InputAreaMode::Modify { id } => {
                messages.insert(id, message);
                baker_state.input_area_mode.set(crate::InputAreaMode::Normal);
            }
        }

        value.set(String::new());
        *baker_state.need_to_scroll_down.write() = true;
    };

    let on_submit_click = move |evt: Event<MouseData>| match evt.modifiers().ctrl() {
        true => match participants_ids_count {
            1 => submit(Some(first_participant)),
            _ => with_sender_selector_open.set(true),
        },
        false => submit(None),
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
                    oninput: move |evt| { value.set(evt.value()) },
                    onkeypress: move |evt: Event<KeyboardData>| {
                        if evt.code() == Code::Enter {
                            match evt.modifiers().ctrl() {
                                true => {
                                    match participants_ids_count {
                                        1 => submit(Some(first_participant)),
                                        _ => with_sender_selector_open.set(true),
                                    }
                                }
                                false => submit(None),
                            }
                        }
                    },
                    r#type: "text",
                    value,
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
                        .map(|x| (*x, baker_state.operators.get(x).unwrap().name.clone()))
                        .collect(),
                    title: "选择发送者",
                    func: move |uuid| {
                        submit(Some(uuid));
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
fn MoreMenu(current_session: Signal<Option<Uuid>>) -> Element {
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
                crate::database::save_multimedia(Uuid::new_v4(), file.into())
                    .await
                    .unwrap();
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
                crate::MessageType::Text(text) => rsx! {
                    span { class: bubble_class, {text} }
                },
                crate::MessageType::Image(_uuid) => rsx! {},
            }

            if avatar_on_left {
                {message_actions_left}
            }
        }
    }
}
