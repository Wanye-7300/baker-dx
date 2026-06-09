use dioxus::prelude::*;

#[component]
pub(super) fn SessionUI() -> Element {
    let baker_state = use_context::<crate::BakerState>();
    let sessions = baker_state.sessions;
    let current_session = baker_state.current_session;

    if current_session.read().is_some() {
        // TODO: 虽然 current_session.read().unwrap() 正常情况下是保证正确的 —— 但是谁知道呢？SessionMainContent 与 InputArea 同
        let current_session_name = sessions.read()[&current_session.read().unwrap()]
            .session_name
            .clone();

        rsx! {
            div { id: "session", class: "flex flex-column",
                div { class: "flex flex-column", id: "session-header",
                    span { {current_session_name} }
                }
                div { id: "session-main", class: "flex flex-column",
                    SessionMainContent {}
                    InputArea {}
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
    let sessions = baker_state.sessions;

    let current_session = baker_state.current_session;
    let current_session = &sessions.read()[&current_session.read().unwrap()];
    let mut iter = current_session.messages.iter().peekable();

    let mut messages = vec![];
    let mut temporary = vec![]; // 用于判断一组消息是不是一个人发的，然后塞进 messages
    let mut sender_uuid_now = iter.peek().and_then(|x| x.sender);

    use_effect(move || {
        if !*baker_state.need_to_scroll_down.read() {
            return;
        }

        spawn(async {
            let _ = document::eval(
                "\n\
            let element = document.querySelector('#session-main-content');\n\
            element.scroll(0, element.scrollHeight);",
            )
            .await;
        });

        *baker_state.need_to_scroll_down.write() = false;
    });

    loop {
        let peek = iter.peek();
        if peek.is_some_and(|x| x.sender == sender_uuid_now) {
            temporary.push(peek.unwrap().content.clone());
        } else {
            if !temporary.is_empty() {
                messages.push((sender_uuid_now.is_some(), temporary));
            }
            if peek.is_none() {
                break;
            }
            temporary = vec![peek.unwrap().content.clone()];
            sender_uuid_now = peek.and_then(|x| x.sender);
        }
        iter.next();
    }

    rsx! {
        div { id: "session-main-content",
            for (avatar_on_left , messages) in messages {
                MessageRow { avatar_on_left, messages }
            }
        }
    }
}

#[component]
fn InputArea() -> Element {
    let mut baker_state = use_context::<crate::BakerState>();

    let mut value = use_signal(String::new);

    let mut submit = move |ctrl: bool| {
        if value.is_empty() {
            return;
        }

        let mut sessions = baker_state.sessions;
        let current_session = baker_state.current_session;

        let sender_uuid = if ctrl {
            // 如果 participants_ids 里没有干员了，那就让 Endministrator 顶替下先（None）
            sessions
                .read()
                .get(&current_session.read().unwrap())
                .unwrap()
                .participants_ids
                .first()
                .copied()
        } else {
            None
        };

        sessions
            .write()
            .get_mut(&current_session.read().unwrap())
            .unwrap()
            .messages
            .push(crate::Message {
                sender: sender_uuid,
                content: value(),
            });
        value.set(String::new());

        *baker_state.need_to_scroll_down.write() = true;
    };

    let on_submit_click = move |evt: Event<MouseData>| {
        submit(evt.modifiers().ctrl());
    };

    rsx! {
        div { id: "input-area", class: "flex flex-row",
            input {
                id: "input-area-input",
                oninput: move |evt| { value.set(evt.value()) },
                onkeypress: move |evt: Event<KeyboardData>| {
                    if evt.code() == Code::Enter {
                        submit(evt.modifiers().ctrl());
                    }
                },
                r#type: "text",
                value,
            }
            button { onclick: on_submit_click, "Submit" }
        }
    }
}

#[component]
fn MessageRow(avatar_on_left: bool, messages: Vec<String>) -> Element {
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
                    img { src: crate::AVATAR_PERLICA }
                }
            }
            // 中间消息
            div { class: if avatar_on_left { "message-row-content message-row-content-left" } else { "message-row-content message-row-content-right" },
                for message in messages {
                    MessageBubble { avatar_on_left, message }
                }
            }
            // 右侧头像
            div { class: avatar_right_class,
                if !avatar_on_left {
                    img { src: crate::AVATAR_ENDMINF }
                }
            }
        }
    }
}

#[component]
fn MessageBubble(avatar_on_left: bool, message: String) -> Element {
    rsx! {
        div { class: "message-bubble-wrapper",
            span { class: if avatar_on_left { "message-bubble-others" } else { "message-bubble-self" },
                {message}
            }
        }
    }
}
