//! Baker：《明日方舟：终末地》二创制作工具
//!
//! > [!WARNING]
//! > 这个分支用于重写整个项目，目前还处在早期开发中。

use dioxus::prelude::*;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

mod utils;

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Operator {
    name: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Message {
    sender: Option<Uuid>,
    content: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Session {
    session_name: String,
    participants_ids: Vec<Uuid>,
    messages: Vec<Message>,
}

#[derive(Clone, Debug)]
struct BakerState {
    operators: Signal<fnv::FnvHashMap<Uuid, Operator>>,
    sessions: Signal<fnv::FnvHashMap<Uuid, Session>>,
    current_session: Signal<Option<Uuid>>,
    need_to_scroll_down: Signal<bool>,
}

#[derive(Debug, Clone, Routable, PartialEq)]
#[rustfmt::skip]
enum Route {
    #[route("/")]
    Baker {},
}

const FAVICON: Asset = asset!("/assets/favicon.ico");
const NORMALIZE_CSS: Asset = asset!("/assets/styling/normalize.css");
const MAIN_CSS: Asset = asset!("/assets/styling/main.css");

const FONT: Asset = asset!("/assets/SourceHanSansSC-Regular.otf");
const FONT_BENDER: Asset = asset!("/assets/bender.otf");

const AVATAR_ENDMINF: Asset =
    asset!("/assets/extracted/avatar/operator/icon_round_chr_0003_endminf.png");
const AVATAR_PERLICA: Asset =
    asset!("/assets/extracted/avatar/operator/icon_round_chr_0004_pelica.png");

const AVATAR_BACKGROUND: Asset = asset!("/assets/extracted/mask/mask_snscharentry_head.png");
const AVATAR_FRAME: Asset = asset!("/assets/extracted/bg/bg_snscharentry_head_Line.png");

const MESSAGE_BUBBLE_SELF: Asset = asset!("/assets/deco/bg_snscontenttextorpic_chat_04.png");
const MESSAGE_BUBBLE_OTHERS: Asset = asset!("/assets/deco/bg_snscontenttextorpic_chat.png");

fn main() {
    dioxus::launch(App);
}

fn provide_baker_state() {
    let (perlica_uuid, session_uuid) = (Uuid::new_v4(), Uuid::new_v4());

    let default_operators = || {
        let mut operators = fnv::FnvHashMap::default();
        operators.insert(
            perlica_uuid,
            Operator {
                name: "Perlica".to_owned(),
            },
        );
        operators
    };

    let default_sessions = || {
        let mut sessions = fnv::FnvHashMap::default();
        sessions.insert(
            session_uuid,
            Session {
                session_name: "Perlica".to_owned(),
                participants_ids: vec![perlica_uuid],
                messages: vec![],
            },
        );
        sessions
    };

    let operators = utils::get_item_or_default("operators", default_operators);
    let sessions = utils::get_item_or_default("sessions", default_sessions);

    let current_session = use_signal(|| None);

    let operators = use_signal(|| operators);
    let sessions = use_signal(|| sessions);

    let need_to_scroll_down = use_signal(|| false);
    use_context_provider(|| BakerState {
        operators,
        sessions,
        current_session,
        need_to_scroll_down,
    });
}

#[component]
fn App() -> Element {
    provide_baker_state();

    let baker_state = use_context::<BakerState>();
    use_effect(move || {
        let operators = baker_state.operators.read();
        let sessions = baker_state.sessions.read();

        utils::set_item("operators", &*operators);
        utils::set_item("sessions", &*sessions);
    });

    let font_face = format!(
        r#"
        @font-face {{
            font-family: 'Source Han Sans SC';
            src: url('{}') format('opentype');
            font-weight: normal;
            font-style: normal;
        }}"#,
        FONT.bundled().bundled_path()
    );

    let font_face_bender = format!(
        r#"
        @font-face {{
            font-family: 'Bender';
            src: url('{}') format('opentype');
            font-weight: normal;
            font-style: normal;
        }}"#,
        FONT_BENDER.bundled().bundled_path()
    );

    let avatar_background_bundled_path = AVATAR_BACKGROUND.bundled();
    let avatar_background_bundled_path = avatar_background_bundled_path.bundled_path();
    let avatar_frame_bundled_path = AVATAR_FRAME.bundled();
    let avatar_frame_bundled_path = avatar_frame_bundled_path.bundled_path();
    let message_bubble_self_bundled_path = MESSAGE_BUBBLE_SELF.bundled();
    let message_bubble_self_bundled_path = message_bubble_self_bundled_path.bundled_path();
    let message_bubble_others_bundled_path = MESSAGE_BUBBLE_OTHERS.bundled();
    let message_bubble_others_bundled_path = message_bubble_others_bundled_path.bundled_path();

    rsx! {
        document::Link { rel: "icon", href: FAVICON, r#type: "image/x-icon" }
        document::Link { rel: "stylesheet", href: NORMALIZE_CSS }
        document::Style { {font_face} }
        document::Style { {font_face_bender} }
        document::Style {
            ":root {{ --avatar-background: url(\"{avatar_background_bundled_path}\"); --avatar-frame: url(\"{avatar_frame_bundled_path}\"); --message-bubble-self: url(\"{message_bubble_self_bundled_path}\"); --message-bubble-others: url(\"{message_bubble_others_bundled_path}\"); }}"
        }
        document::Link { rel: "stylesheet", href: MAIN_CSS }

        Router::<Route> {}
    }
}

#[component]
fn Baker() -> Element {
    rsx! {
        div { id: "app", class: "flex flex-column",
            div { id: "title", "// BAKER / Messages" }
            div { id: "main-content", class: "flex flex-row",
                SessionCards {}
                SessionUI {}
            }
        }
    }
}

#[component]
fn SessionCards() -> Element {
    let baker_state = use_context::<BakerState>();

    rsx! {
        div { id: "session-cards",
            for (uuid , session) in baker_state.sessions.read().iter() {
                Card { uuid: *uuid, name: session.session_name.clone() }
            }
        }
    }
}

#[component]
fn Card(uuid: Uuid, name: String) -> Element {
    let mut baker_state = use_context::<BakerState>();

    let on_choose_card = move |_| {
        *baker_state.current_session.write() = Some(uuid);
        *baker_state.need_to_scroll_down.write() = true;
    };

    rsx! {
        div {
            key: "{uuid.to_string()}",
            class: "card",
            onclick: on_choose_card,
            {name}
        }
    }
}

#[component]
fn SessionUI() -> Element {
    let baker_state = use_context::<BakerState>();
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
    let mut baker_state = use_context::<BakerState>();
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
    let mut baker_state = use_context::<BakerState>();

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
            .push(Message {
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
                    img { src: AVATAR_PERLICA }
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
                    img { src: AVATAR_ENDMINF }
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
