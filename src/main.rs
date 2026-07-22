//! Baker：《明日方舟：终末地》二创制作工具
//!
//! > [!WARNING]
//! > 这个分支用于重写整个项目，目前还处在早期开发中。
//! > 目前仅支持 Web Platform。

use std::collections;

use crate::ui::Baker;
use dioxus::prelude::*;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

mod ui;
mod utils;

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Operator {
    name: String,
    avatar: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Message {
    sender: Option<Uuid>,
    content: String,
    #[serde(skip_serializing)]
    #[serde(default)]
    animation: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Session {
    session_name: String,
    participants_ids: Vec<Uuid>,
    messages: collections::BTreeMap<u64, Message>,
    // 下次 push 消息时应该插入的编号，然后 +1
    id: u64,
}

impl Session {
    fn make_no_animation(&mut self) {
        for v in self.messages.values_mut().rev() {
            if !v.animation {
                // 一般来说，只有最后几个 messages 是 true，自然遍历到 false 那么后面基本不可能有了，有了我给你了
                break;
            }
            v.animation = false;
        }
    }
}

#[derive(Clone, Debug)]
struct BakerState {
    operators: Signal<fnv::FnvHashMap<Uuid, Operator>>,
    sessions: Signal<fnv::FnvHashMap<Uuid, Session>>,
    current_session: Signal<Option<Uuid>>,
    need_to_scroll_down: Signal<bool>,
    dialogs: Signal<fnv::FnvHashMap<Uuid, Element>>,
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

const FONT_THIN: Asset = asset!("/assets/HarmonyOS_Sans_SC_Thin.ttf");
const FONT_LIGHT: Asset = asset!("/assets/HarmonyOS_Sans_SC_Light.ttf");
const FONT_REGULAR: Asset = asset!("/assets/HarmonyOS_Sans_SC_Regular.ttf");
const FONT_MEDIUM: Asset = asset!("/assets/HarmonyOS_Sans_SC_Medium.ttf");
const FONT_BOLD: Asset = asset!("/assets/HarmonyOS_Sans_SC_Bold.ttf");
const FONT_BLACK: Asset = asset!("/assets/HarmonyOS_Sans_SC_Black.ttf");
const FONT_BENDER: Asset = asset!("/assets/bender.otf");
const AVATAR_BACKGROUND: Asset = asset!("/assets/extracted/mask/mask_snscharentry_head.png");
const AVATAR_FRAME: Asset = asset!("/assets/extracted/bg/bg_snscharentry_head_Line.png");

const MESSAGE_BUBBLE_SELF: Asset = asset!("/assets/deco/bg_message_right.png");
const MESSAGE_BUBBLE_OTHERS: Asset = asset!("/assets/deco/bg_message_left.png");
const SESSION_TITLE_LEFT_BAR: Asset = asset!("/assets/deco/session_title_left_bar.png");
const SESSION_TITLE_RIGHT_BAR: Asset = asset!("/assets/deco/session_title_right_bar.png");
const ICON_SNS_MESSAGE_02: Asset = asset!("/assets/extracted/icon/icon_sns_message_02.png");
const ICON_SNS_CHAT_EMOTICON: Asset = asset!("/assets/deco/input_area_emoticon.png");
const INPUT_AREA_MORE: Asset = asset!("/assets/deco/input_area_more.png");

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
                avatar: "perlica".to_owned(),
            },
        );
        operators.insert(
            Uuid::new_v4(),
            Operator {
                name: "Chen Qianyu".to_owned(),
                avatar: "chenqy".to_owned(),
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
                messages: collections::BTreeMap::new(),
                id: 0u64,
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

    let dialogs = use_signal(fnv::FnvHashMap::default);
    use_context_provider(|| BakerState {
        operators,
        sessions,
        current_session,
        need_to_scroll_down,
        dialogs,
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
            font-family: 'HarmonyOS Sans';
            src: url('{}') format('truetype');
            font-weight: 100;
            font-style: normal;
        }}

        @font-face {{
            font-family: 'HarmonyOS Sans';
            src: url('{}') format('truetype');
            font-weight: 300;
            font-style: normal;
        }}

        @font-face {{
            font-family: 'HarmonyOS Sans';
            src: url('{}') format('truetype');
            font-weight: 400;
            font-style: normal;
        }}

        @font-face {{
            font-family: 'HarmonyOS Sans';
            src: url('{}') format('truetype');
            font-weight: 500;
            font-style: normal;
        }}

        @font-face {{
            font-family: 'HarmonyOS Sans';
            src: url('{}') format('truetype');
            font-weight: 700;
            font-style: normal;
        }}

        @font-face {{
            font-family: 'HarmonyOS Sans';
            src: url('{}') format('truetype');
            font-weight: 900;
            font-style: normal;
        }}"#,
        FONT_THIN.bundled().bundled_path(),
        FONT_LIGHT.bundled().bundled_path(),
        FONT_REGULAR.bundled().bundled_path(),
        FONT_MEDIUM.bundled().bundled_path(),
        FONT_BOLD.bundled().bundled_path(),
        FONT_BLACK.bundled().bundled_path(),
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
    let session_title_left_bar_bundled_path = SESSION_TITLE_LEFT_BAR.bundled();
    let session_title_left_bar_bundled_path = session_title_left_bar_bundled_path.bundled_path();
    let session_title_right_bar_bundled_path = SESSION_TITLE_RIGHT_BAR.bundled();
    let session_title_right_bar_bundled_path = session_title_right_bar_bundled_path.bundled_path();
    let icon_sns_chat_emoticon_bundled_path = ICON_SNS_CHAT_EMOTICON.bundled();
    let icon_sns_chat_emoticon_bundled_path = icon_sns_chat_emoticon_bundled_path.bundled_path();
    let icon_sns_message_02_bundled_path = ICON_SNS_MESSAGE_02.bundled();
    let icon_sns_message_02_bundled_path = icon_sns_message_02_bundled_path.bundled_path();
    let input_area_more_bundled_path = INPUT_AREA_MORE.bundled();
    let input_area_more_bundled_path = input_area_more_bundled_path.bundled_path();

    rsx! {
        document::Link { rel: "icon", href: FAVICON, r#type: "image/x-icon" }
        document::Link { rel: "stylesheet", href: NORMALIZE_CSS }
        document::Style { {font_face} }
        document::Style { {font_face_bender} }
        document::Style {
            ":root {{ --avatar-background: url(\"{avatar_background_bundled_path}\"); --avatar-frame: url(\"{avatar_frame_bundled_path}\"); --message-bubble-self: url(\"{message_bubble_self_bundled_path}\"); --message-bubble-others: url(\"{message_bubble_others_bundled_path}\"); --session-title-left-bar: url(\"{session_title_left_bar_bundled_path}\"); --session-title-right-bar: url(\"{session_title_right_bar_bundled_path}\"); --icon-sns-chat-emoticon: url(\"{icon_sns_chat_emoticon_bundled_path}\"); --icon-sns-message-02: url(\"{icon_sns_message_02_bundled_path}\"); --input-area-more: url(\"{input_area_more_bundled_path}\"); }}"
        }
        document::Link { rel: "stylesheet", href: MAIN_CSS }

        Router::<Route> {}
    }
}
