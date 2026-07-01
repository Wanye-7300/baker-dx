//! Baker：《明日方舟：终末地》二创制作工具
//!
//! > [!WARNING]
//! > 这个分支用于重写整个项目，目前还处在早期开发中。

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
    messages: collections::BTreeMap<u64, Message>,
    // 下次 push 消息时应该插入的编号，然后 +1
    id: u64,
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
        operators.insert(
            Uuid::new_v4(),
            Operator {
                name: "Chen Qianyu".to_owned(),
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
