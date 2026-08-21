//! Baker：《明日方舟：终末地》二创制作工具
//!
//! > [!WARNING]
//! > 这个分支用于重写整个项目，目前还处在早期开发中。
//! > 目前仅支持 Web Platform。

use crate::{operator::model::Avatar, ui::Baker};
use dioxus::prelude::*;
use uuid::Uuid;

mod settings;
mod ui;

mod operator;
mod session;
mod shared;

#[derive(Clone, Debug)]
struct BakerState {
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
const SELECTOR_CSS: Asset = asset!("/assets/styling/selector.css");
const MENU_CSS: Asset = asset!("/assets/styling/menu.css");

const FONT_THIN: Asset = asset!("/assets/HarmonyOS_Sans_Thin.ttf");
const FONT_LIGHT: Asset = asset!("/assets/HarmonyOS_Sans_Light.ttf");
const FONT_REGULAR: Asset = asset!("/assets/HarmonyOS_Sans_Regular.ttf");
const FONT_MEDIUM: Asset = asset!("/assets/HarmonyOS_Sans_Medium.ttf");
const FONT_BOLD: Asset = asset!("/assets/HarmonyOS_Sans_Bold.ttf");
const FONT_BLACK: Asset = asset!("/assets/HarmonyOS_Sans_Black.ttf");
const FONT_BENDER: Asset = asset!("/assets/bender.otf");
const AVATAR_BACKGROUND: Asset = asset!("/assets/extracted/mask/mask_snscharentry_head.png");
const AVATAR_FRAME: Asset = asset!("/assets/extracted/bg/bg_snscharentry_head_Line.png");

const MESSAGE_BUBBLE_SELF: Asset = asset!("/assets/deco/bg_message_right.png");
const MESSAGE_BUBBLE_OTHERS: Asset = asset!("/assets/deco/bg_message_left.png");
const SESSION_TITLE_LEFT_BAR: Asset = asset!("/assets/deco/session_title_left_bar.png");
const SESSION_TITLE_RIGHT_BAR: Asset = asset!("/assets/deco/session_title_right_bar.png");
const ICON_SNS_MESSAGE_02: Asset = asset!("/assets/extracted/icon/icon_sns_message_02.png");
const ICON_SNS_CHAT_EMOTICON: Asset = asset!("/assets/deco/input_area_emoticon.png");
const ICON_SNS_CHAT_EMOTICON_SELECTED: Asset = asset!("/assets/deco/input_area_emoticon_selected.png");
const INPUT_AREA_MORE: Asset = asset!("/assets/deco/input_area_more.png");
const INPUT_AREA_MORE_SELECTED: Asset = asset!("/assets/deco/input_area_more_selected.png");
const DECO_SNS_TWEET_DECORATE_10: Asset = asset!("/assets/extracted/decorate/deco_sns_tweet_decorate_10.png");
const DECO_SNS_TWEET_DECORATE_11: Asset = asset!("/assets/extracted/decorate/deco_sns_tweet_decorate_11.png");
const LINE_SNS_TWEET_DECORATE: Asset = asset!("/assets/extracted/decorate/line_sns_tweet_decorate.png");
const DECO_SNS_TWEET_DECORATE_02: Asset = asset!("/assets/extracted/decorate/deco_sns_tweet_decorate_02.png");
const SNS_LIST_DECORATE_2: Asset = asset!("/assets/extracted/decorate/sns_list_decorate_2.png");
const DECO_SNS_TWEET_DECORATE: Asset = asset!("/assets/extracted/decorate/deco_sns_tweet_decorate.png");
const ACHIEVEMENT_MAIN_DECO05: Asset = asset!("/assets/deco/achievement_main_deco05.png");

fn main() {
    dioxus::launch(App);
}

fn provide_baker_state() {
    let dialogs = use_signal(fnv::FnvHashMap::default);

    use_context_provider(|| BakerState { dialogs });
}

fn provide_settings() {
    let image = use_signal(|| shared::utils::get_item_or_default("wallpaper", || None).unwrap_or(None));
    let endministrator_avatar = use_signal(|| {
        shared::utils::get_item_or_default("E_avatar", || Avatar::Preset("endministratorf".to_owned()))
            .unwrap_or(Avatar::Preset("endministratorf".to_owned()))
    });

    use_context_provider(|| settings::state::SettingsState {
        image,
        endministrator_avatar,
    });
}

#[component]
fn App() -> Element {
    provide_baker_state();
    provide_settings();
    crate::session::view_model::input_view_model::InputViewModel::use_input_view_model_provider();
    crate::session::view_model::session_view_model::SessionViewModel::use_session_view_model_provider().unwrap();
    crate::session::view_model::session_view_model::SessionUIViewModel::use_session_ui_view_model_provider();
    crate::operator::view_model::OperatorViewModel::use_operator_view_model_provider().unwrap();

    let _database = use_resource(|| async { shared::database::open_db().await });

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
        FONT_THIN, FONT_LIGHT, FONT_REGULAR, FONT_MEDIUM, FONT_BOLD, FONT_BLACK,
    );

    let font_face_bender = format!(
        r#"
        @font-face {{
            font-family: 'Bender';
            src: url('{}') format('opentype');
            font-weight: normal;
            font-style: normal;
        }}"#,
        FONT_BENDER
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
    let input_area_more_selected_bundled_path = INPUT_AREA_MORE_SELECTED.bundled();
    let input_area_more_selected_bundled_path = input_area_more_selected_bundled_path.bundled_path();

    rsx! {
        document::Link { rel: "icon", href: FAVICON, r#type: "image/x-icon" }
        document::Link { rel: "stylesheet", href: NORMALIZE_CSS }
        document::Style { {font_face} }
        document::Style { {font_face_bender} }
        document::Style {
            ":root {{ --avatar-background: url(\"{avatar_background_bundled_path}\"); --avatar-frame: url(\"{avatar_frame_bundled_path}\"); --message-bubble-self: url(\"{message_bubble_self_bundled_path}\"); --message-bubble-others: url(\"{message_bubble_others_bundled_path}\"); --session-title-left-bar: url(\"{session_title_left_bar_bundled_path}\"); --session-title-right-bar: url(\"{session_title_right_bar_bundled_path}\"); --icon-sns-chat-emoticon: url(\"{icon_sns_chat_emoticon_bundled_path}\"); --icon-sns-chat-emoticon-selected: url(\"{ICON_SNS_CHAT_EMOTICON_SELECTED.bundled().bundled_path()}\"); --icon-sns-message-02: url(\"{icon_sns_message_02_bundled_path}\"); --input-area-more: url(\"{input_area_more_bundled_path}\"); --input-area-more-selected: url(\"{input_area_more_selected_bundled_path}\"); --deco_sns_tweet_decorate_10: url(\"{DECO_SNS_TWEET_DECORATE_10.bundled().bundled_path()}\"); --deco_sns_tweet_decorate_02: url(\"{DECO_SNS_TWEET_DECORATE_02.bundled().bundled_path()}\"); --sns_list_decorate_2: url(\"{SNS_LIST_DECORATE_2.bundled().bundled_path()}\"); --deco_sns_tweet_decorate: url(\"{DECO_SNS_TWEET_DECORATE.bundled().bundled_path()}\"); --achievement_main_deco05: url(\"{ACHIEVEMENT_MAIN_DECO05.bundled().bundled_path()}\"); }}"
        }
        document::Link { rel: "stylesheet", href: MAIN_CSS }
        document::Link { rel: "stylesheet", href: SELECTOR_CSS }
        document::Link { rel: "stylesheet", href: MENU_CSS }

        if shared::database::is_ready() {
            Router::<Route> {}
        } else {
            div { id: "database-loading", class: "flex", "加载数据库" }
        }
    }
}
