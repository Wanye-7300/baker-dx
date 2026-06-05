//! Baker：《明日方舟：终末地》二创制作工具
//! 
//! > [!WARNING]
//! > 这个分支用于重写整个项目，目前还处在早期开发中。

use dioxus::prelude::*;

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

fn main() {
    dioxus::launch(App);
}

#[component]
fn App() -> Element {
    let font_face = format!(
        r#"
        @font-face {{
            font-family: 'Source Han Sans SC';
            src: url('/assets/{}') format('opentype');
            font-weight: normal;
            font-style: normal;
        }}"#,
        FONT.bundled().bundled_path()
    );

    let font_face_bender = format!(
        r#"
        @font-face {{
            font-family: 'Bender';
            src: url('/assets/{}') format('opentype');
            font-weight: normal;
            font-style: normal;
        }}"#,
        FONT_BENDER.bundled().bundled_path()
    );

    rsx! {
        document::Link { rel: "icon", href: FAVICON, r#type: "image/x-icon" }
        document::Link { rel: "stylesheet", href: NORMALIZE_CSS }
        document::Style { {font_face} }
        document::Style { {font_face_bender} }
        document::Link { rel: "stylesheet", href: MAIN_CSS }

        Router::<Route> {}
    }
}

#[component]
fn Baker() -> Element {
    rsx! { "// BAKER // Messages" }
}
