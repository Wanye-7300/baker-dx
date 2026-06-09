use dioxus::prelude::*;

mod session;
mod session_cards;

#[component]
pub(super) fn Baker() -> Element {
    rsx! {
        div { id: "app", class: "flex flex-column",
            div { id: "title", "// BAKER / Messages" }
            div { id: "main-content", class: "flex flex-row",
                session_cards::SessionCards {}
                session::SessionUI {}
            }
        }
    }
}
