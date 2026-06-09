use dioxus::prelude::*;
use uuid::Uuid;

#[component]
pub(super) fn SessionCards() -> Element {
    let baker_state = use_context::<crate::BakerState>();

    rsx! {
        div { id: "session-cards",
            for (uuid , session) in baker_state.sessions.read().iter() {
                Card { uuid: *uuid, name: session.session_name.clone() }
            }
        }
    }
}

#[component]
pub(super) fn Card(uuid: Uuid, name: String) -> Element {
    let mut baker_state = use_context::<crate::BakerState>();

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
