use dioxus::prelude::*;
use uuid::Uuid;

#[component]
pub(super) fn SessionCards(session_name: Signal<String>, participants_ids: Signal<fnv::FnvHashSet<Uuid>>) -> Element {
    let mut baker_state = use_context::<crate::BakerState>();

    rsx! {
        div { id: "session-cards", class: "flex flex-column",
            div { id: "cards-wrapper",
                for (uuid , session) in baker_state.sessions.read().iter() {
                    Card { uuid: *uuid, name: session.session_name.clone() }
                }
            }
            button {
                id: "cards-new",
                onclick: move |_| {
                    let uuid = Uuid::new_v4();
                    baker_state.dialogs.write().insert(uuid, rsx! {
                        super::DialogNewSession { session_name, participants_ids, uuid }
                    });
                },
                "添加新会话"
            }
        }
    }
}

#[component]
pub(super) fn Card(uuid: Uuid, name: String) -> Element {
    let mut baker_state = use_context::<crate::BakerState>();

    let on_choose_card = move |_| {
        if let Some(uuid) = *baker_state.current_session.read() {
            if let Some(session) = baker_state.sessions.write().get_mut(&uuid) {
                session.make_no_animation();
            }
        }
        *baker_state.current_session.write() = Some(uuid);
        *baker_state.need_to_scroll_down.write() = true;
    };

    rsx! {
        div {
            key: "{uuid.to_string()}",
            class: "card flex flex-row",
            onclick: on_choose_card,

            div { id: "card-img-wrapper",
                img { src: crate::ui::assets::get_avatar(&baker_state.sessions.get(&uuid).unwrap().avatar) }
            }

            {name}
        }
    }
}
