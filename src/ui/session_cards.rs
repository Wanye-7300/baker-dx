use dioxus::prelude::*;
use uuid::Uuid;

#[component]
pub(super) fn SessionCards() -> Element {
    let mut baker_state = use_context::<crate::BakerState>();

    let session_name = use_signal(String::new);
    let participants_ids = use_signal(fnv::FnvHashSet::default);

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
                        super::DialogNewSession {
                            session_name,
                            participants_ids,
                            on_confirm: move |_| {
                                baker_state
                                    .sessions
                                    .write()
                                    .insert(
                                        Uuid::new_v4(),
                                        crate::Session {
                                            session_name: session_name(),
                                            participants_ids: participants_ids
                                                .read()
                                                .iter()
                                                .cloned()
                                                .collect::<Vec<Uuid>>(),
                                            messages: vec![],
                                        },
                                    );
                                let mut baker_state = use_context::<crate::BakerState>();
                                baker_state.dialogs.write().remove(&uuid);
                            },
                            uuid,
                        }
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
