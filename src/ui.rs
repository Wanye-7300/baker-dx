use dioxus::prelude::*;
use uuid::Uuid;

mod assets;
mod session;
mod session_cards;

#[component]
pub(super) fn Baker() -> Element {
    let baker_state = use_context::<crate::BakerState>();
    let dialogs = baker_state.dialogs.read();
    
    let session_name = use_signal(String::new);
    let participants_ids = use_signal(fnv::FnvHashSet::default);

    rsx! {
        div { id: "app", class: "flex flex-column",
            div { id: "title", "// BAKER / Messages" }
            div { id: "main-content", class: "flex flex-row",
                session_cards::SessionCards { session_name, participants_ids }
                session::SessionUI {}
            }
        }

        for dialog in dialogs.values() {
            {dialog}
        }
    }
}

#[component]
pub(crate) fn Dialog(title: String, on_confirm: EventHandler, uuid: Uuid, children: Element) -> Element {
    let mut baker_state = use_context::<crate::BakerState>();
    
    rsx! {
        div {
            class: "backdrop",
            onclick: move |_| {
                baker_state.dialogs.write().remove(&uuid);
            },
            div {
                key: "{uuid.to_string()}",
                class: "dialog flex flex-column",
                onclick: move |e| {
                    e.stop_propagation();
                },
                div { class: "dialog-title flex flex-row",
                    {title}
                    button {
                        class: "dialog-title-close",
                        onclick: move |_| {
                            baker_state.dialogs.write().remove(&uuid);
                        },
                        "×"
                    }
                }
                div { class: "dialog-content", {children} }
                div { class: "dialog-buttons flex flex-row",
                    button {
                        class: "dialog-buttons-confirm",
                        onclick: move |_| on_confirm.call(()),
                        "好"
                    }
                }
            }
        }
    }
}

#[component]
pub(crate) fn DialogNewSession(session_name: Signal<String>, participants_ids: Signal<fnv::FnvHashSet<Uuid>>, on_confirm: EventHandler, uuid: Uuid) -> Element {
    let baker_state = use_context::<crate::BakerState>();
    let operators = (*baker_state.operators.read()).clone();

    rsx! {
        Dialog { title: "添加新会话", on_confirm, uuid,
            div { id: "new-sessions-dialog", class: "flex flex-column",
                input {
                    class: "form-input",
                    placeholder: "会话名",
                    value: session_name,
                    onchange: move |evt| {
                        *session_name.write() = evt.value();
                    },
                }

                for (k , v) in operators {
                    div {
                        class: "participant",
                        onclick: move |_| {
                            if participants_ids.read().get(&k).is_none() {
                                participants_ids.write().insert(k);
                            } else {
                                participants_ids.write().remove(&k);
                            }
                        },
                        input {
                            r#type: "checkbox",
                            id: k.to_string(),
                            name: k.to_string(),
                            checked: participants_ids.read().get(&k).is_some(),
                            onchange: move |_| {
                                if participants_ids.read().get(&k).is_none() {
                                    participants_ids.write().insert(k);
                                } else {
                                    participants_ids.write().remove(&k);
                                }
                            },
                        }
                        label { r#for: k.to_string(),
                            {v.name.clone()}
                            {"   "}
                            {k.to_string()}
                        }
                    }
                }
            }
        }
    }
}
