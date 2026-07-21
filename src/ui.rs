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

        for (_uuid , dialog) in dialogs.iter() {
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
            key: "{uuid}",
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
pub(crate) fn DialogNewSession(
    session_name: Signal<String>,
    participants_ids: Signal<fnv::FnvHashSet<Uuid>>,
    on_confirm: EventHandler,
    uuid: Uuid,
) -> Element {
    let mut baker_state = use_context::<crate::BakerState>();
    let operators = (*baker_state.operators.read()).clone();

    rsx! {
        Dialog { title: "添加新会话", on_confirm, uuid,
            div { id: "new-sessions-dialog", class: "flex flex-column",
                button {
                    id: "button-new-operator",
                    onclick: move |_| {
                        let uuid_neo = Uuid::new_v4();
                        baker_state.dialogs.write().insert(uuid_neo, rsx! {
                            DialogManageOperators { uuid: uuid_neo }
                        });
                        baker_state.dialogs.write().remove(&uuid);
                    },
                    "添加新干员"
                }

                input {
                    class: "form-input",
                    placeholder: "会话名",
                    value: session_name,
                    onchange: move |evt| {
                        *session_name.write() = evt.value();
                    },
                }

                div { class: "participants",
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
}

#[component]
pub(crate) fn DialogManageOperators(uuid: Uuid) -> Element {
    let mut baker_state = use_context::<crate::BakerState>();
    let mut name = use_signal(String::new);

    let mut edit_selected_operator_id = use_signal(|| None);
    let mut edit_selected_operator_name = use_signal(String::new);

    rsx! {
        Dialog { title: "管理干员列表", on_confirm: move |_| {}, uuid,

            div { id: "new-operator-dialog", class: "flex flex-column",
                // 添加新干员的输入区
                div { class: "flex flex-row",
                    input {
                        class: "form-input",
                        placeholder: "干员名",
                        value: name,
                        onchange: move |evt| {
                            *name.write() = evt.value();
                        },
                    }
                    button {
                        class: "dialog-buttons-confirm",
                        onclick: move |_| {
                            let trimmed = name.read().trim().to_owned();
                            if !trimmed.is_empty() {
                                baker_state
                                    .operators
                                    .write()
                                    .insert(Uuid::new_v4(), crate::Operator { name: trimmed });
                                name.write().clear();
                            }
                        },
                        "添加"
                    }
                }

                // 已有干员列表
                div { class: "participants",
                    for (id , op) in baker_state.operators.read().iter() {
                        div { class: "participant participant-setting flex flex-row",
                            span { class: "flex-1", "{op.name}" }
                            span { class: "actions-participant-setting",
                                span {
                                    onclick: {
                                        let id = *id;
                                        move |_| {
                                            baker_state.operators.write().remove(&id);
                                        }
                                    },
                                    "删除此干员"
                                }
                                span {
                                    onclick: {
                                        let id = *id;
                                        move |_| {
                                            edit_selected_operator_id.set(Some(id));
                                        }
                                    },
                                    "改名"
                                }
                            }
                            if let Some(selected_id) = edit_selected_operator_id() {
                                if selected_id == *id {
                                    div { class: "edit-operator",
                                        input {
                                            r#type: "text",
                                            value: "{edit_selected_operator_name()}",
                                            onchange: move |evt| {
                                                edit_selected_operator_name.set(evt.value());
                                            },
                                        }
                                        button {
                                            onclick: {
                                                let id = *id;
                                                move |_| {
                                                    edit_selected_operator_id.set(None);
                                                }
                                            },
                                            "取消"
                                        }
                                        button {
                                            onclick: {
                                                let id = *id;
                                                move |_| {
                                                    // TODO: Unicode 规范化
                                                    edit_selected_operator_id.set(None);
                                                    if edit_selected_operator_name.is_empty() {
                                                        return;
                                                    }
                                                    baker_state.operators.get_mut(&id).unwrap().name = edit_selected_operator_name();
                                                    edit_selected_operator_name.clear();
                                                }
                                            },
                                            "确定"
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
