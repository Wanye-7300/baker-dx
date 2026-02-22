use crate::components::baker::models::{Contact, Operator};
use dioxus::prelude::*;

const LIST_NEW_SESSION: Asset = asset!("/assets/images/list_new_session.png");

#[component]
pub fn Sidebar(
    contacts: ReadSignal<Vec<Contact>>,
    operators: ReadSignal<Vec<Operator>>,
    selected_contact_id: Signal<Option<String>>,
    on_add_click: EventHandler<()>,
) -> Element {
    let contacts_list = contacts.read().clone();
    let ops_list = operators.read().clone();

    rsx! {
        div { class: "w-80 h-full flex flex-col min-h-0 bg-transparent relative",

            // 联系人列表区域
            div { class: "flex-1 overflow-y-auto space-y-3 pr-1 custom-scrollbar pb-20",
                for contact in contacts_list {
                    {
                        let operator = ops_list.iter().find(|op| op.id == contact.id).cloned();
                        rsx! {
                            ContactItem {
                                key: "{contact.id}",
                                contact: contact.clone(),
                                operator,
                                is_selected: selected_contact_id() == Some(contact.id.clone()),
                                onclick: move |_| selected_contact_id.set(Some(contact.id.clone())),
                            }
                        }
                    }
                }
            }

            // 底部添加按钮 (固定在底部)
            div { class: "absolute bottom-0 left-0 right-0 p-4 bg-gradient-to-t",
                button {
                    class: "w-full h-10 px-4 rounded-full flex items-center justify-between cursor-pointer hover:brightness-95 transition-all shadow-lg",
                    style: "background-color: rgb(238, 236, 236);",
                    onclick: move |_| on_add_click.call(()),
                    span {
                        class: "font-bold text-sm",
                        style: "color: rgb(68, 68, 68);",
                        "添加新会话"
                    }
                    img {
                        src: "{LIST_NEW_SESSION}",
                        class: "w-5 h-5 object-contain",
                    }
                }
            }
        }
    }
}

#[component]
fn ContactItem(
    contact: Contact,
    operator: Option<Operator>,
    is_selected: bool,
    onclick: EventHandler<MouseEvent>,
) -> Element {
    let display_name = if !contact.name.is_empty() {
        contact.name.clone()
    } else if let Some(op) = operator.as_ref() {
        op.name.clone()
    } else {
        "未命名会话".to_string()
    };
    let avatar_url = if !contact.avatar_url.is_empty() {
        contact.avatar_url.clone()
    } else if let Some(op) = operator.as_ref() {
        op.avatar_url.clone()
    } else {
        "".to_string()
    };
    // 基础样式
    let base_classes = "w-full h-[88px] relative rounded-xl flex items-center p-3 cursor-pointer transition-all group overflow-hidden shrink-0";

    // 选中状态样式
    let active_classes = if is_selected {
        "border-[3px] border-white/60 z-10"
    } else {
        "border border-transparent hover:bg-white/5 opacity-80 hover:opacity-100"
    };

    rsx! {
        div {
            class: "{base_classes} {active_classes}",
            style: "background-color: rgb(53, 53, 53);",
            onclick: move |evt| onclick.call(evt),

            // 选中状态下的四角装饰 (已移除)


            // 头像区域
            div { class: "relative w-[60px] h-[60px] shrink-0 mr-3",
                // 头像容器
                div { class: "w-full h-full rounded-lg overflow-hidden border border-gray-500/50 bg-gray-700 flex items-center justify-center",
                    // 这里可以用图片，暂时用首字母代替
                    if !avatar_url.is_empty() {
                        img {
                            src: "{avatar_url}",
                            class: "w-full h-full object-cover",
                        }
                    } else {
                        span { class: "text-2xl text-gray-300 font-bold",
                            "{display_name.chars().next().unwrap_or('?')}"
                        }
                    }
                }

            }

            // 信息区域
            div { class: "flex flex-col justify-center flex-1 min-w-0 h-full py-1",
                // 第一行：昵称
                div { class: "flex items-center",
                    span { class: "text-white text-lg font-bold truncate tracking-wide",
                        "{display_name}"
                    }
                }
            }
        }
    }
}
