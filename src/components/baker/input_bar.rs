use dioxus::prelude::*;

const CHAT_ENTER: Asset = asset!("/assets/images/chat_enter.png");
const CHAT_EMOJI: Asset = asset!("/assets/images/chat_emoji.png");
const CHAT_PLUS: Asset = asset!("/assets/images/chat_plus.png");

#[component]
pub fn InputBar(
    on_send: EventHandler<String>,
    on_send_other: EventHandler<String>,
    is_group: bool,
    on_send_status: EventHandler<String>,
    menu_close_token: ReadSignal<usize>,
    clear_text_token: ReadSignal<usize>,
) -> Element {
    let mut text = use_signal(String::new);
    let mut send_menu = use_signal(|| Option::<(i32, i32)>::None);

    let mut handle_submit = move || {
        let val = text.read().clone();
        if !val.trim().is_empty() {
            on_send.call(val);
            text.set(String::new());
        }
    };
    let handle_submit_other = move || {
        let val = text.read().clone();
        if !val.trim().is_empty() {
            on_send_other.call(val);
        }
    };
    let mut handle_submit_status = move || {
        let val = text.read().clone();
        if !val.trim().is_empty() {
            on_send_status.call(val);
            text.set(String::new());
        }
    };
    use_effect(move || {
        menu_close_token.read();
        send_menu.set(None);
    });
    use_effect(move || {
        clear_text_token.read();
        text.set(String::new());
    });

    rsx! {
        div {
            class: "w-full h-12 flex items-center gap-3",
            onclick: move |_| send_menu.set(None),

            if let Some((x, y)) = send_menu() {
                div {
                    class: "fixed z-[100] bg-[#2b2b2b] border border-gray-600 rounded shadow-xl py-1 w-36",
                    style: "left: {x}px; top: {y}px;",
                    onclick: |e| e.stop_propagation(),
                    div {
                        class: "px-4 py-2 hover:bg-[#3a3a3a] cursor-pointer text-white text-sm transition-colors",
                        onclick: move |_| {
                            handle_submit_other();
                            send_menu.set(None);
                        },
                        if is_group {
                            "为……发送……"
                        } else {
                            "为对方发送…"
                        }
                    }
                    div {
                        class: "px-4 py-2 hover:bg-[#3a3a3a] cursor-pointer text-white text-sm transition-colors",
                        onclick: move |_| {
                            handle_submit_status();
                            send_menu.set(None);
                        },
                        "发送为状态"
                    }
                }
            }

            // 左侧输入条
            div {
                class: "flex-1 h-10 rounded-full flex items-center px-4 shadow-sm",
                style: "background-color: rgb(240, 238, 238);",
                input {
                    class: "flex-1 bg-transparent border-none outline-none text-black font-medium text-sm placeholder-gray-500",
                    placeholder: "发消息",
                    value: "{text}",
                    oninput: move |evt| text.set(evt.value()),
                    onkeydown: move |evt| {
                        if evt.key() == Key::Enter {
                            handle_submit();
                        }
                    },
                }
                // 输入框内的右侧图标 (气泡/发送)
                div {
                    class: "w-6 h-6 flex items-center justify-center cursor-pointer opacity-70 hover:opacity-100",
                    onclick: move |_| handle_submit(),
                    oncontextmenu: move |evt| {
                        evt.prevent_default();
                        send_menu
                            .set(
                                Some((
                                    evt.client_coordinates().x as i32,
                                    evt.client_coordinates().y as i32,
                                )),
                            );
                    },
                    img {
                        src: "{CHAT_ENTER}",
                        class: "w-full h-full object-contain",
                    }
                }
            }

            // 右侧圆形功能按钮
            div { class: "flex gap-2 items-center",
                // 表情按钮
                button {
                    class: "w-10 h-10 rounded-full flex items-center justify-center shadow-sm hover:brightness-95 transition-all cursor-pointer",
                    style: "background-color: rgb(240, 238, 238);",
                    img {
                        src: "{CHAT_EMOJI}",
                        class: "w-6 h-6 object-contain opacity-80",
                    }
                }
                // 加号按钮
                button {
                    class: "w-10 h-10 rounded-full flex items-center justify-center shadow-sm hover:brightness-95 transition-all cursor-pointer",
                    style: "background-color: rgb(240, 238, 238);",
                    onclick: move |_| handle_submit(),
                    img {
                        src: "{CHAT_PLUS}",
                        class: "w-6 h-6 object-contain opacity-80",
                    }
                }
            }
        }
    }
}
