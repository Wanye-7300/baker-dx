use crate::components::baker::{avif_data_url_from_bytes, data_url_from_bytes, mime_from_filename};
use crate::dioxus_elements::FileData;
use dioxus::prelude::*;

fn menu_style(x: i32, y: i32, width: i32, height: i32) -> String {
    format!(
        "left: clamp(8px, {x}px, calc(100vw - {width}px - 8px)); top: clamp(8px, {y}px, calc(100vh - {height}px - 8px));"
    )
}

const CHAT_ENTER: Asset = asset!("/assets/images/chat_enter.png");
const CHAT_EMOJI: Asset = asset!("/assets/images/chat_emoji.png");
const CHAT_PLUS: Asset = asset!("/assets/images/chat_plus.png");
const STICKER_WRITEDOWN: Asset = asset!("/assets/sticker/writedown.png");

#[component]
pub fn InputBar(
    on_send: EventHandler<String>,
    on_send_other: EventHandler<String>,
    is_group: bool,
    on_send_status: EventHandler<String>,
    on_send_image: EventHandler<(String, bool)>,
    on_send_sticker: EventHandler<(String, bool)>,
    stickers: ReadSignal<Vec<String>>,
    on_add_sticker: EventHandler<String>,
    menu_close_token: ReadSignal<usize>,
    sticker_menu: Signal<Option<(i32, i32)>>,
    clear_text_token: ReadSignal<usize>,
    need_to_scroll_down: Signal<bool>,
) -> Element {
    let mut text = use_signal(String::new);
    let mut send_menu = use_signal(|| Option::<(i32, i32)>::None);
    let mut plus_menu = use_signal(|| Option::<(i32, i32)>::None);
    let mut image_input_token = use_signal(|| 0usize);
    let mut image_send_other = use_signal(|| false);
    let mut sticker_input_token = use_signal(|| 0usize);

    let mut handle_submit = move || {
        let val = text.read().clone();
        if !val.trim().is_empty() {
            on_send.call(val);
            need_to_scroll_down.set(true);
            text.set(String::new());
        }
    };
    let mut handle_submit_other = move || {
        let val = text.read().clone();
        if !val.trim().is_empty() {
            on_send_other.call(val);
            need_to_scroll_down.set(true);
            text.set(String::new());
        }
    };
    let mut handle_submit_status = move || {
        let val = text.read().clone();
        if !val.trim().is_empty() {
            on_send_status.call(val);
            need_to_scroll_down.set(true);
            text.set(String::new());
        }
    };
    use_effect(move || {
        menu_close_token.read();
        send_menu.set(None);
        plus_menu.set(None);
        sticker_menu.set(None);
    });
    use_effect(move || {
        clear_text_token.read();
        text.set(String::new());
    });

    rsx! {
        div {
            class: "w-full h-12 flex items-center gap-3",
            onclick: move |_| {
                send_menu.set(None);
                plus_menu.set(None);
                sticker_menu.set(None);
            },

            if let Some((x, y)) = send_menu() {
                div {
                    class: "fixed z-[100] bg-[#2b2b2b] border border-gray-600 rounded shadow-xl py-1 w-36",
                    style: "{menu_style(x, y, 144, 96)}",
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

            if let Some((x, y)) = plus_menu() {
                div {
                    class: "fixed z-[100] bg-[#2b2b2b] border border-gray-600 rounded shadow-xl py-1 w-36",
                    style: "{menu_style(x, y, 144, 56)}",
                    onclick: |e| e.stop_propagation(),
                    div { class: "px-4 py-2 hover:bg-[#3a3a3a] cursor-pointer text-white text-sm transition-colors relative overflow-hidden",
                        "发送图片……"
                        input {
                            key: "{image_input_token()}",
                            r#type: "file",
                            accept: "image/*",
                            class: "absolute inset-0 opacity-0 cursor-pointer",
                            onclick: move |evt| {
                                image_send_other.set(evt.modifiers().ctrl());
                            },
                            onchange: move |evt| {
                                let files: Vec<FileData> = evt.files();
                                if let Some(file) = files.first().cloned() {
                                    let send_other = image_send_other();
                                    let file_name: String = file.name();
                                    let mime = file
                                        .content_type()
                                        .unwrap_or_else(|| mime_from_filename(&file_name).to_string());
                                    let mut token = image_input_token;
                                    let mut image_send_other = image_send_other;
                                    let send_image = on_send_image;
                                    spawn(async move {
                                        if let Ok(bytes) = file.read_bytes().await {
                                            let bytes_vec = bytes.to_vec();
                                            let data_url = avif_data_url_from_bytes(bytes_vec.clone())
                                                .unwrap_or_else(|| data_url_from_bytes(&mime, bytes_vec));
                                            send_image.call((data_url, send_other));
                                            token.set(token() + 1);
                                            image_send_other.set(false);
                                        }
                                    });
                                } else {
                                    image_input_token.set(image_input_token() + 1);
                                    image_send_other.set(false);
                                }
                                plus_menu.set(None);
                            },
                        }
                    }
                }
            }

            {
                let stickers_list = stickers.read().clone();
                if let Some((x, y)) = sticker_menu() {
                    rsx! {
                        div {
                            class: "fixed z-[100] rounded-xl shadow-xl p-3",
                            style: "{menu_style(x, y, 520, 180)}; background-color: rgb(220, 220, 220);",
                            onclick: |e| e.stop_propagation(),
                            div { class: "grid grid-cols-8 gap-2",
                                label { class: "w-14 h-14 rounded-lg bg-white/60 hover:bg-white/80 transition-colors flex items-center justify-center cursor-pointer text-xs text-gray-700",
                                    input {
                                        key: "{sticker_input_token}",
                                        class: "hidden",
                                        r#type: "file",
                                        accept: "image/*",
                                        onchange: move |evt| {
                                            let files: Vec<FileData> = evt.files();
                                            if let Some(file) = files.first().cloned() {
                                                let file_name: String = file.name();
                                                let mime = file
                                                    .content_type()
                                                    .unwrap_or_else(|| mime_from_filename(&file_name).to_string());
                                                let add_sticker = on_add_sticker;
                                                let mut token = sticker_input_token;
                                                spawn(async move {
                                                    if let Ok(bytes) = file.read_bytes().await {
                                                        let bytes_vec = bytes.to_vec();
                                                        let data_url = avif_data_url_from_bytes(bytes_vec.clone())
                                                            .unwrap_or_else(|| data_url_from_bytes(&mime, bytes_vec));
                                                        add_sticker.call(data_url);
                                                        token.set(token() + 1);
                                                    }
                                                });
                                            } else {
                                                sticker_input_token.set(sticker_input_token() + 1);
                                            }
                                        },
                                    }
                                    "上传"
                                }
                                button {
                                    class: "w-14 h-14 rounded-lg bg-white/60 hover:bg-white/80 transition-colors flex items-center justify-center",
                                    onclick: move |evt| {
                                        let is_ctrl = evt.modifiers().ctrl();
                                        on_send_sticker.call((STICKER_WRITEDOWN.to_string(), is_ctrl));
                                        sticker_menu.set(None);
                                    },
                                    img { src: "{STICKER_WRITEDOWN}", class: "w-12 h-12 object-contain" }
                                }
                                for sticker_src in stickers_list {
                                    {
                                        let sticker_value = sticker_src.clone();
                                        rsx! {
                                            button {
                                                class: "w-14 h-14 rounded-lg bg-white/60 hover:bg-white/80 transition-colors flex items-center justify-center",
                                                onclick: move |evt| {
                                                    let is_ctrl = evt.modifiers().ctrl();
                                                    on_send_sticker.call((sticker_value.clone(), is_ctrl));
                                                    sticker_menu.set(None);
                                                },
                                                img { src: "{sticker_src}", class: "w-12 h-12 object-contain" }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                } else {
                    rsx! {}
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
                    onclick: move |evt| {
                        evt.stop_propagation();
                        let x = evt.client_coordinates().x as i32 - 520;
                        let y = evt.client_coordinates().y as i32 - 200;
                        sticker_menu.set(Some((x, y)));
                        send_menu.set(None);
                        plus_menu.set(None);
                    },
                    img {
                        src: "{CHAT_EMOJI}",
                        class: "w-6 h-6 object-contain opacity-80",
                    }
                }
                // 加号按钮
                button {
                    class: "w-10 h-10 rounded-full flex items-center justify-center shadow-sm hover:brightness-95 transition-all cursor-pointer",
                    style: "background-color: rgb(240, 238, 238);",
                    onclick: move |evt| {
                        evt.stop_propagation();
                        plus_menu
                            .set(
                                Some((
                                    evt.client_coordinates().x as i32,
                                    evt.client_coordinates().y as i32,
                                )),
                            );
                    },
                    img {
                        src: "{CHAT_PLUS}",
                        class: "w-6 h-6 object-contain opacity-80",
                    }
                }
            }
        }
    }
}
