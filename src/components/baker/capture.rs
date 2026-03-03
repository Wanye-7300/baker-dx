use dioxus::prelude::*;

use crate::components::baker::{capture, chat_area::ChatArea, Route};

#[component]
pub(super) fn CapturePage(contact_id: String) -> Element {
    let app_state = use_context::<Signal<crate::components::baker::models::AppState>>();
    let contacts = &app_state.read().contacts;
    let mut img_src = use_signal(|| String::new());

    let navigator = navigator();

    let contact = use_signal(|| {
        contacts
            .iter()
            .find(|x| x.id == contact_id)
            .cloned()
            .unwrap()
    });

    let operators = use_signal(move || app_state.read().operators.clone());
    let messages = use_memo(move || {
        app_state
            .read()
            .messages
            .get(&contact_id)
            .cloned()
            .unwrap_or_default()
    });
    let user_profile = app_state.read().user_profile.clone();
    let menu_close_token = use_signal(|| 0usize);
    let need_to_scroll_down = use_signal(|| false);
    let stickers = use_memo(move || app_state.read().stickers.clone());

    let mut width = use_signal(|| 800i64);
    let mut height = use_signal(|| 1200i64);

    use_effect(move || {
        width.read();
        height.read();

        info!("effect");

        spawn(async move {
            match capture("#chat_area").await {
                Some(src) => img_src.set(src),
                None => error!("capture chat area failed"),
            }
        });
    });

    let chat_area = rsx! {
        ChatArea {
            contact: contact(),
            operators,
            messages,
            user_profile,
            menu_close_token,
            first_prev_sender_id: None,
            force_first_avatar: false,
            pending_typing: None,
            need_to_scroll_down,
            on_send_message: move |_| {},
            on_send_other_message: move |_| {},
            on_send_status: move |_| {},
            on_send_image: move |_| {},
            on_send_sticker: move |_| {},
            on_send_sticker_other: move |_| {},
            stickers,
            on_add_sticker: move |_| {},
            on_delete_message: move |_| {},
            on_edit_message: move |_| {},
            on_add_reaction: move |_| {},
            on_delete_reaction: move |_| {},
            on_insert_message: move |_| {},
            on_start_replay: move |_| {},
            on_update_chat_head_style: move |_| {},
            on_clear_messages: move |_| {},
            on_clear_chat: move |_| {},
            on_set_group_ops_list: move |_| {},
        }
    };

    rsx! {
        div {
            style: "transform: translateX(-325000px) translateY(-325000px); overflow-y: hidden",
            position: "absolute",
            width: "{width}px",
            height: "{height}px",
            {chat_area}
        }
        div {
            width: "100%",
            height: "100vh",
            display: "flex",
            flex_direction: "column",
            position: "relative",
            overflow: "hidden",
            div { class: "h-14 shrink-0 flex items-center gap-3 px-6 border-b border-gray-600 bg-[#1f1f1f]/80 backdrop-blur-sm",
                button {
                    class: "text-gray-300 hover:text-white text-lg px-2 py-1 rounded-lg hover:bg-white/5 transition-colors",
                    onclick: move |_| {
                        navigator.push(Route::BakerLayout {});
                    },
                    "←"
                }
                h1 { class: "text-white text-lg font-bold", "导出截图" }
            }
            div {
                width: "100%",
                flex: "1",
                display: "flex",
                gap: "16px",
                padding: "16px",
                box_sizing: "border-box",
                overflow_x: "hidden",
                overflow_y: "auto",
                div {
                    flex: "1 1 auto",
                    min_width: "200px",
                    min_height: "200px",
                    div { class: "space-y-1",
                        label { class: "block text-white text-sm", "宽度" }
                        input {
                            class: "w-full bg-[#e9e9e9] border border-black/10 rounded p-3 text-black text-sm focus:outline-none focus:border-black/30 resize-none",
                            r#type: "number",
                            min: "200",
                            step: "100",
                            value: "{width}",
                            oninput: move |e| width.set(e.value().parse().unwrap_or(0)),
                        }
                    }
                    div { class: "space-y-1",
                        label { class: "block text-white text-sm", "高度" }
                        input {
                            class: "w-full bg-[#e9e9e9] border border-black/10 rounded p-3 text-black text-sm focus:outline-none focus:border-black/30 resize-none",
                            r#type: "number",
                            min: "500",
                            step: "100",
                            value: "{height}",
                            oninput: move |e| height.set(e.value().parse().unwrap_or(0)),
                        }
                    }
                }
                div {
                    flex: "0 1 40%",
                    width: "40%",
                    min_width: "500px",
                    min_height: 0,
                    img { class: "w-full", src: img_src }
                }
            }
        }
    }
}
