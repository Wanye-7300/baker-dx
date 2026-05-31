use crate::components::assets::emojis::{EMOJI_KEYS, to_emoji};
use crate::components::baker::locale::use_locale_refresh;
use crate::components::baker::{data_url_from_bytes, mime_from_filename};
use crate::dioxus_elements::FileData;
use dioxus::prelude::*;
use rust_i18n::t;

const EMOJI_COMPLETION_MAX_HEIGHT_PX: usize = 256;

#[derive(Clone, PartialEq, Debug)]
struct EmojiCompletion {
    start: usize,
    query: String,
    keys: Vec<&'static str>,
}

fn menu_style(x: i32, y: i32, width: i32, height: i32) -> String {
    format!(
        "left: clamp(8px, {x}px, calc(100vw - {width}px - 8px)); top: clamp(8px, {y}px, calc(100vh - {height}px - 8px));"
    )
}

fn emoji_completion_for(text: &str) -> Option<EmojiCompletion> {
    let start = text.rfind(':')?;
    let query = &text[start + 1..];

    if query.is_empty()
        && start > 0
        && text[..start]
            .chars()
            .next_back()
            .is_some_and(|ch| !ch.is_whitespace())
    {
        return None;
    }

    if query.contains(':')
        || query
            .chars()
            .any(|ch| !(ch.is_ascii_alphanumeric() || ch == '_' || ch == '-'))
    {
        return None;
    }

    let normalized_query = query.to_ascii_lowercase().replace('-', "_");
    let keys = EMOJI_KEYS
        .iter()
        .copied()
        .filter(|key| key.starts_with(&normalized_query))
        .collect::<Vec<_>>();

    if keys.is_empty() {
        return None;
    }

    Some(EmojiCompletion {
        start,
        query: query.to_string(),
        keys,
    })
}

fn complete_emoji_text(text: &str, completion: &EmojiCompletion, key: &str) -> String {
    let mut next = String::with_capacity(text.len() + key.len() + 2);
    next.push_str(&text[..completion.start]);
    next.push(':');
    next.push_str(key);
    next.push(':');
    next.push_str(&text[completion.start + completion.query.len() + 1..]);
    next
}

const CHAT_ENTER: Asset = asset!("/assets/images/chat_enter.png");
const CHAT_EMOJI: Asset = asset!("/assets/images/chat_emoji.png");
const CHAT_PLUS: Asset = asset!("/assets/images/chat_plus.png");
const STICKER_WRITEDOWN: Asset = asset!("/assets/sticker/writedown.png");
const STICKER_PACK_COUNT: usize = 6;
const CUSTOM_STICKER_PACK_INDEX: usize = STICKER_PACK_COUNT;

const GAME_STICKERS_01: [Asset; 16] = [
    asset!("/assets/extracted/sticker/game/sns_sticker_01_01.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_01_02.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_01_03.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_01_04.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_01_05.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_01_06.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_01_07.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_01_08.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_01_09.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_01_10.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_01_11.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_01_12.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_01_13.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_01_14.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_01_15.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_01_16.png"),
];

const GAME_STICKERS_02: [Asset; 18] = [
    asset!("/assets/extracted/sticker/game/sns_sticker_02_01.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_02_02.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_02_03.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_02_04.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_02_05.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_02_06.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_02_07.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_02_08.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_02_09.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_02_10.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_02_11.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_02_12.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_02_13.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_02_14.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_02_15.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_02_16.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_02_17.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_02_18.png"),
];

const GAME_STICKERS_03: [Asset; 20] = [
    asset!("/assets/extracted/sticker/game/sns_sticker_03_01.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_03_02.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_03_03.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_03_04.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_03_05.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_03_06.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_03_07.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_03_08.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_03_09.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_03_10.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_03_11.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_03_12.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_03_13.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_03_14.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_03_15.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_03_16.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_03_17.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_03_18.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_03_19.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_03_20.png"),
];

const GAME_STICKERS_04: [Asset; 16] = [
    asset!("/assets/extracted/sticker/game/sns_sticker_04_01.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_04_02.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_04_03.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_04_04.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_04_05.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_04_06.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_04_07.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_04_08.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_04_09.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_04_10.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_04_11.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_04_12.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_04_13.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_04_14.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_04_15.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_04_16.png"),
];

const GAME_STICKERS_05: [Asset; 16] = [
    asset!("/assets/extracted/sticker/game/sns_sticker_05_01.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_05_02.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_05_03.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_05_04.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_05_05.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_05_06.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_05_07.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_05_08.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_05_09.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_05_10.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_05_11.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_05_12.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_05_13.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_05_14.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_05_15.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_05_16.png"),
];

const GAME_STICKERS_06: [Asset; 16] = [
    asset!("/assets/extracted/sticker/game/sns_sticker_06_01.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_06_02.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_06_03.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_06_04.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_06_05.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_06_06.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_06_07.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_06_08.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_06_09.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_06_10.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_06_11.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_06_12.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_06_13.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_06_14.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_06_15.png"),
    asset!("/assets/extracted/sticker/game/sns_sticker_06_16.png"),
];

fn sticker_pack(index: usize) -> (&'static str, Asset, &'static [Asset]) {
    match index {
        0 => ("Sticker pack 1", GAME_STICKERS_01[0], &GAME_STICKERS_01),
        1 => ("Sticker pack 2", GAME_STICKERS_02[0], &GAME_STICKERS_02),
        2 => ("Sticker pack 3", GAME_STICKERS_03[0], &GAME_STICKERS_03),
        3 => ("Sticker pack 4", GAME_STICKERS_04[0], &GAME_STICKERS_04),
        4 => ("Sticker pack 5", GAME_STICKERS_05[0], &GAME_STICKERS_05),
        _ => ("Sticker pack 6", GAME_STICKERS_06[0], &GAME_STICKERS_06),
    }
}

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
    let _current_locale = use_locale_refresh();
    let send_message_placeholder = t!("input.send_message_placeholder").to_string();
    let send_for_member_label = t!("input.send_for_member").to_string();
    let send_for_other_label = t!("input.send_for_other").to_string();
    let send_status_label = t!("input.send_status").to_string();
    let send_image_label = t!("input.send_image").to_string();
    let upload_label = t!("input.upload").to_string();

    let mut text = use_signal(String::new);
    let mut send_menu = use_signal(|| Option::<(i32, i32)>::None);
    let mut plus_menu = use_signal(|| Option::<(i32, i32)>::None);
    let mut image_input_token = use_signal(|| 0usize);
    let mut image_send_other = use_signal(|| false);
    let mut sticker_input_token = use_signal(|| 0usize);
    let mut active_sticker_pack = use_signal(|| 0usize);
    let mut active_emoji_completion = use_signal(|| 0usize);
    let mut dismissed_emoji_completion = use_signal(|| Option::<usize>::None);

    let mut handle_submit = move || {
        let val = text.read().clone();
        if !val.trim().is_empty() {
            on_send.call(val);
            need_to_scroll_down.set(true);
            text.set(String::new());
            active_emoji_completion.set(0);
            dismissed_emoji_completion.set(None);
        }
    };
    let mut handle_submit_other = move || {
        let val = text.read().clone();
        if !val.trim().is_empty() {
            on_send_other.call(val);
            need_to_scroll_down.set(true);
            if !is_group {
                text.set(String::new());
                active_emoji_completion.set(0);
                dismissed_emoji_completion.set(None);
            }
        }
    };
    let mut handle_submit_status = move || {
        let val = text.read().clone();
        if !val.trim().is_empty() {
            on_send_status.call(val);
            need_to_scroll_down.set(true);
            text.set(String::new());
            active_emoji_completion.set(0);
            dismissed_emoji_completion.set(None);
        }
    };
    let mut apply_emoji_completion = move |completion: EmojiCompletion, key: &'static str| {
        let current = text.read().clone();
        text.set(complete_emoji_text(&current, &completion, key));
        active_emoji_completion.set(0);
        dismissed_emoji_completion.set(None);
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
        active_emoji_completion.set(0);
        dismissed_emoji_completion.set(None);
    });
    use_effect(move || {
        let Some(completion) = emoji_completion_for(&text()) else {
            return;
        };
        if dismissed_emoji_completion() == Some(completion.start) {
            return;
        }

        let active_index = active_emoji_completion().min(completion.keys.len() - 1);
        spawn(async move {
            let script = format!(
                r#"
                requestAnimationFrame(() => {{
                    const item = document.getElementById("emoji-completion-option-{active_index}");
                    if (!item) return;
                    item.scrollIntoView({{
                        block: "nearest",
                        inline: "nearest",
                        behavior: "instant"
                    }});
                }});
                "#
            );
            let _ = document::eval(&script).await;
        });
    });

    rsx! {
        div {
            class: "relative w-full h-12 flex items-center gap-3",
            onclick: move |_| {
                send_menu.set(None);
                plus_menu.set(None);
                sticker_menu.set(None);
            },

            if sticker_menu().is_some() {
                div {
                    class: "pointer-events-none absolute",
                    style: "inset: -16px; z-index: 50; border-radius: 10px; background-color: #000; box-shadow: 0 -18px 30px rgba(0, 0, 0, 0.42);",
                }
            }

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
                            "{send_for_member_label}"
                        } else {
                            "{send_for_other_label}"
                        }
                    }
                    div {
                        class: "px-4 py-2 hover:bg-[#3a3a3a] cursor-pointer text-white text-sm transition-colors",
                        onclick: move |_| {
                            handle_submit_status();
                            send_menu.set(None);
                        },
                        "{send_status_label}"
                    }
                }
            }

            if let Some((x, y)) = plus_menu() {
                div {
                    class: "fixed z-[100] bg-[#2b2b2b] border border-gray-600 rounded shadow-xl py-1 w-36",
                    style: "{menu_style(x, y, 144, 56)}",
                    onclick: |e| e.stop_propagation(),
                    div { class: "px-4 py-2 hover:bg-[#3a3a3a] cursor-pointer text-white text-sm transition-colors relative overflow-hidden",
                        "{send_image_label}"
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
                                            let data_url = data_url_from_bytes(&mime, bytes_vec);
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
                if sticker_menu().is_some() {
                    let active_pack_index = active_sticker_pack().min(CUSTOM_STICKER_PACK_INDEX);
                    rsx! {
                        div {
                            class: "sticker-panel-pop absolute flex overflow-hidden border border-white/70 bg-[#e4e3e3] shadow-[0_16px_44px_rgba(0,0,0,0.45)]",
                            style: "left: -16px; right: -16px; bottom: 100%; z-index: 40; height: min(390px, calc(100vh - 190px)); min-height: 230px; border-radius: 18px 18px 0 0;",
                            onclick: |e| e.stop_propagation(),
                            div { class: "sticker-panel-bg absolute inset-0" }
                            div { class: "relative flex min-w-0 flex-1 flex-col",
                                div {
                                    class: "custom-scrollbar min-h-0 flex-1 overflow-y-auto px-6 pb-5 pt-6",
                                    div {
                                        class: "grid gap-x-5 gap-y-3",
                                        style: "grid-template-columns: repeat(auto-fill, minmax(116px, 1fr));",
                                        {
                                            if active_pack_index < STICKER_PACK_COUNT {
                                                let (_, _, active_stickers) = sticker_pack(active_pack_index);
                                                rsx! {
                                                    for sticker_asset in active_stickers {
                                                        {
                                                            let sticker_value = sticker_asset.to_string();
                                                            rsx! {
                                                                button {
                                                                    key: "{sticker_value}",
                                                                    class: "rounded-md bg-transparent transition-colors hover:bg-white/70 active:scale-[0.98] flex items-center justify-center",
                                                                    style: "height: 108px;",
                                                                    onclick: move |evt| {
                                                                        let is_ctrl = evt.modifiers().ctrl();
                                                                        on_send_sticker.call((sticker_value.clone(), is_ctrl));
                                                                        sticker_menu.set(None);
                                                                    },
                                                                    img {
                                                                        src: "{sticker_asset}",
                                                                        class: "object-contain select-none pointer-events-none",
                                                                        style: "width: 104px; height: 104px;",
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            } else {
                                                rsx! {
                                                    label {
                                                        class: "rounded-md bg-white/55 transition-colors hover:bg-white/80 flex items-center justify-center cursor-pointer",
                                                        style: "height: 108px;",
                                                        title: "{upload_label}",
                                                        input {
                                                            key: "{sticker_input_token()}",
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
                                                                            let data_url = data_url_from_bytes(&mime, bytes_vec);
                                                                            add_sticker.call(data_url);
                                                                            token.set(token() + 1);
                                                                        }
                                                                    });
                                                                } else {
                                                                    sticker_input_token.set(sticker_input_token() + 1);
                                                                }
                                                            },
                                                        }
                                                        img {
                                                            src: "{CHAT_PLUS}",
                                                            class: "h-10 w-10 object-contain opacity-75",
                                                        }
                                                    }
                                                    button {
                                                        class: "rounded-md bg-transparent transition-colors hover:bg-white/70 active:scale-[0.98] flex items-center justify-center",
                                                        style: "height: 108px;",
                                                        onclick: move |evt| {
                                                            let is_ctrl = evt.modifiers().ctrl();
                                                            on_send_sticker.call((STICKER_WRITEDOWN.to_string(), is_ctrl));
                                                            sticker_menu.set(None);
                                                        },
                                                        img {
                                                            src: "{STICKER_WRITEDOWN}",
                                                            class: "object-contain select-none pointer-events-none",
                                                            style: "width: 104px; height: 104px;",
                                                        }
                                                    }
                                                    for sticker_src in stickers_list {
                                                        {
                                                            let sticker_value = sticker_src.clone();
                                                            rsx! {
                                                                button {
                                                                    key: "{sticker_value}",
                                                                    class: "rounded-md bg-transparent transition-colors hover:bg-white/70 active:scale-[0.98] flex items-center justify-center",
                                                                    style: "height: 108px;",
                                                                    onclick: move |evt| {
                                                                        let is_ctrl = evt.modifiers().ctrl();
                                                                        on_send_sticker.call((sticker_value.clone(), is_ctrl));
                                                                        sticker_menu.set(None);
                                                                    },
                                                                    img {
                                                                        src: "{sticker_src}",
                                                                        class: "object-contain select-none pointer-events-none",
                                                                        style: "width: 104px; height: 104px;",
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
                                div { class: "px-6 pb-5",
                                    div {
                                        class: "custom-scrollbar flex h-14 items-center gap-5 overflow-x-auto rounded-md bg-[#373737] px-4 shadow-[inset_0_1px_4px_rgba(0,0,0,0.55)]",
                                        for pack_index in 0..STICKER_PACK_COUNT {
                                            {
                                                let (pack_label, pack_icon, _) = sticker_pack(pack_index);
                                                let is_active = pack_index == active_pack_index;
                                                rsx! {
                                                    button {
                                                        class: if is_active {
                                                            "h-12 w-20 shrink-0 rounded-md bg-[#f0eeee] shadow-[0_2px_6px_rgba(0,0,0,0.28)] transition-colors flex items-center justify-center"
                                                        } else {
                                                            "h-12 w-16 shrink-0 rounded-md bg-transparent transition-colors hover:bg-white/10 flex items-center justify-center"
                                                        },
                                                        title: "{pack_label}",
                                                        onclick: move |evt| {
                                                            evt.stop_propagation();
                                                            active_sticker_pack.set(pack_index);
                                                        },
                                                        img {
                                                            src: "{pack_icon}",
                                                            class: "h-12 w-12 object-contain select-none pointer-events-none",
                                                        }
                                                    }
                                                    div { class: "h-8 w-[2px] shrink-0 bg-[#d9d9d9]" }
                                                }
                                            }
                                        }
                                        button {
                                            class: if active_pack_index == CUSTOM_STICKER_PACK_INDEX {
                                                "h-12 w-20 shrink-0 rounded-md bg-[#f0eeee] shadow-[0_2px_6px_rgba(0,0,0,0.28)] transition-colors flex items-center justify-center"
                                            } else {
                                                "h-12 w-16 shrink-0 rounded-md bg-transparent transition-colors hover:bg-white/10 flex items-center justify-center"
                                            },
                                            title: "{upload_label}",
                                            onclick: move |evt| {
                                                evt.stop_propagation();
                                                active_sticker_pack.set(CUSTOM_STICKER_PACK_INDEX);
                                            },
                                            img {
                                                src: "{STICKER_WRITEDOWN}",
                                                class: "h-12 w-12 object-contain select-none pointer-events-none",
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
                class: "relative flex-1 h-10 rounded-full flex items-center px-4 shadow-sm",
                style: "z-index: 60; background-color: rgb(240, 238, 238);",
                onclick: |e| e.stop_propagation(),
                {
                    let completion = emoji_completion_for(&text());
                    if let Some(completion) = completion {
                        if dismissed_emoji_completion() == Some(completion.start) {
                            rsx! {}
                        } else {
                            let active_index = active_emoji_completion().min(completion.keys.len() - 1);
                            rsx! {
                                div {
                                    class: "absolute left-4 bottom-12 z-[90] w-56 overflow-y-auto rounded-md border border-gray-300 bg-white shadow-xl",
                                    style: "max-height: {EMOJI_COMPLETION_MAX_HEIGHT_PX}px;",
                                    onclick: |e| e.stop_propagation(),
                                    for (index, key) in completion.keys.iter().enumerate() {
                                        {
                                            let key = *key;
                                            let emoji_asset = to_emoji(key);
                                            let item_completion = completion.clone();
                                            let is_active = index == active_index;
                                            rsx! {
                                                button {
                                                    key: "{key}",
                                                    id: "emoji-completion-option-{index}",
                                                    class: if is_active {
                                                        "flex w-full items-center gap-2 px-3 py-2 text-left text-sm text-black bg-[#e8edf5]"
                                                    } else {
                                                        "flex w-full items-center gap-2 px-3 py-2 text-left text-sm text-black hover:bg-[#f1f3f6]"
                                                    },
                                                    onclick: move |_| apply_emoji_completion(item_completion.clone(), key),
                                                    if let Some(asset) = emoji_asset {
                                                        img {
                                                            src: "{asset}",
                                                            class: "h-6 w-6 shrink-0 object-contain",
                                                            alt: ":{key}:",
                                                        }
                                                    }
                                                    span { class: "min-w-0 flex-1 truncate font-mono", ":{key}:" }
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
                input {
                    class: "flex-1 bg-transparent border-none outline-none text-black font-medium text-sm placeholder-gray-500",
                    placeholder: "{send_message_placeholder}",
                    value: "{text}",
                    oninput: move |evt| {
                        text.set(evt.value());
                        active_emoji_completion.set(0);
                        dismissed_emoji_completion.set(None);
                    },
                    onkeydown: move |evt| {
                        if let Some(completion) = emoji_completion_for(&text())
                            && dismissed_emoji_completion() != Some(completion.start)
                        {
                            let key_count = completion.keys.len();
                            match evt.key() {
                                Key::ArrowDown => {
                                    evt.prevent_default();
                                    active_emoji_completion
                                        .set((active_emoji_completion() + 1) % key_count);
                                    return;
                                }
                                Key::ArrowUp => {
                                    evt.prevent_default();
                                    active_emoji_completion
                                        .set((active_emoji_completion() + key_count - 1) % key_count);
                                    return;
                                }
                                Key::Enter | Key::Tab => {
                                    evt.prevent_default();
                                    let active_index = active_emoji_completion().min(key_count - 1);
                                    if let Some(key) = completion.keys.get(active_index).copied() {
                                        apply_emoji_completion(completion, key);
                                    }
                                    return;
                                }
                                Key::Escape => {
                                    evt.prevent_default();
                                    active_emoji_completion.set(0);
                                    dismissed_emoji_completion.set(Some(completion.start));
                                    return;
                                }
                                _ => {}
                            }
                        }
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
            div { class: "relative flex gap-2 items-center", style: "z-index: 60;",
                // 表情按钮
                button {
                    class: "w-10 h-10 rounded-full flex items-center justify-center shadow-sm hover:brightness-95 transition-all cursor-pointer",
                    style: "background-color: rgb(240, 238, 238);",
                    onclick: move |evt| {
                        evt.stop_propagation();
                        if sticker_menu().is_some() {
                            sticker_menu.set(None);
                        } else {
                            sticker_menu.set(Some((0, 0)));
                        }
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
                        sticker_menu.set(None);
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

#[cfg(test)]
mod tests {
    use super::{complete_emoji_text, emoji_completion_for};

    #[test]
    fn emoji_completion_opens_after_colon() {
        let completion = emoji_completion_for("hello :").unwrap();
        assert_eq!(completion.start, 6);
        assert!(completion.keys.contains(&"happy"));
    }

    #[test]
    fn emoji_completion_filters_by_prefix() {
        let completion = emoji_completion_for(":hun").unwrap();
        assert_eq!(completion.keys, vec!["hundred"]);
    }

    #[test]
    fn emoji_completion_replaces_current_token() {
        let completion = emoji_completion_for("hello :hun").unwrap();
        assert_eq!(
            complete_emoji_text("hello :hun", &completion, "hundred"),
            "hello :hundred:"
        );
    }

    #[test]
    fn emoji_completion_ignores_closed_tokens() {
        assert!(emoji_completion_for("hello :happy:").is_none());
    }
}
