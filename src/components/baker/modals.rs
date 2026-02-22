use crate::components::baker::models::{BackgroundMode, BackgroundSettings, Operator};
use crate::dioxus_elements::FileData;
use base64::Engine;
use dioxus::prelude::*;
use uuid::Uuid;

fn mime_from_filename(name: &str) -> &'static str {
    match name
        .rsplit('.')
        .next()
        .map(|ext| ext.to_lowercase())
        .as_deref()
    {
        Some("png") => "image/png",
        Some("jpg") | Some("jpeg") => "image/jpeg",
        Some("gif") => "image/gif",
        Some("webp") => "image/webp",
        _ => "application/octet-stream",
    }
}

fn data_url_from_bytes(mime: &str, bytes: Vec<u8>) -> String {
    let encoded = base64::engine::general_purpose::STANDARD.encode(bytes);
    format!("data:{mime};base64,{encoded}")
}

#[derive(Clone, PartialEq)]
pub enum ReplayIntervalMode {
    Fixed,
    PerChar,
}

#[derive(Clone, PartialEq)]
pub struct ReplaySettings {
    pub mode: ReplayIntervalMode,
    pub fixed_ms: u64,
    pub per_char_ms: u64,
    pub gap_ms: u64,
}

#[component]
pub fn ReplaySettingsModal(
    on_close: EventHandler<()>,
    on_start: EventHandler<ReplaySettings>,
) -> Element {
    let mut mode = use_signal(|| ReplayIntervalMode::Fixed);
    let mut fixed_ms = use_signal(|| "800".to_string());
    let mut per_char_ms = use_signal(|| "40".to_string());
    let mut gap_ms = use_signal(|| "200".to_string());

    let fixed_class = if matches!(mode(), ReplayIntervalMode::Fixed) {
        "bg-blue-600 text-white"
    } else {
        "bg-[#3a3a3a] text-gray-300"
    };
    let per_char_class = if matches!(mode(), ReplayIntervalMode::PerChar) {
        "bg-blue-600 text-white"
    } else {
        "bg-[#3a3a3a] text-gray-300"
    };

    rsx! {
        div {
            class: "fixed inset-0 bg-black/50 flex items-center justify-center z-50 backdrop-blur-sm",
            onclick: move |_| on_close.call(()),
            div {
                class: "bg-[#2b2b2b] w-[420px] rounded-xl shadow-2xl overflow-hidden border border-gray-600",
                onclick: |e| e.stop_propagation(),

                div { class: "px-6 py-4 border-b border-gray-600 flex justify-between items-center bg-[#333]",
                    h2 { class: "text-white text-lg font-bold", "回放设置" }
                    button {
                        class: "text-gray-400 hover:text-white transition-colors",
                        onclick: move |_| on_close.call(()),
                        "✕"
                    }
                }

                div { class: "p-4 space-y-4",
                    div { class: "flex gap-2",
                        button {
                            class: "flex-1 px-3 py-2 rounded text-sm font-medium transition-colors {fixed_class}",
                            onclick: move |_| mode.set(ReplayIntervalMode::Fixed),
                            "固定间隔"
                        }
                        button {
                            class: "flex-1 px-3 py-2 rounded text-sm font-medium transition-colors {per_char_class}",
                            onclick: move |_| mode.set(ReplayIntervalMode::PerChar),
                            "按字数"
                        }
                    }

                    div { class: "space-y-3",
                        div { class: "space-y-1",
                            label { class: "block text-gray-400 text-sm", "固定间隔 (ms)" }
                            input {
                                class: "w-full bg-[#222] border border-gray-600 rounded px-3 py-2 text-white text-sm focus:outline-none focus:border-blue-500",
                                r#type: "number",
                                min: "0",
                                value: "{fixed_ms}",
                                oninput: move |e| fixed_ms.set(e.value()),
                            }
                        }
                        div { class: "space-y-1",
                            label { class: "block text-gray-400 text-sm", "每字时间 (ms)" }
                            input {
                                class: "w-full bg-[#222] border border-gray-600 rounded px-3 py-2 text-white text-sm focus:outline-none focus:border-blue-500",
                                r#type: "number",
                                min: "0",
                                value: "{per_char_ms}",
                                oninput: move |e| per_char_ms.set(e.value()),
                            }
                        }
                        div { class: "space-y-1",
                            label { class: "block text-gray-400 text-sm", "发送后间隔 (ms)" }
                            input {
                                class: "w-full bg-[#222] border border-gray-600 rounded px-3 py-2 text-white text-sm focus:outline-none focus:border-blue-500",
                                r#type: "number",
                                min: "0",
                                value: "{gap_ms}",
                                oninput: move |e| gap_ms.set(e.value()),
                            }
                        }
                    }

                    div { class: "flex justify-end gap-3",
                        button {
                            class: "px-4 py-2 text-gray-400 hover:text-white text-sm",
                            onclick: move |_| on_close.call(()),
                            "取消"
                        }
                        button {
                            class: "px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded text-sm font-medium",
                            onclick: move |_| {
                                let fixed = fixed_ms().parse::<u64>().unwrap_or(800);
                                let per_char = per_char_ms().parse::<u64>().unwrap_or(40);
                                let gap = gap_ms().parse::<u64>().unwrap_or(200);
                                on_start
                                    .call(ReplaySettings {
                                        mode: mode(),
                                        fixed_ms: fixed,
                                        per_char_ms: per_char,
                                        gap_ms: gap,
                                    });
                                on_close.call(());
                            },
                            "开始回放"
                        }
                    }
                }
            }
        }
    }
}

#[component]
pub fn SettingsModal(
    operators: Signal<Vec<Operator>>,
    background: Signal<BackgroundSettings>,
    on_close: EventHandler<()>,
) -> Element {
    let mut new_name = use_signal(|| "".to_string());
    let mut new_avatar_preview = use_signal(|| "".to_string());

    let handle_add = move |_| {
        let name = new_name();
        let avatar = new_avatar_preview();
        if !name.is_empty() {
            let id = Uuid::new_v4().to_string();
            operators.write().push(Operator {
                id,
                name,
                avatar_url: avatar,
            });
            new_name.set("".to_string());
            new_avatar_preview.set("".to_string());
        }
    };

    let mut handle_delete = move |id: String| {
        operators.write().retain(|op| op.id != id);
    };

    let ops_list = operators.read().clone();
    let current_background = background.read().clone();
    let dot_dark_class = if matches!(current_background.mode, BackgroundMode::DotDark) {
        "bg-blue-600 text-white"
    } else {
        "bg-[#3a3a3a] text-gray-300"
    };
    let dot_light_class = if matches!(current_background.mode, BackgroundMode::DotLight) {
        "bg-blue-600 text-white"
    } else {
        "bg-[#3a3a3a] text-gray-300"
    };
    let color_class = if matches!(current_background.mode, BackgroundMode::CustomColor) {
        "bg-blue-600 text-white"
    } else {
        "bg-[#3a3a3a] text-gray-300"
    };
    let image_class = if matches!(current_background.mode, BackgroundMode::CustomImage) {
        "bg-blue-600 text-white"
    } else {
        "bg-[#3a3a3a] text-gray-300"
    };

    rsx! {
        div {
            class: "fixed inset-0 bg-black/50 flex items-center justify-center z-50 backdrop-blur-sm",
            onclick: move |_| on_close.call(()),
            div {
                class: "bg-[#2b2b2b] w-[500px] rounded-xl shadow-2xl overflow-hidden border border-gray-600",
                onclick: |e| e.stop_propagation(), // Prevent closing when clicking inside

                // Header
                div { class: "px-6 py-4 border-b border-gray-600 flex justify-between items-center bg-[#333]",
                    h2 { class: "text-white text-lg font-bold", "干员管理设置" }
                    button {
                        class: "text-gray-400 hover:text-white transition-colors",
                        onclick: move |_| on_close.call(()),
                        "✕"
                    }
                }

                div { class: "p-6 max-h-[60vh] overflow-y-auto custom-scrollbar space-y-6",
                    div { class: "p-4 bg-[#3a3a3a] rounded-lg border border-gray-600",
                        h3 { class: "text-gray-300 text-sm font-bold mb-3", "添加新干员" }
                        div { class: "space-y-3 mb-3",
                            input {
                                class: "w-full bg-[#222] border border-gray-600 rounded px-3 py-2 text-white text-sm focus:outline-none focus:border-blue-500",
                                placeholder: "干员代号 (Name)",
                                value: "{new_name}",
                                oninput: move |e| new_name.set(e.value()),
                            }
                            input {
                                class: "w-full bg-[#222] border border-gray-600 rounded px-3 py-2 text-white text-sm focus:outline-none focus:border-blue-500",
                                r#type: "file",
                                accept: "image/*",
                                onchange: move |evt| {
                                    let files: Vec<FileData> = evt.files();
                                    if let Some(file) = files.first().cloned() {
                                        let file_name: String = file.name();
                                        let mime = file
                                            .content_type()
                                            .unwrap_or_else(|| mime_from_filename(&file_name).to_string());
                                        let mut preview = new_avatar_preview;
                                        spawn(async move {
                                            if let Ok(bytes) = file.read_bytes().await {
                                                let data_url = data_url_from_bytes(&mime, bytes.to_vec());
                                                preview.set(data_url);
                                            }
                                        });
                                    }
                                },
                            }
                        }
                        button {
                            class: "w-full bg-blue-600 hover:bg-blue-500 text-white py-2 rounded text-sm font-medium transition-colors",
                            onclick: handle_add,
                            "添加干员"
                        }
                    }

                    div { class: "space-y-2",
                        for op in ops_list {
                            div { class: "flex items-center justify-between p-3 bg-[#333] rounded border border-gray-700",
                                div { class: "flex items-center gap-3",
                                    div { class: "w-10 h-10 rounded bg-gray-600 flex items-center justify-center overflow-hidden",
                                        if !op.avatar_url.is_empty() {
                                            img {
                                                src: "{op.avatar_url}",
                                                class: "w-full h-full object-cover",
                                            }
                                        } else {
                                            span { class: "text-white font-bold",
                                                "{op.name.chars().next().unwrap_or('?')}"
                                            }
                                        }
                                    }
                                    span { class: "text-white font-medium", "{op.name}" }
                                }
                                button {
                                    class: "text-red-400 hover:text-red-300 text-sm px-2 py-1",
                                    onclick: move |_| handle_delete(op.id.clone()),
                                    "删除"
                                }
                            }
                        }
                    }

                    div { class: "p-4 bg-[#3a3a3a] rounded-lg border border-gray-600 space-y-4",
                        h3 { class: "text-gray-300 text-sm font-bold", "背景设置" }
                        div { class: "grid grid-cols-2 gap-3",
                            button {
                                class: "px-3 py-2 rounded text-sm font-medium transition-colors {dot_dark_class}",
                                onclick: move |_| {
                                    let mut bg = background.write();
                                    bg.mode = BackgroundMode::DotDark;
                                },
                                "点状-深色"
                            }
                            button {
                                class: "px-3 py-2 rounded text-sm font-medium transition-colors {dot_light_class}",
                                onclick: move |_| {
                                    let mut bg = background.write();
                                    bg.mode = BackgroundMode::DotLight;
                                },
                                "点状-浅色"
                            }
                            button {
                                class: "px-3 py-2 rounded text-sm font-medium transition-colors {color_class}",
                                onclick: move |_| {
                                    let mut bg = background.write();
                                    bg.mode = BackgroundMode::CustomColor;
                                },
                                "自定义颜色"
                            }
                            button {
                                class: "px-3 py-2 rounded text-sm font-medium transition-colors {image_class}",
                                onclick: move |_| {
                                    let mut bg = background.write();
                                    bg.mode = BackgroundMode::CustomImage;
                                },
                                "自定义图片"
                            }
                        }
                        div { class: "space-y-2",
                            div { class: "flex items-center gap-3",
                                input {
                                    class: "w-24 h-10 bg-transparent border border-gray-600 rounded",
                                    r#type: "color",
                                    value: "{current_background.custom_color}",
                                    oninput: move |e| {
                                        let mut bg = background.write();
                                        bg.custom_color = e.value();
                                        bg.mode = BackgroundMode::CustomColor;
                                    },
                                }
                                input {
                                    class: "flex-1 bg-[#222] border border-gray-600 rounded px-3 py-2 text-white text-sm focus:outline-none focus:border-blue-500",
                                    r#type: "file",
                                    accept: "image/*",
                                    onchange: move |evt| {
                                        let files: Vec<FileData> = evt.files();
                                        if let Some(file) = files.first().cloned() {
                                            let file_name: String = file.name();
                                            let mime = file
                                                .content_type()
                                                .unwrap_or_else(|| mime_from_filename(&file_name).to_string());
                                            let mut bg = background;
                                            spawn(async move {
                                                if let Ok(bytes) = file.read_bytes().await {
                                                    let data_url = data_url_from_bytes(&mime, bytes.to_vec());
                                                    let mut settings = bg.write();
                                                    settings.custom_image = data_url;
                                                    settings.mode = BackgroundMode::CustomImage;
                                                }
                                            });
                                        }
                                    },
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[component]
pub fn ProfileModal(
    current_name: String,
    current_avatar: String,
    on_close: EventHandler<()>,
    on_save: EventHandler<(String, String)>,
) -> Element {
    let mut name = use_signal(|| current_name);
    let avatar_preview = use_signal(|| current_avatar);

    rsx! {
        div {
            class: "fixed inset-0 bg-black/50 flex items-center justify-center z-50 backdrop-blur-sm",
            onclick: move |_| on_close.call(()),
            div {
                class: "bg-[#2b2b2b] w-[400px] rounded-xl shadow-2xl overflow-hidden border border-gray-600",
                onclick: |e| e.stop_propagation(),

                div { class: "px-6 py-4 border-b border-gray-600 flex justify-between items-center bg-[#333]",
                    h2 { class: "text-white text-lg font-bold", "个人资料设置" }
                    button {
                        class: "text-gray-400 hover:text-white transition-colors",
                        onclick: move |_| on_close.call(()),
                        "✕"
                    }
                }

                div { class: "p-6",
                    div { class: "space-y-4",
                        div {
                            label { class: "block text-gray-400 text-sm mb-1", "昵称" }
                            input {
                                class: "w-full bg-[#222] border border-gray-600 rounded px-3 py-2 text-white text-sm focus:outline-none focus:border-blue-500",
                                value: "{name}",
                                oninput: move |e| name.set(e.value()),
                            }
                        }
                        div {
                            label { class: "block text-gray-400 text-sm mb-1", "头像文件" }
                            input {
                                class: "w-full bg-[#222] border border-gray-600 rounded px-3 py-2 text-white text-sm focus:outline-none focus:border-blue-500",
                                r#type: "file",
                                accept: "image/*",
                                onchange: move |evt| {
                                    let files: Vec<FileData> = evt.files();
                                    if let Some(file) = files.first().cloned() {
                                        let file_name: String = file.name();
                                        let mime = file
                                            .content_type()
                                            .unwrap_or_else(|| mime_from_filename(&file_name).to_string());
                                        let mut preview = avatar_preview;
                                        spawn(async move {
                                            if let Ok(bytes) = file.read_bytes().await {
                                                let data_url = data_url_from_bytes(&mime, bytes.to_vec());
                                                preview.set(data_url);
                                            }
                                        });
                                    }
                                },
                            }
                        }

                        // Preview
                        div { class: "flex justify-center mt-4",
                            div { class: "w-20 h-20 rounded bg-gray-600 flex items-center justify-center overflow-hidden border border-gray-500",
                                if !avatar_preview().is_empty() {
                                    img {
                                        src: "{avatar_preview}",
                                        class: "w-full h-full object-cover",
                                    }
                                } else {
                                    span { class: "text-white font-bold text-xl",
                                        "{name.read().chars().next().unwrap_or('?')}"
                                    }
                                }
                            }
                        }
                    }

                    div { class: "flex justify-end gap-3 mt-6",
                        button {
                            class: "px-4 py-2 text-gray-400 hover:text-white text-sm",
                            onclick: move |_| on_close.call(()),
                            "取消"
                        }
                        button {
                            class: "px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded text-sm font-medium",
                            onclick: move |_| { on_save.call((name(), avatar_preview())) },
                            "保存"
                        }
                    }
                }
            }
        }
    }
}

#[component]
pub fn EditMessageModal(
    initial_content: String,
    on_close: EventHandler<()>,
    on_save: EventHandler<String>,
) -> Element {
    let mut content = use_signal(|| initial_content);

    rsx! {
        div {
            class: "fixed inset-0 bg-black/50 flex items-center justify-center z-50 backdrop-blur-sm",
            onclick: move |_| on_close.call(()),
            div {
                class: "bg-[#2b2b2b] w-[400px] rounded-xl shadow-2xl overflow-hidden border border-gray-600",
                onclick: |e| e.stop_propagation(),

                div { class: "px-6 py-4 border-b border-gray-600 flex justify-between items-center bg-[#333]",
                    h2 { class: "text-white text-lg font-bold", "编辑消息" }
                    button {
                        class: "text-gray-400 hover:text-white transition-colors",
                        onclick: move |_| on_close.call(()),
                        "✕"
                    }
                }

                div { class: "p-4",
                    textarea {
                        class: "w-full h-32 bg-[#222] border border-gray-600 rounded p-3 text-white text-sm focus:outline-none focus:border-blue-500 resize-none",
                        value: "{content}",
                        oninput: move |e| content.set(e.value()),
                    }
                    div { class: "flex justify-end gap-3 mt-4",
                        button {
                            class: "px-4 py-2 text-gray-400 hover:text-white text-sm",
                            onclick: move |_| on_close.call(()),
                            "取消"
                        }
                        button {
                            class: "px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded text-sm font-medium",
                            onclick: move |_| on_save.call(content()),
                            "保存"
                        }
                    }
                }
            }
        }
    }
}

#[component]
pub fn UpdateAvailableModal(
    latest_version: String,
    release_url: String,
    on_update_now: EventHandler<String>,
    on_close: EventHandler<()>,
    on_skip_today: EventHandler<()>,
) -> Element {
    rsx! {
        div {
            class: "fixed inset-0 bg-black/50 flex items-center justify-center z-50 backdrop-blur-sm",
            onclick: move |_| on_close.call(()),
            div {
                class: "bg-[#2b2b2b] w-[420px] rounded-xl shadow-2xl overflow-hidden border border-gray-600",
                onclick: |e| e.stop_propagation(),
                div { class: "px-6 py-4 border-b border-gray-600 flex justify-between items-center bg-[#333]",
                    h2 { class: "text-white text-lg font-bold", "发现新版本" }
                    button {
                        class: "text-gray-400 hover:text-white transition-colors",
                        onclick: move |_| on_close.call(()),
                        "✕"
                    }
                }
                div { class: "p-4 text-sm text-gray-300",
                    div { class: "mb-4",
                        "最新版本："
                        span { class: "text-white font-semibold", "{latest_version}" }
                    }
                    div { class: "flex justify-end gap-3",
                        button {
                            class: "px-4 py-2 text-gray-400 hover:text-white text-sm",
                            onclick: move |_| on_skip_today.call(()),
                            "今日内不再提醒"
                        }
                        button {
                            class: "px-4 py-2 text-gray-400 hover:text-white text-sm",
                            onclick: move |_| on_close.call(()),
                            "现在算了"
                        }
                        button {
                            class: "px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded text-sm font-medium",
                            onclick: move |_| on_update_now.call(release_url.clone()),
                            "立即更新"
                        }
                    }
                }
            }
        }
    }
}

#[component]
pub fn PickSenderModal(
    members: Vec<Operator>,
    on_close: EventHandler<()>,
    on_send: EventHandler<String>,
) -> Element {
    rsx! {
        div {
            class: "fixed inset-0 bg-black/50 flex items-center justify-center z-50 backdrop-blur-sm",
            onclick: move |_| on_close.call(()),
            div {
                class: "bg-[#2b2b2b] w-[420px] rounded-xl shadow-2xl overflow-hidden border border-gray-600",
                onclick: |e| e.stop_propagation(),
                div { class: "px-6 py-4 border-b border-gray-600 flex justify-between items-center bg-[#333]",
                    h2 { class: "text-white text-lg font-bold", "选择发送对象" }
                    button {
                        class: "text-gray-400 hover:text-white transition-colors",
                        onclick: move |_| on_close.call(()),
                        "✕"
                    }
                }
                div { class: "p-4 max-h-[50vh] overflow-y-auto custom-scrollbar",
                    if members.is_empty() {
                        div { class: "text-center text-gray-500 py-6",
                            "暂无可选成员"
                        }
                    } else {
                        div { class: "grid grid-cols-1 gap-2",
                            for member in members {
                                {
                                    let member_id = member.id.clone();
                                    let member_name = member.name.clone();
                                    let member_avatar = member.avatar_url.clone();
                                    rsx! {
                                        button {
                                            class: "flex items-center gap-3 p-3 rounded hover:bg-[#3a3a3a] transition-colors text-left group",
                                            onclick: move |_| on_send.call(member_id.clone()),
                                            div { class: "w-10 h-10 rounded bg-gray-600 flex items-center justify-center overflow-hidden border border-gray-500 group-hover:border-blue-500",
                                                if !member_avatar.is_empty() {
                                                    img {
                                                        src: "{member_avatar}",
                                                        class: "w-full h-full object-cover",
                                                    }
                                                } else {
                                                    span { class: "text-white font-bold",
                                                        "{member_name.chars().next().unwrap_or('?')}"
                                                    }
                                                }
                                            }
                                            span { class: "text-white font-medium group-hover:text-blue-400",
                                                "{member_name}"
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
}

#[component]
pub fn InsertMessageModal(
    on_close: EventHandler<()>,
    on_save: EventHandler<(String, bool)>,
) -> Element {
    let mut content = use_signal(String::new);
    let mut is_self = use_signal(|| true);

    let self_class = if is_self() {
        "bg-blue-600 text-white"
    } else {
        "bg-[#3a3a3a] text-gray-300"
    };
    let other_class = if !is_self() {
        "bg-blue-600 text-white"
    } else {
        "bg-[#3a3a3a] text-gray-300"
    };

    rsx! {
        div {
            class: "fixed inset-0 bg-black/50 flex items-center justify-center z-50 backdrop-blur-sm",
            onclick: move |_| on_close.call(()),
            div {
                class: "bg-[#2b2b2b] w-[420px] rounded-xl shadow-2xl overflow-hidden border border-gray-600",
                onclick: |e| e.stop_propagation(),

                div { class: "px-6 py-4 border-b border-gray-600 flex justify-between items-center bg-[#333]",
                    h2 { class: "text-white text-lg font-bold", "在此前插入消息" }
                    button {
                        class: "text-gray-400 hover:text-white transition-colors",
                        onclick: move |_| on_close.call(()),
                        "✕"
                    }
                }

                div { class: "p-4 space-y-4",
                    div { class: "flex gap-2",
                        button {
                            class: "flex-1 px-3 py-2 rounded text-sm font-medium transition-colors {self_class}",
                            onclick: move |_| is_self.set(true),
                            "我方"
                        }
                        button {
                            class: "flex-1 px-3 py-2 rounded text-sm font-medium transition-colors {other_class}",
                            onclick: move |_| is_self.set(false),
                            "对方"
                        }
                    }
                    textarea {
                        class: "w-full h-32 bg-[#222] border border-gray-600 rounded p-3 text-white text-sm focus:outline-none focus:border-blue-500 resize-none",
                        value: "{content}",
                        oninput: move |e| content.set(e.value()),
                    }
                    div { class: "flex justify-end gap-3",
                        button {
                            class: "px-4 py-2 text-gray-400 hover:text-white text-sm",
                            onclick: move |_| on_close.call(()),
                            "取消"
                        }
                        button {
                            class: "px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded text-sm font-medium",
                            onclick: move |_| {
                                let val = content();
                                if !val.trim().is_empty() {
                                    on_save.call((val, is_self()));
                                }
                            },
                            "插入"
                        }
                    }
                }
            }
        }
    }
}

#[derive(Clone, PartialEq)]
pub enum NewChatSelection {
    Single(Operator),
    Group {
        name: String,
        avatar_url: String,
        member_ids: Vec<String>,
    },
}

#[component]
pub fn NewChatModal(
    operators: Signal<Vec<Operator>>,
    on_close: EventHandler<()>,
    on_select: EventHandler<NewChatSelection>,
) -> Element {
    let ops_list = operators.read().clone();
    let mut selected_ids = use_signal(Vec::<String>::new);
    let mut group_name = use_signal(|| "".to_string());
    let group_avatar = use_signal(|| "".to_string());
    let mut error_text = use_signal(|| "".to_string());
    let selected_count = selected_ids().len();

    rsx! {
        div {
            class: "fixed inset-0 bg-black/50 flex items-center justify-center z-50 backdrop-blur-sm",
            onclick: move |_| on_close.call(()),
            div {
                class: "bg-[#2b2b2b] w-[400px] rounded-xl shadow-2xl overflow-hidden border border-gray-600",
                onclick: |e| e.stop_propagation(),

                div { class: "px-6 py-4 border-b border-gray-600 flex justify-between items-center bg-[#333]",
                    h2 { class: "text-white text-lg font-bold", "发起新会话" }
                    button {
                        class: "text-gray-400 hover:text-white transition-colors",
                        onclick: move |_| on_close.call(()),
                        "✕"
                    }
                }

                div { class: "p-4 max-h-[60vh] overflow-y-auto custom-scrollbar",
                    if ops_list.is_empty() {
                        div { class: "text-center text-gray-500 py-8",
                            "暂无干员数据，请先双击标题栏进行设置"
                        }
                    } else {
                        div { class: "grid grid-cols-1 gap-2",
                            for op in ops_list.iter().cloned() {
                                {
                                    let op_id = op.id.clone();
                                    let op_name = op.name.clone();
                                    let op_avatar = op.avatar_url.clone();
                                    let op_id_for_click = op_id.clone();
                                    rsx! {
                                        div {
                                            class: "flex items-center gap-3 p-3 rounded hover:bg-[#3a3a3a] transition-colors text-left group",
                                            onclick: move |_| {
                                                error_text.set("".to_string());
                                                selected_ids.with_mut(|list| {
                                                    if let Some(pos) = list.iter().position(|id| id == &op_id_for_click) {
                                                        list.remove(pos);
                                                    } else {
                                                        list.push(op_id_for_click.clone());
                                                    }
                                                });
                                            },
                                            input {
                                                r#type: "checkbox",
                                                class: "w-4 h-4 accent-blue-600",
                                                checked: selected_ids().contains(&op_id),
                                            }
                                            div { class: "w-10 h-10 rounded bg-gray-600 flex items-center justify-center overflow-hidden border border-gray-500 group-hover:border-blue-500",
                                                if !op_avatar.is_empty() {
                                                    img {
                                                        src: "{op_avatar}",
                                                        class: "w-full h-full object-cover",
                                                    }
                                                } else {
                                                    span { class: "text-white font-bold",
                                                        "{op_name.chars().next().unwrap_or('?')}"
                                                    }
                                                }
                                            }
                                            span { class: "text-white font-medium group-hover:text-blue-400",
                                                "{op_name}"
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                if !ops_list.is_empty() {
                    div { class: "px-4 pb-4 space-y-3",
                        if selected_count == 1 {
                            div { class: "flex justify-end",
                                button {
                                    class: "px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded text-sm font-medium",
                                    onclick: move |_| {
                                        if let Some(op_id) = selected_ids().first().cloned() {
                                            if let Some(op) = ops_list.iter().find(|op| op.id == op_id).cloned() {
                                                on_select.call(NewChatSelection::Single(op));
                                            }
                                        }
                                    },
                                    "新建会话"
                                }
                            }
                        }
                        if selected_count > 1 {
                            div { class: "space-y-3",
                                input {
                                    class: "w-full bg-[#222] border border-gray-600 rounded px-3 py-2 text-white text-sm focus:outline-none focus:border-blue-500",
                                    placeholder: "群组名称",
                                    value: "{group_name}",
                                    oninput: move |e| {
                                        group_name.set(e.value());
                                        error_text.set("".to_string());
                                    },
                                }
                                input {
                                    class: "w-full bg-[#222] border border-gray-600 rounded px-3 py-2 text-white text-sm focus:outline-none focus:border-blue-500",
                                    r#type: "file",
                                    accept: "image/*",
                                    onchange: move |evt| {
                                        let files: Vec<FileData> = evt.files();
                                        if let Some(file) = files.first().cloned() {
                                            let file_name: String = file.name();
                                            let mime = file
                                                .content_type()
                                                .unwrap_or_else(|| mime_from_filename(&file_name).to_string());
                                            let mut preview = group_avatar;
                                            spawn(async move {
                                                if let Ok(bytes) = file.read_bytes().await {
                                                    let data_url = data_url_from_bytes(&mime, bytes.to_vec());
                                                    preview.set(data_url);
                                                }
                                            });
                                        }
                                    },
                                }
                                if !group_avatar().is_empty() {
                                    div { class: "flex justify-center",
                                        div { class: "w-14 h-14 rounded bg-gray-600 flex items-center justify-center overflow-hidden border border-gray-500",
                                            img { src: "{group_avatar}", class: "w-full h-full object-cover" }
                                        }
                                    }
                                }
                                if !error_text().is_empty() {
                                    div { class: "text-red-400 text-sm", "{error_text}" }
                                }
                                div { class: "flex justify-end",
                                    button {
                                        class: "px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded text-sm font-medium",
                                        onclick: move |_| {
                                            let name = group_name().trim().to_string();
                                            if name.is_empty() {
                                                error_text.set("请输入群组名称".to_string());
                                                return;
                                            }
                                            on_select.call(NewChatSelection::Group {
                                                name,
                                                avatar_url: group_avatar(),
                                                member_ids: selected_ids(),
                                            });
                                        },
                                        "新建群组会话"
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
