use crate::components::baker::models::Operator;
use crate::components::baker::{avif_data_url_from_bytes, data_url_from_bytes, mime_from_filename};
use crate::dioxus_elements::FileData;
use dioxus::prelude::*;

#[component]
fn Modal(
    title: &'static str,
    content_confirmation_button: &'static str,
    children: Element,
    on_close: EventHandler,
    on_confirm: EventHandler,
    max_width: Option<u32>,
) -> Element {
    let content_style = if let Some(mw) = max_width {
        format!("w-full max-w-[{}px] mx-auto", mw)
    } else {
        "w-full max-w-[340px] mx-auto".to_owned()
    };

    rsx! {
        div {
            class: "fixed inset-0 bg-black/50 flex items-center justify-center z-50 backdrop-blur-sm",
            onmousedown: move |_| on_close.call(()),

            div { class: "modal-mask w-screen",

                div { class: "modal-reveal",

                    div {
                        class: "modal-panel bg-[#f0f0f0] shadow-2xl overflow-hidden border border-gray-600",
                        style: "background-image: linear-gradient(rgba(0,0,0,0.06) 1px, transparent 1px), linear-gradient(90deg, rgba(0,0,0,0.06) 1px, transparent 1px); background-size: 6px 6px",
                        onclick: |e| e.stop_propagation(),
                        onmousedown: |e| e.stop_propagation(),

                        div { class: "px-5 py-3 flex justify-between items-center bg-[#fdfc00] border-b border-black/10",
                            h2 { class: "text-black text-xl font-semibold tracking-wide",
                                {title}
                            }
                            button {
                                class: "w-7 h-7 rounded flex items-center justify-center text-black hover:bg-black/10 transition-colors",
                                onclick: move |_| on_close.call(()),
                                "✕"
                            }
                        }

                        div { class: content_style,
                            div { class: "p-4 space-y-4",

                                {children}

                                div { class: "flex justify-end gap-3",
                                    button {
                                        class: "px-4 py-2 text-black hover:text-gray-400 text-sm",
                                        onclick: move |_| on_close.call(()),
                                        "取消"
                                    }
                                    button {
                                        class: "px-4 py-2 bg-[#fdfc00] hover:bg-[#fdfc00]/60 text-black rounded text-sm font-medium",
                                        onclick: move |_| {
                                            on_confirm.call(());
                                        },
                                        {content_confirmation_button}
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
        "bg-[#fdfc00] text-black"
    } else {
        "bg-[#fdfc00]/0 text-black"
    };
    let per_char_class = if matches!(mode(), ReplayIntervalMode::PerChar) {
        "bg-[#fdfc00] text-black"
    } else {
        "bg-[#fdfc00]/0 text-black"
    };

    rsx! {
        Modal {
            title: "回放设置",
            content_confirmation_button: "开始回放",
            on_confirm: move |_| {
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
            on_close,

            {
                rsx! {
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
                            label { class: "block text-black text-sm", "固定间隔 (ms)" }
                            input {
                                class: "w-full bg-[#e9e9e9] border border-black/10 rounded p-3 text-black text-sm focus:outline-none focus:border-black/30 resize-none",
                                r#type: "number",
                                min: "0",
                                value: "{fixed_ms}",
                                oninput: move |e| fixed_ms.set(e.value()),
                            }
                        }
                        div { class: "space-y-1",
                            label { class: "block text-black text-sm", "每字时间 (ms)" }
                            input {
                                class: "w-full bg-[#e9e9e9] border border-black/10 rounded p-3 text-black text-sm focus:outline-none focus:border-black/30 resize-none",
                                r#type: "number",
                                min: "0",
                                value: "{per_char_ms}",
                                oninput: move |e| per_char_ms.set(e.value()),
                            }
                        }
                        div { class: "space-y-1",
                            label { class: "block text-black text-sm", "发送后间隔 (ms)" }
                            input {
                                class: "w-full bg-[#e9e9e9] border border-black/10 rounded p-3 text-black text-sm focus:outline-none focus:border-black/30 resize-none",
                                r#type: "number",
                                min: "0",
                                value: "{gap_ms}",
                                oninput: move |e| gap_ms.set(e.value()),
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
                                                let bytes_vec = bytes.to_vec();
                                                let data_url = avif_data_url_from_bytes(bytes_vec.clone())
                                                    .unwrap_or_else(|| data_url_from_bytes(&mime, bytes_vec));
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
        Modal {
            title: "编辑消息",
            content_confirmation_button: "保存",
            on_close,
            on_confirm: move |_| {
                on_save.call(content());
            },

            {
                rsx! {
                    textarea {
                        class: "w-full h-32 bg-[#e9e9e9] border border-black/10 rounded p-3 text-black text-sm focus:outline-none focus:border-black/30 resize-none",
                        value: "{content}",
                        oninput: move |e| content.set(e.value()),
                    }
                }
            }
        }
    }
}

#[component]
pub fn ReactionModal(on_close: EventHandler<()>, on_save: EventHandler<String>) -> Element {
    let mut reaction = use_signal(|| "".to_string());

    rsx! {
        Modal {
            title: "添加反应",
            content_confirmation_button: "添加",
            on_close,
            on_confirm: move |_| {
                let val = reaction();
                if !val.trim().is_empty() {
                    on_save.call(val);
                }
            },

            {
                rsx! {
                    input {
                        class: "w-full bg-[#e9e9e9] border border-black/10 rounded p-3 text-black text-sm focus:outline-none focus:border-black/30 resize-none",
                        placeholder: "输入 reaction（例如 😀）",
                        value: "{reaction}",
                        oninput: move |e| reaction.set(e.value()),
                        onkeydown: move |e| {
                            if e.key() == Key::Enter {
                                let val = reaction();
                                if !val.trim().is_empty() {
                                    on_save.call(val);
                                }
                            }
                        },
                    }
                    // 常用表情快捷按钮
                    div { class: "flex flex-wrap gap-2",
                        for emoji in ["😀", "😂", "😭", "👍", "❤️", "❗", "❓"] {
                            {
                                let emoji_str = emoji.to_string();
                                rsx! {
                                    button {
                                        class: "px-2 py-1 rounded bg-black/10 hover:bg-black/20 text-lg text-black",
                                        onclick: {
                                            let emoji_val = emoji_str.clone();
                                            move |_| {
                                                on_save.call(emoji_val.clone());
                                            }
                                        },
                                        "{emoji}"
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
pub fn UpdateAvailableModal(
    latest_version: String,
    release_url: String,
    on_update_now: EventHandler<String>,
    on_close: EventHandler<()>,
    on_skip_today: EventHandler<()>,
) -> Element {
    let release_url = use_hook(|| release_url);
    rsx! {
        Modal {
            title: "发现新版本",
            content_confirmation_button: "现在更新",
            on_close,
            on_confirm: move |_| {
                on_update_now.call(release_url.clone());
                on_close.call(());
            },

            {
                rsx! {
                    div { class: "mb-4 text-black",
                        "最新版本："
                        span { class: "font-semibold", "{latest_version}" }
                    }
                    a {
                        class: "text-blue-400 hover:underline hover:cursor-pointer",
                        onclick: move |_| {
                            on_skip_today.call(());
                            on_close.call(());
                        },
                        "今日内不再提醒"
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
    let mut selected_id = use_signal(|| Option::<String>::None);
    rsx! {
        Modal {
            title: "选择发送对象",
            content_confirmation_button: "确定",
            on_close,
            on_confirm: move |_| {
                if let Some(id) = selected_id() {
                    on_send.call(id);
                }
                on_close.call(());
            },
            div { class: "max-h-[50vh] overflow-y-auto custom-scrollbar",
                if members.is_empty() {
                    div { class: "text-center text-black/60 py-6", "暂无可选成员" }
                } else {
                    div { class: "grid grid-cols-1 gap-2",
                        for member in members {
                            {
                                let member_id = member.id.clone();
                                let member_name = member.name.clone();
                                let member_avatar = member.avatar_url.clone();
                                let is_selected = selected_id() == Some(member_id.clone());
                                rsx! {
                                    button {
                                        class: if is_selected { "flex items-center gap-3 p-3 rounded bg-black/10 transition-colors text-left group" } else { "flex items-center gap-3 p-3 rounded hover:bg-black/5 transition-colors text-left group" },
                                        onclick: move |_| selected_id.set(Some(member_id.clone())),
                                        div { class: if is_selected { "w-10 h-10 rounded bg-gray-300 flex items-center justify-center overflow-hidden border border-black/40" } else { "w-10 h-10 rounded bg-gray-300 flex items-center justify-center overflow-hidden border border-black/10 group-hover:border-black/30" },
                                            if !member_avatar.is_empty() {
                                                img { src: "{member_avatar}", class: "w-full h-full object-cover" }
                                            } else {
                                                span { class: "text-black font-semibold", "{member_name.chars().next().unwrap_or('?')}" }
                                            }
                                        }
                                        span { class: if is_selected { "text-black font-semibold" } else { "text-black font-medium group-hover:text-black/70" },
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

#[component]
pub fn InsertMessageModal(
    members: Vec<Operator>,
    on_close: EventHandler<()>,
    on_save: EventHandler<(String, Option<String>)>,
) -> Element {
    let mut content = use_signal(String::new);
    let mut is_self = use_signal(|| true);
    // 群组模式下，选"对方"后弹出成员选择
    let mut pick_sender = use_signal(|| false);

    let is_group = members.len() > 1;

    let self_class = if is_self() {
        "bg-[#fdfc00] text-black"
    } else {
        "bg-[#fdfc00]/0 text-black"
    };
    let other_class = if !is_self() {
        "bg-[#fdfc00] text-black"
    } else {
        "bg-[#fdfc00]/0 text-black"
    };
    let on_close_safe = {
        let on_close = on_close;
        let pick_sender = pick_sender;
        move |_| {
            if !pick_sender() {
                on_close.call(());
            }
        }
    };

    if pick_sender() {
        return rsx! {
            PickSenderModal {
                members,
                on_close: move |_| pick_sender.set(false),
                on_send: move |sender_id: String| {
                    let val = content();
                    if !val.trim().is_empty() {
                        on_save.call((val, Some(sender_id)));
                    }
                    pick_sender.set(false);
                },
            }
        };
    }

    rsx! {
        Modal {
            title: "在此前插入消息",
            content_confirmation_button: "插入",
            on_close: on_close_safe,
            on_confirm: move |_| {
                let val = content();
                if val.trim().is_empty() {
                    return;
                }
                if is_self() {
                    on_save.call((val, None));
                } else if is_group {
                    pick_sender.set(true);
                } else {
                    on_save.call((val, members.first().map(|op| op.id.clone())));
                }
            },
            div { class: "space-y-4",
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
                    class: "w-full h-32 bg-[#e9e9e9] border border-black/10 rounded p-3 text-black text-sm focus:outline-none focus:border-black/30 resize-none",
                    placeholder: "输入想要插入的消息……",
                    value: "{content}",
                    oninput: move |e| content.set(e.value()),
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
    let mut selected_ids = use_signal(Vec::<String>::new);
    let mut group_name = use_signal(|| "".to_string());
    let group_avatar = use_signal(|| "".to_string());
    let mut error_text = use_signal(|| "".to_string());
    let selected_count = selected_ids().len();

    rsx! {
        Modal {
            title: "发起新会话",
            content_confirmation_button: "发起",
            on_close,
            on_confirm: move |_| {
                if selected_count == 1 {
                    if let Some(op_id) = selected_ids().first().cloned() {
                        if let Some(op) = operators
                            .read()
                            .iter()
                            .find(|op| op.id == op_id)
                            .cloned()
                        {
                            on_select.call(NewChatSelection::Single(op));
                            on_close.call(());
                        }
                    }
                } else if selected_count > 1 {
                    let name = group_name().trim().to_string();
                    if name.is_empty() {
                        error_text.set("请输入群组名称".to_string());
                        return;
                    }
                    on_select
                        .call(NewChatSelection::Group {
                            name,
                            avatar_url: group_avatar(),
                            member_ids: selected_ids(),
                        });
                    on_close.call(());
                }
            },

            {
                rsx! {
                    div { class: "p-4 max-h-[60vh] overflow-y-auto custom-scrollbar",
                        if operators.read().is_empty() {
                            div { class: "text-center text-gray-500 py-8",
                                "暂无干员数据，请先双击标题栏进行设置"
                            }
                        } else {
                            div { class: "grid grid-cols-1 gap-2",
                                for op in operators.read().iter().cloned() {
                                    {
                                        let op_id = op.id.clone();
                                        let op_name = op.name.clone();
                                        let op_avatar = op.avatar_url.clone();
                                        let op_id_for_click = op_id.clone();
                                        rsx! {
                                            div {
                                                class: "flex items-center gap-3 p-3 rounded hover:bg-black/20 transition-colors text-left group",
                                                onclick: move |_| {
                                                    error_text.set("".to_string());
                                                    selected_ids
                                                        .with_mut(|list| {
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
                                                        img { src: "{op_avatar}", class: "w-full h-full object-cover" }
                                                    } else {
                                                        span { class: "text-white font-bold", "{op_name.chars().next().unwrap_or('?')}" }
                                                    }
                                                }
                                                span { class: "text-black font-medium", "{op_name}" }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    if !operators.read().is_empty() {
                        div { class: "px-4 pb-4 space-y-3",
                            if selected_count > 1 {
                                div { class: "space-y-3",
                                    input {
                                        class: "w-full bg-[#e9e9e9] border border-black/10 rounded p-3 text-black text-sm focus:outline-none focus:border-black/30 resize-none",
                                        placeholder: "群组名称",
                                        value: "{group_name}",
                                        oninput: move |e| {
                                            group_name.set(e.value());
                                            error_text.set("".to_string());
                                        },
                                    }
                                    input {
                                        class: "w-full bg-[#e9e9e9] border border-black/10 rounded p-3 text-black text-sm focus:outline-none focus:border-black/30 resize-none",
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
                                                        let bytes_vec = bytes.to_vec();
                                                        let data_url = avif_data_url_from_bytes(bytes_vec.clone())
                                                            .unwrap_or_else(|| data_url_from_bytes(&mime, bytes_vec));
                                                        preview.set(data_url);
                                                    }
                                                });
                                            }
                                        },
                                    }
                                    if !group_avatar().is_empty() {
                                        div { class: "flex justify-center",
                                            div { class: "w-14 h-14 rounded bg-gray-600 flex items-center justify-center overflow-hidden border border-gray-500",
                                                img {
                                                    src: "{group_avatar}",
                                                    class: "w-full h-full object-cover",
                                                }
                                            }
                                        }
                                    }
                                    if !error_text().is_empty() {
                                        div { class: "text-red-400 text-sm", "{error_text}" }
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

#[derive(PartialEq, Clone)]
pub(crate) struct OpsSelection {
    pub ops: Vec<String>,
}

///
/// 用于设置特定群聊中干员名单的列表弹窗。
///
#[component]
pub fn SetGroupOpsListModal(
    on_close: EventHandler,
    on_select: EventHandler<OpsSelection>,
    selected_contact_id: Option<String>,
) -> Element {
    let app_state = use_context::<Signal<crate::components::baker::models::AppState>>();

    let operators = &app_state.read().operators;
    let app_state_read = app_state.read();
    let group_ops_list = match selected_contact_id.clone() {
        Some(id) => app_state_read.contacts.iter().find(|x| x.id == id),
        None => None,
    };

    if group_ops_list.is_none() {
        return rsx! {
            Modal {
                title: "错误",
                content_confirmation_button: "确定",
                on_close,
                on_confirm: move |_| on_close.call(()),

                {
                    rsx! { "无法找到名单。是否存在这个群聊？" }
                }
            }
        };
    }

    let mut group_ops_list = use_signal(|| group_ops_list.unwrap().participant_ids.clone());

    rsx! {
        Modal {
            title: "设置群组干员列表",
            content_confirmation_button: "确定",
            on_close,
            on_confirm: move |_| {
                on_select
                    .call(OpsSelection {
                        ops: group_ops_list(),
                    })
            },

            {
                rsx! {
                    div { class: "p-4 max-h-[60vh] overflow-y-auto custom-scrollbar",
                        div { class: "grid grid-cols-1 gap-2",
                            for op in operators {
                                {
                                    let op_id = op.id.clone();
                                    let op_name = op.name.clone();
                                    let op_avatar = op.avatar_url.clone();
                                    let op_id_for_click = op_id.clone();
                                    rsx! {
                                        div {
                                            class: "flex items-center gap-3 p-3 rounded hover:bg-black/20 transition-colors text-left group",
                                            onclick: move |_| {
                                                group_ops_list
                                                    .with_mut(|list| {
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
                                                checked: group_ops_list().contains(&op_id),
                                            }
                                            div { class: "w-10 h-10 rounded bg-gray-600 flex items-center justify-center overflow-hidden border border-gray-500 group-hover:border-blue-500",
                                                if !op_avatar.is_empty() {
                                                    img { src: "{op_avatar}", class: "w-full h-full object-cover" }
                                                } else {
                                                    span { class: "text-white font-bold", "{op_name.chars().next().unwrap_or('?')}" }
                                                }
                                            }
                                            span { class: "text-black font-medium", "{op_name}" }
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

const IMAGE_TUTORIAL_1: Asset = asset!("/tutorial/1.png");
const IMAGE_TUTORIAL_2: Asset = asset!("/tutorial/2.png");
const IMAGE_TUTORIAL_3: Asset = asset!("/tutorial/3.png");
const IMAGE_TUTORIAL_4: Asset = asset!("/tutorial/4.png");
const IMAGE_TUTORIAL_5: Asset = asset!("/tutorial/5.png");

#[component]
pub fn TutorialModal(on_close: EventHandler<()>, on_confirm: EventHandler<bool>) -> Element {
    let mut dont_show_again = use_signal(|| false);

    rsx! {
        Modal {
            title: "教程",
            content_confirmation_button: "关闭",
            on_close,
            on_confirm: move |_| { on_confirm.call(dont_show_again()) },
            max_width: 1280,

            {
                rsx! {
                    div { class: "p-6 max-h-[60vh] overflow-y-auto custom-scrollbar text-black text-base leading-relaxed space-y-3",
                        h1 { class: "text-3xl font-bold", "对于 baker-dx 的简略教程" }
                        h2 { class: "text-2xl font-bold", "1. 添加干员" }
                        p {
                            img {
                                class: "max-w-[600px]",
                                alt: "添加干员",
                                src: IMAGE_TUTORIAL_1,
                            }
                        }
                        p {
                            img {
                                class: "max-w-[600px]",
                                alt: "添加干员",
                                src: IMAGE_TUTORIAL_2,
                            }
                        }
                        p { "左键双击左上角的 //BAKER/会话消息，打开设置界面。" }
                        p { "第一个输入框是干员名称，第二个是干员头像。" }
                        p { "两个空填完之后点击添加干员即可，然后关闭设置界面。" }
                        h2 { class: "text-2xl font-bold mt-10", "2. 会话" }
                        p {
                            img { class: "max-w-[600px]", alt: "会话", src: IMAGE_TUTORIAL_3 }
                        }
                        p { "幸好默认就有 Perlica 的实例会话，我们可以直接用这个。" }
                        p { "点击 Perlica 的名片就可以切换到她的会话了。" }
                        ul { style: "list-style: circle inside",
                            li {
                                "1 处按钮可以更改会话头部的样式，点击后会弹出一个菜单，你可以选择 2 个不同的样式。"
                            }
                            li {
                                "右键输入框右侧的菜单按钮，可以选择："
                                ul { class: "ml-10", style: "list-style: square inside",
                                    li { "为对方发送：将输入框中的内容以对方的身份发送。" }
                                    li {
                                        "发送为状态：将输入框中的内容以状态行的形式发送。"
                                        ul { class: "ml-10", style: "list-style: inside",
                                            li {
                                                "状态行：状态行是一种特殊的消息，它会在会话中以独立的行展示，通常用于展示时间等其他重要信息。"
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        h2 { class: "text-2xl font-bold mt-10", "3. 回放" }
                        p {
                            img {
                                class: "max-w-[600px]",
                                alt: "完整的聊天",
                                src: IMAGE_TUTORIAL_4,
                            }
                            img {
                                class: "max-w-[600px]",
                                alt: "回放界面",
                                src: IMAGE_TUTORIAL_5,
                            }
                        }
                        p { "现在我们写好一段对话了。" }
                        p { "右键一个消息，即可开始回放。" }
                        p { "回放间隔计算有两种模式：" }
                        ul { style: "list-style: circle inside",
                            li { "固定间隔" }
                            li { "按字数：根据消息的字数计算间隔" }
                        }
                        p {
                            "那么两条消息发送的间隔就为：发送后间隔（第三个） + 输入间隔（就是那个输入动画的间隔）（前两个）"
                        }
                        p {
                            "推荐设置为：\r\n    固定间隔 400ms + 发送后间隔 1000ms，这样子可能大差不差。\r\n    点击开始回放就好了。"
                        }
                        p {
                            "（回放完之后发送消息（或者历史消息）看不到？切换其他的会话再回来就行了。）"
                        }
                        hr {}
                        p {
                            em { "如果你觉得这个软件有用，不妨分享一下？！" }
                        }
                    }
                    label { class: "flex items-center gap-2 text-black text-base cursor-pointer select-none",
                        input {
                            r#type: "checkbox",
                            class: "w-4 h-4 accent-blue-600",
                            checked: dont_show_again(),
                            onclick: move |_| dont_show_again.set(!dont_show_again()),
                        }
                        span { "不再显示" }
                    }
                }
            }
        }
    }
}
