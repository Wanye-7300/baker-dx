use dioxus::prelude::*;
use dioxus::web::WebFileExt as _;
use uuid::Uuid;

use crate::ui::components::{InputComponent, InputComponentType, InputType, RichText};

/// 设置窗口在根页面时的标题；进入子页后由当前子页名替换。
pub(crate) const SETTING_WINDOW_TITLE: &str = "/ Baker // 设置";

#[allow(dead_code)]
#[rustfmt::skip]
#[derive(Clone, PartialEq)]
pub(crate) enum SettingItemType {
    Int { min: i64, max: i64, step: i64, value: i64 },
    Float { min: f64, max: f64, step: f64, value: f64 },
    Str { value: String },
    Bool { value: bool },
    Selection { selections: Vec<String>, value: String },
    Image { value: Option<Uuid> },
    Button,
    Empty,
    Header,
    Page(SettingItemPage),
}

#[derive(Clone, PartialEq)]
pub(crate) struct SettingItemPage {
    items: Vec<SettingItem>,
}

#[derive(Clone, PartialEq)]
pub(crate) struct SettingItem {
    name: String,
    desc: Option<String>,
    content: SettingItemType,
    on_change: Option<EventHandler<SettingItemValue>>,
}

#[derive(Clone, PartialEq)]
pub(crate) enum SettingItemValue {
    Int(i64),
    Float(f64),
    Str(String),
    Bool(bool),
    Selection(String),
    Image(Uuid),
    None,
}

impl SettingItemType {
    fn initial_value(&self) -> SettingItemValue {
        match self {
            SettingItemType::Int { value, .. } => SettingItemValue::Int(*value),

            SettingItemType::Float { value, .. } => SettingItemValue::Float(*value),

            SettingItemType::Str { value } => SettingItemValue::Str(value.clone()),

            SettingItemType::Bool { value } => SettingItemValue::Bool(*value),

            SettingItemType::Selection { value, .. } => SettingItemValue::Selection(value.clone()),

            SettingItemType::Image { value } => match value {
                Some(uuid) => SettingItemValue::Image(*uuid),
                None => SettingItemValue::None,
            },

            SettingItemType::Button
            | SettingItemType::Empty
            | SettingItemType::Header
            | SettingItemType::Page(_) => SettingItemValue::None,
        }
    }

    fn as_page(&self) -> Option<&SettingItemPage> {
        match self {
            SettingItemType::Page(page) => Some(page),
            _ => None,
        }
    }
}

impl SettingItemPage {
    pub(crate) fn new() -> SettingItemPage {
        SettingItemPage { items: Vec::new() }
    }

    pub(crate) fn with_child(mut self, item: SettingItem) -> SettingItemPage {
        self.items.push(item);
        self
    }
}

impl SettingItem {
    pub(crate) fn new(
        name: String,
        desc: Option<String>,
        content: SettingItemType,
        on_change: Option<EventHandler<SettingItemValue>>,
    ) -> SettingItem {
        SettingItem {
            name,
            desc,
            content,
            on_change,
        }
    }
}

#[derive(Clone, PartialEq)]
pub(crate) struct SettingViewModel {
    name: String,
    page: SettingItemPage,
    auto_save: bool,
}

impl SettingViewModel {
    pub(crate) fn new(name: String, page: SettingItemPage, auto_save: bool) -> SettingViewModel {
        SettingViewModel { name, page, auto_save }
    }
}

fn page_at_path<'a>(root: &'a SettingItemPage, path: &[usize]) -> Option<&'a SettingItemPage> {
    let mut page = root;

    for &index in path {
        let item = page.items.get(index)?;

        page = item.content.as_page()?;
    }

    Some(page)
}

fn page_name_at_path(root: &SettingItemPage, path: &[usize]) -> Option<String> {
    let mut page = root;
    let mut name = None;

    for &index in path {
        let item = page.items.get(index)?;

        name = Some(item.name.clone());
        page = item.content.as_page()?;
    }

    name
}

fn emit_change(handler: &Option<EventHandler<SettingItemValue>>, value: SettingItemValue) {
    if let Some(handler) = handler {
        handler.call(value);
    }
}

#[component]
pub(crate) fn SettingPageView(vm: Signal<SettingViewModel>, mut caption: Signal<String>) -> Element {
    let mut path = use_signal(Vec::<usize>::new);

    let current_page = {
        let vm = vm.read();
        let path = path.read();

        page_at_path(&vm.page, &path)
            .cloned()
            .unwrap_or_else(|| vm.page.clone())
    };

    // 把当前子页名同步给窗口标题：不在子页时用窗口自己的名字
    use_effect(move || {
        let caption_text = {
            let vm = vm.read();
            let path = path.read();

            page_name_at_path(&vm.page, &path).unwrap_or_else(|| vm.name.clone())
        };

        caption.set(caption_text);
    });

    let can_go_back = !path.read().is_empty();

    rsx! {
        div { class: "general-setting-message",

            if can_go_back {
                div { class: "gsp-title-row",

                    button {
                        class: "gsp-back-button",
                        r#type: "button",

                        onclick: move |_| {
                            path.write().pop();
                        },

                        "‹"
                    }
                }
            }

            div { class: "gsp-items",

                for (index , item) in current_page.items.into_iter().enumerate() {
                    SettingItemView {
                        key: "{index}",
                        item,

                        on_open_page: move |_: ()| {
                            path.write().push(index);
                        },
                    }
                }
            }
        }
    }
}

#[component]
pub(crate) fn SettingItemView(item: SettingItem, on_open_page: EventHandler<()>) -> Element {
    let initial_value = item.content.initial_value();
    let mut value = use_signal(|| initial_value);

    let SettingItem {
        name,
        desc,
        content,
        on_change,
    } = item;

    match content {
        // ====================================================
        // Int
        // ====================================================
        SettingItemType::Int {
            min,
            max,
            step,
            value: initial,
        } => {
            let current = {
                match &*value.read() {
                    SettingItemValue::Int(value) => *value,
                    _ => initial,
                }
            };

            rsx! {
                div { class: "gsp-item",

                    SettingItemLabel { name, desc }

                    input {
                        class: "gsp-item-input",
                        r#type: "number",

                        min: "{min}",
                        max: "{max}",
                        step: "{step}",
                        value: "{current}",

                        oninput: move |event| {
                            let Ok(new_value) =
                                event.value().parse::<i64>()
                            else {
                                return;
                            };

                            let new_value =
                                new_value.clamp(min, max);
                            value.set(SettingItemValue::Int(new_value));
                            emit_change(&on_change, SettingItemValue::Int(new_value));
                        },
                    }
                }
            }
        }

        // ====================================================
        // Float
        // ====================================================
        SettingItemType::Float {
            min,
            max,
            step,
            value: initial,
        } => {
            let current = {
                match &*value.read() {
                    SettingItemValue::Float(value) => *value,
                    _ => initial,
                }
            };

            rsx! {
                div { class: "gsp-item",

                    SettingItemLabel { name, desc }

                    input {
                        class: "gsp-item-input",
                        r#type: "number",

                        min: "{min}",
                        max: "{max}",
                        step: "{step}",
                        value: "{current}",

                        oninput: move |event| {
                            let Ok(new_value) =
                                event.value().parse::<f64>()
                            else {
                                return;
                            };

                            let new_value =
                                new_value.clamp(min, max);
                            value.set(SettingItemValue::Float(new_value));
                            emit_change(&on_change, SettingItemValue::Float(new_value));
                        },
                    }
                }
            }
        }

        // ====================================================
        // Str
        // ====================================================
        SettingItemType::Str { value: initial } => {
            let current = {
                match &*value.read() {
                    SettingItemValue::Str(value) => value.clone(),
                    _ => initial,
                }
            };

            rsx! {
                div { class: "gsp-item",

                    SettingItemLabel { name: name.clone(), desc }

                    InputComponent {
                        id: "setting-input-{name}",
                        label: name.clone(),
                        component_type: InputComponentType::Text,
                        value: Some(current),
                        on_value_change: move |new_value| {
                            if let InputType::Text(new_value) = new_value {
                                value.set(SettingItemValue::Str(new_value.clone()));
                                emit_change(&on_change, SettingItemValue::Str(new_value));
                            }
                        },
                    }
                }
            }
        }

        // ====================================================
        // Bool
        // ====================================================
        SettingItemType::Bool { value: initial } => {
            let current = {
                match &*value.read() {
                    SettingItemValue::Bool(value) => *value,
                    _ => initial,
                }
            };

            rsx! {
                div { class: "gsp-item",

                    SettingItemLabel { name, desc }

                    input {
                        class: "gsp-item-bool",
                        r#type: "checkbox",
                        checked: current,

                        onchange: move |event| {
                            let new_value = event.checked();
                            value.set(SettingItemValue::Bool(new_value));
                            emit_change(&on_change, SettingItemValue::Bool(new_value));
                        },
                    }
                }
            }
        }

        // ====================================================
        // Selection
        // ====================================================
        SettingItemType::Selection { selections, value: initial } => {
            let current = {
                match &*value.read() {
                    SettingItemValue::Selection(value) => value.clone(),

                    _ => initial,
                }
            };

            rsx! {
                div { class: "gsp-item",

                    SettingItemLabel { name, desc }

                    select {
                        class: "gsp-item-selection",
                        value: current.clone(),

                        onchange: move |event| {
                            let new_value = event.value();
                            value.set(SettingItemValue::Selection(new_value.clone()));
                            emit_change(&on_change, SettingItemValue::Selection(new_value));
                        },

                        for selection in selections {
                            option {
                                value: selection.clone(),
                                selected: if selection == current { true },
                                {selection.clone()}
                            }
                        }
                    }
                }
            }
        }

        // ====================================================
        // Image
        // ====================================================
        SettingItemType::Image { .. } => {
            let mut with_input_disabled = use_signal(|| false);
            let has_image = matches!(&*value.read(), SettingItemValue::Image(_));

            rsx! {
                div { class: "gsp-item",

                    SettingItemLabel { name, desc }

                    div {
                        div { class: "gsp-item-desc",
                            if has_image {
                                "已设置，选择新文件可替换"
                            } else {
                                "未设置"
                            }
                        }

                        input {
                            class: "gsp-item-image",
                            r#type: "file",
                            accept: "image/*",
                            disabled: with_input_disabled(),

                            onchange: move |evt: Event<FormData>| {
                                if let Some(file_data) = evt.files().first() {
                                    let file = file_data.get_web_file().unwrap();

                                    spawn(async move {
                                        with_input_disabled.set(true);

                                        let uuid = Uuid::new_v4();
                                        crate::shared::database::save_multimedia(uuid, file.into()).await.unwrap();

                                        // 旧的图不再被引用，顺手删掉，避免媒体库里留孤儿
                                        let previous = match &*value.read() {
                                            SettingItemValue::Image(previous) => Some(*previous),
                                            _ => None,
                                        };
                                        if let Some(previous) = previous {
                                            crate::shared::database::remove_multimedia(previous).await.unwrap();
                                        }

                                        value.set(SettingItemValue::Image(uuid));
                                        with_input_disabled.set(false);
                                        emit_change(&on_change, SettingItemValue::Image(uuid));
                                    });
                                }
                            },
                        }
                    }
                }
            }
        }

        // ====================================================
        // Button
        // ====================================================
        SettingItemType::Button => {
            rsx! {
                div { class: "gsp-item",

                    SettingItemLabel { name: name.clone(), desc }

                    button {
                        class: "gsp-item-button",
                        r#type: "button",

                        onclick: move |_| {
                            emit_change(&on_change, SettingItemValue::None);
                        },

                        {name}
                    }
                }
            }
        }

        // ====================================================
        // Empty
        // ====================================================
        SettingItemType::Empty => {
            rsx! {
                div { class: "gsp-item",

                    SettingItemLabel { name, desc }
                }
            }
        }

        // ====================================================
        // Header
        // ====================================================
        SettingItemType::Header => {
            rsx! {
                div { class: "gsp-item-section-header",

                    h4 { {name} }

                    if let Some(desc) = desc {
                        div { class: "gsp-item-desc", {desc} }
                    }
                }
            }
        }

        // ====================================================
        // Page
        // ====================================================
        SettingItemType::Page(_) => {
            rsx! {
                button {
                    class: "gsp-item gsp-page-item",
                    r#type: "button",

                    onclick: move |_| {
                        on_open_page.call(());
                    },

                    span {
                        SettingItemLabel { name, desc }
                    }

                    span { class: "gsp-page-arrow", "›" }
                }
            }
        }
    }
}

#[component]
fn SettingItemLabel(name: String, desc: Option<String>) -> Element {
    rsx! {
        span { class: "gsp-item-label",

            h4 { class: "gsp-item-header", {name} }

            if let Some(desc) = desc {
                RichText { class: "gsp-item-desc", text: desc }
            }
        }
    }
}
