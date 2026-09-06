use dioxus::prelude::*;
use uuid::Uuid;

#[rustfmt::skip]
#[derive(Clone, PartialEq)]
pub(crate) enum SettingItemType {
    Int { min: i64, max: i64, step: i64, default: i64 },
    Float { min: f64, max: f64, step: f64, default: f64 },
    Str { default: String },
    Bool { default: bool },
    Selection { selections: Vec<String>, default: String },
    Image,
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
    fn default_value(&self) -> SettingItemValue {
        match self {
            SettingItemType::Int { default, .. } => SettingItemValue::Int(*default),

            SettingItemType::Float { default, .. } => SettingItemValue::Float(*default),

            SettingItemType::Str { default } => SettingItemValue::Str(default.clone()),

            SettingItemType::Bool { default } => SettingItemValue::Bool(*default),

            SettingItemType::Selection { default, .. } => SettingItemValue::Selection(default.clone()),

            SettingItemType::Image
            | SettingItemType::Button
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
pub(crate) fn SettingPageView(vm: Signal<SettingViewModel>) -> Element {
    let mut path = use_signal(Vec::<usize>::new);

    let current_page = {
        let vm = vm.read();
        let path = path.read();

        page_at_path(&vm.page, &path)
            .cloned()
            .unwrap_or_else(|| vm.page.clone())
    };

    let title = {
        let vm = vm.read();
        let path = path.read();

        page_name_at_path(&vm.page, &path).unwrap_or_else(|| vm.name.clone())
    };

    let can_go_back = !path.read().is_empty();

    rsx! {
        div { class: "general-setting-message",

            div { class: "gsp-title-row",

                if can_go_back {
                    button {
                        class: "gsp-back-button",
                        r#type: "button",

                        onclick: move |_| {
                            path.write().pop();
                        },

                        "‹"
                    }
                }

                h3 { class: "gsp-header", "{title}" }
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
    let initial_value = item.content.default_value();
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
            default,
        } => {
            let current = {
                match &*value.read() {
                    SettingItemValue::Int(value) => *value,
                    _ => default,
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
            default,
        } => {
            let current = {
                match &*value.read() {
                    SettingItemValue::Float(value) => *value,
                    _ => default,
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
        SettingItemType::Str { default } => {
            let current = {
                match &*value.read() {
                    SettingItemValue::Str(value) => value.clone(),
                    _ => default,
                }
            };

            rsx! {
                div { class: "gsp-item",

                    SettingItemLabel { name, desc }

                    input {
                        class: "gsp-item-input",
                        r#type: "text",
                        value: "{current}",

                        oninput: move |event| {
                            let new_value = event.value();
                            value.set(SettingItemValue::Str(new_value.clone()));
                            emit_change(&on_change, SettingItemValue::Str(new_value));
                        },
                    }
                }
            }
        }

        // ====================================================
        // Bool
        // ====================================================
        SettingItemType::Bool { default } => {
            let current = {
                match &*value.read() {
                    SettingItemValue::Bool(value) => *value,
                    _ => default,
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
        SettingItemType::Selection { selections, default } => {
            let current = {
                match &*value.read() {
                    SettingItemValue::Selection(value) => value.clone(),

                    _ => default,
                }
            };

            rsx! {
                div { class: "gsp-item",

                    SettingItemLabel { name, desc }

                    select {
                        class: "gsp-item-selection",
                        value: "{current}",

                        onchange: move |event| {
                            let new_value = event.value();
                            value.set(SettingItemValue::Selection(new_value.clone()));
                            emit_change(&on_change, SettingItemValue::Selection(new_value));
                        },

                        for selection in selections {
                            option { value: "{selection}", "{selection}" }
                        }
                    }
                }
            }
        }

        // ====================================================
        // Image
        // ====================================================
        SettingItemType::Image => {
            rsx! {
                div { class: "gsp-item",

                    SettingItemLabel { name, desc }

                    input {
                        class: "gsp-item-image",
                        r#type: "file",
                        accept: "image/*",

                    // TODO:
                    //
                    // 这里之后读取文件、存入你的资源系统，
                    // 得到 Uuid 后：
                    //
                    // emit_change(
                    //     &on_change,
                    //     SettingItemValue::Image(uuid),
                    // );
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

                        "{name}"
                    }
                }
            }
        }

        // ====================================================
        // Empty
        // ====================================================
        SettingItemType::Empty => {
            rsx! {
                div { class: "gsp-item-empty" }
            }
        }

        // ====================================================
        // Header
        // ====================================================
        SettingItemType::Header => {
            rsx! {
                div { class: "gsp-item-section-header",

                    h4 { "{name}" }

                    if let Some(desc) = desc {
                        div { class: "gsp-item-desc", "{desc}" }
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

            h4 { class: "gsp-item-header", "{name}" }

            if let Some(desc) = desc {
                div { class: "gsp-item-desc", "{desc}" }
            }
        }
    }
}
