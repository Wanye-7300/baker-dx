use crate::{operator::view_model::OperatorViewModel, session::view_model::session_view_model::SessionViewModel};

use super::assets;
use dioxus::prelude::*;
use strum::VariantArray;
use uuid::Uuid;

const MENU_VIEWPORT_GAP: f64 = 8.0;

fn viewport_size() -> Option<(f64, f64)> {
    let window = web_sys::window()?;

    Some((
        window.inner_width().ok()?.as_f64()?,
        window.inner_height().ok()?.as_f64()?,
    ))
}

fn fit_menu_position(x: f64, y: f64, width: f64, height: f64, viewport_width: f64, viewport_height: f64) -> (f64, f64) {
    let gap = MENU_VIEWPORT_GAP;

    let mut x = x;
    let mut y = y;

    // 默认右下展开；右边放不下就翻到左边
    if x + width > viewport_width - gap {
        x -= width;
    }

    // 下边放不下就翻到上边
    if y + height > viewport_height - gap {
        y -= height;
    }

    // 最后兜底，保证无论如何都不会跑出 viewport
    let max_x = (viewport_width - width - gap).max(gap);
    let max_y = (viewport_height - height - gap).max(gap);

    x = x.clamp(gap, max_x);
    y = y.clamp(gap, max_y);

    (x, y)
}

#[component]
pub(crate) fn RichText(
    text: String,
    #[props(extends = GlobalAttributes, extends = div)] attributes: Vec<Attribute>,
    children: Element,
) -> Element {
    let mut string_build = String::with_capacity(text.capacity());
    let mut emoji_text: Option<String> = None;
    for chr in text.chars() {
        match chr {
            ':' => match &emoji_text {
                Some(emoji_content) => {
                    let emoji = assets::Emoji::try_from(emoji_content.as_str());

                    match emoji {
                        Ok(emoji) => {
                            let asset: Asset = emoji.into();

                            string_build.push_str(&format!(
                                "![:{}:](/assets/{})",
                                emoji_content,
                                asset.bundled().bundled_path(),
                            ));
                        }
                        Err(_) => {
                            string_build.push(':');
                            string_build.push_str(emoji_content);
                            string_build.push(':');
                        }
                    }
                    emoji_text = None;
                }
                None => emoji_text = Some(String::new()),
            },
            ch => match &emoji_text {
                Some(_) => emoji_text.as_mut().unwrap().push(ch),
                None => string_build.push(ch),
            },
        }
    }

    if let Some(emoji_content) = emoji_text {
        string_build.push(':');
        string_build.push_str(&emoji_content);
    }

    let markdown = match markdown::to_html_with_options(&string_build, &markdown::Options::gfm()) {
        Ok(markdown) => markdown,
        Err(_) => markdown::to_html(&string_build),
    };

    rsx! {
        div { class: "markdown", ..attributes,
            div { dangerous_inner_html: markdown }

            {children}
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum InputComponentType {
    Text,
    Int {
        min: Option<i64>,
        max: Option<i64>,
        step: Option<i64>,
    },
    Float {
        min: Option<f64>,
        max: Option<f64>,
        step: Option<f64>,
    },
}

pub(crate) enum InputType {
    Text(String),
    Int(i64),
    Float(f64),
}

#[component]
pub(crate) fn InputComponent(
    id: String,
    label: String,
    component_type: InputComponentType,
    on_value_change: EventHandler<InputType>,
    #[props(extends = GlobalAttributes, extends = div)] attributes: Vec<Attribute>,
) -> Element {
    let mut value = use_signal(String::new);

    let r#type = match component_type {
        InputComponentType::Text => "text",
        InputComponentType::Int { .. } => "number",
        InputComponentType::Float { .. } => "number",
    };

    rsx! {
        div { class: "component-input", ..attributes,
            input {
                id: id.clone(),
                value: value(),
                placeholder: " ",
                r#type,
                oninput: move |evt| {
                    let new_value = evt.value();
                    value.set(new_value.clone());
                    match component_type {
                        InputComponentType::Text => {
                            on_value_change.call(InputType::Text(new_value));
                        }
                        InputComponentType::Int { .. } => {
                            if let Ok(number) = new_value.parse() {
                                on_value_change.call(InputType::Int(number));
                            }
                        }
                        InputComponentType::Float { .. } => {
                            if let Ok(number) = new_value.parse() {
                                on_value_change.call(InputType::Float(number));
                            }
                        }
                    }
                },
            }
            label { r#for: id.clone(), {label} }
        }
    }
}

#[component]
pub(crate) fn GeneralMenu(
    children: Element,
    on_close: EventHandler,
    x: f64,
    y: f64,
    #[props(extends = GlobalAttributes, extends = div)] attributes: Vec<Attribute>,
) -> Element {
    let mut position = use_signal(|| (x, y));
    let mut positioned = use_signal(|| false);

    let (menu_x, menu_y) = position();
    let visibility = if positioned() { "visible" } else { "hidden" };

    rsx! {
        div {
            class: "backdrop",
            onclick: move |_| on_close.call(()),
            oncontextmenu: move |evt| {
                on_close.call(());
                evt.stop_propagation();
                evt.prevent_default();
            },

            div {
                class: "_menu",
                style: "left: {menu_x}px; top: {menu_y}px; visibility: {visibility};",
                onclick: move |evt| {
                    evt.stop_propagation();
                },
                oncontextmenu: move |evt| {
                    evt.stop_propagation();
                    evt.prevent_default();
                },
                onmounted: move |evt| async move {
                    let Ok(rect) = evt.data().get_client_rect().await else {
                        positioned.set(true);
                        return;
                    };

                    let Some((viewport_width, viewport_height)) = viewport_size() else {
                        positioned.set(true);
                        return;
                    };

                    let (new_x, new_y) = fit_menu_position(
                        x,
                        y,
                        rect.size.width,
                        rect.size.height,
                        viewport_width,
                        viewport_height,
                    );
                    position.set((new_x, new_y));
                    positioned.set(true);
                },
                ..attributes,

                {children}
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct MenuItem {
    pub(crate) icon: Option<Asset>,
    pub(crate) label: String,
    pub(crate) on_click: EventHandler,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct MenuGroup {
    pub(crate) title: Option<String>,
    pub(crate) items: Vec<MenuItem>,
}

#[component]
pub(crate) fn Menu(groups: Vec<MenuGroup>, on_close: EventHandler, x: f64, y: f64) -> Element {
    rsx! {
        GeneralMenu { on_close, x, y,
            for group in groups {
                if let Some(title) = group.title {
                    h3 { class: "_menu-group-header", {title} }
                }

                for item in group.items {
                    button {
                        class: "_menu-group-item-button",
                        onclick: move |_| {
                            item.on_click.call(());
                            on_close.call(());
                        },

                        div { class: "_menu-group-item-button-wrapper",
                            if let Some(icon) = item.icon {
                                img { src: icon }
                            } else {
                                span {}
                            }

                            {item.label}
                        }
                    }
                }
            }
        }
    }
}

#[component]
pub(crate) fn ReactionMenu(
    session_uuid: Uuid,
    /// on_confirm 中不需要调用 close 代码，因为在组件中会自动调用 on_close
    on_confirm: EventHandler<(Vec<Option<Uuid>>, crate::shared::assets::Emoji)>,
    on_close: EventHandler,
    x: f64,
    y: f64,
) -> Element {
    let operator_view_model = use_context::<OperatorViewModel>();
    let operators = operator_view_model.operator_repository;

    let session_view_model = use_context::<SessionViewModel>();
    let sessions = session_view_model.sessions;

    let participants = sessions
        .read()
        .get(&session_uuid)
        .unwrap()
        .participants_ids()
        .iter()
        .map(|x| (Some(*x), operators.read().get(*x).unwrap().name().clone()))
        .chain(std::iter::once((None, String::from("管理员"))))
        .collect::<Vec<_>>();

    let mut emoji_selected = use_signal(|| None);
    let mut participants_ids_selected = use_signal(Vec::new);

    rsx! {
        GeneralMenu { on_close, x, y,
            h3 { class: "_reaction-menu-group-header", "添加Reaction" }

            div { class: "_reaction_menu_main",
                div { class: "_reaction_menu_emojis",
                    for emoji in crate::shared::assets::Emoji::VARIANTS {
                        button {
                            class: if emoji_selected() == Some(*emoji) { "_reaction_menu_emojis_button _selected" } else { "_reaction_menu_emojis_button" },
                            onclick: move |_| {
                                emoji_selected.set(Some(*emoji));
                            },
                            img { src: Asset::from(*emoji) }
                        }
                    }
                }

                div { class: "_reaction_menu_participants",
                    for participant in participants {
                        button {
                            class: if participants_ids_selected.read().contains(&participant.0) { "_reaction_menu_participants_button _selected" } else { "_reaction_menu_participants_button" },
                            onclick: move |_| {
                                if participants_ids_selected.read().contains(&participant.0) {
                                    participants_ids_selected.retain(|x| x != &participant.0);
                                } else {
                                    participants_ids_selected.push(participant.0);
                                }
                            },

                            if let Some(index) = participants_ids_selected.iter().position(|x| *x == participant.0) {
                                span { "{index + 1}" }
                            }

                            {participant.1}
                        }
                    }
                }
            }

            button {
                disabled: emoji_selected.read().is_none() || participants_ids_selected.is_empty(),
                class: "_reaction_menu_button",
                onclick: move |_| {
                    on_confirm.call((participants_ids_selected(), emoji_selected.unwrap()));
                    on_close.call(());
                },

                img { src: crate::shared::assets::icons::ADD_REACTION_48DP_000000_FILL0_WGHT400_GRAD0_OPSZ48 }

                "确定"
            }
        }
    }
}

#[component]
pub(crate) fn ReplayMenu(on_confirm: EventHandler<(i64, i64, i64)>, on_close: EventHandler, x: f64, y: f64) -> Element {
    let mut delay_input = use_signal(|| None);
    let mut delay_message = use_signal(|| None);
    let mut delay_reaction = use_signal(|| None);

    rsx! {
        GeneralMenu {
            on_close,
            x,
            y,
            class: "_menu replay-menu",
            h3 { class: "_reaction-menu-group-header", "回放" }
            InputComponent {
                id: "delay-input",
                label: "输入消息间隔（按ms记）",
                component_type: InputComponentType::Int {
                    min: Some(0),
                    max: Some(10000),
                    step: Some(100),
                },
                on_value_change: move |evt| {
                    match evt {
                        InputType::Int(delay) => delay_input.set(Some(delay)),
                        _ => unreachable!(),
                    }
                },
            }
            InputComponent {
                id: "delay-message",
                label: "消息间间隔（按ms记）",
                component_type: InputComponentType::Int {
                    min: Some(0),
                    max: Some(10000),
                    step: Some(100),
                },
                on_value_change: move |evt| {
                    match evt {
                        InputType::Int(delay) => delay_message.set(Some(delay)),
                        _ => unreachable!(),
                    }
                },
            }
            InputComponent {
                id: "delay-reaction",
                label: "Reaction间间隔（按ms记）",
                component_type: InputComponentType::Int {
                    min: Some(0),
                    max: Some(10000),
                    step: Some(100),
                },
                on_value_change: move |evt| {
                    match evt {
                        InputType::Int(delay) => delay_reaction.set(Some(delay)),
                        _ => unreachable!(),
                    }
                },
            }
            button {
                disabled: delay_input.read().is_none() || delay_message.read().is_none(),
                class: "_replay_menu_button",
                onclick: move |_| {
                    on_confirm
                        .call((
                            delay_input().unwrap(),
                            delay_message().unwrap(),
                            delay_reaction().unwrap(),
                        ));
                    on_close.call(());
                },

                img { src: crate::shared::assets::icons::REPLAY_48DP_000000_FILL0_WGHT400_GRAD0_OPSZ48 }

                "确定"
            }
        }
    }
}

#[component]
pub(crate) fn InputAreaMenu(children: Element) -> Element {
    let mut animation_end = use_signal(|| false);

    rsx! {
        div {
            id: "more-menu-wrapper",
            onanimationend: move |_| {
                animation_end.set(true);
            },
            if animation_end() {
                div { id: "more-menu", class: "menu", {children} }
            }
        }
    }
}
