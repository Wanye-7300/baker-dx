use super::assets;
use dioxus::prelude::*;
use strum::VariantArray;
use uuid::Uuid;

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
                style: "left: {x}px; top: {y}px",
                onclick: move |evt| {
                    evt.stop_propagation();
                },
                oncontextmenu: move |evt| {
                    evt.stop_propagation();
                    evt.prevent_default();
                },

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
}

#[component]
pub(crate) fn ReactionMenu(
    session_uuid: Uuid,
    /// on_confirm 中不需要调用 close 代码，因为在组件中会自动调用 on_close
    on_confirm: EventHandler<(Vec<Option<Uuid>>, assets::Emoji)>,
    on_close: EventHandler,
    x: f64,
    y: f64,
) -> Element {
    let baker = use_context::<crate::BakerState>();
    let participants = baker
        .sessions
        .get(&session_uuid)
        .unwrap()
        .participants_ids
        .iter()
        .map(|x| (Some(*x), baker.operators.get(x).unwrap().name.clone()))
        .chain(std::iter::once((None, String::from("管理员"))))
        .collect::<Vec<_>>();

    let mut emoji_selected = use_signal(|| None);
    let mut participants_ids_selected = use_signal(Vec::new);

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
                class: "_menu _reaction_menu",
                style: "left: {x}px; top: {y}px",
                onclick: move |evt| {
                    evt.stop_propagation();
                },
                oncontextmenu: move |evt| {
                    evt.stop_propagation();
                    evt.prevent_default();
                },

                h3 { class: "_reaction-menu-group-header", "添加Reaction" }

                div { class: "_reaction_menu_main",
                    div { class: "_reaction_menu_emojis",
                        for emoji in assets::Emoji::VARIANTS {
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

                    img { src: assets::icons::ADD_REACTION_48DP_000000_FILL0_WGHT400_GRAD0_OPSZ48 }

                    "确定"
                }
            }
        }
    }
}
