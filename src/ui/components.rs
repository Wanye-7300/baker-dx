use super::assets::EMOJI;
use dioxus::prelude::*;

#[component]
pub(crate) fn RichText(
    text: String,
    #[props(extends = GlobalAttributes, extends = div)] attributes: Vec<Attribute>,
) -> Element {
    let mut string_build = String::with_capacity(text.capacity());
    let mut emoji: Option<String> = None;
    for chr in text.chars() {
        match chr {
            ':' => match &emoji {
                Some(emoji_content) => {
                    if let Some(asset) = EMOJI.get(emoji_content.as_str()) {
                        string_build.push_str(&format!(
                            "![:{}:](/assets/{})",
                            emoji_content,
                            asset.bundled().bundled_path(),
                        ));
                    } else {
                        string_build.push(':');
                        string_build.push_str(emoji_content);
                        string_build.push(':');
                    }
                    emoji = None;
                }
                None => emoji = Some(String::new()),
            },
            ch => match &emoji {
                Some(_) => emoji.as_mut().unwrap().push(ch),
                None => string_build.push(ch),
            },
        }
    }

    if let Some(emoji_content) = emoji {
        string_build.push(':');
        string_build.push_str(&emoji_content);
    }

    let markdown = match markdown::to_html_with_options(&string_build, &markdown::Options::gfm()) {
        Ok(markdown) => markdown,
        Err(_) => markdown::to_html(&string_build),
    };

    rsx! {
        div { class: "markdown", dangerous_inner_html: markdown, ..attributes }
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
        div { class: "backdrop", onclick: move |_| on_close.call(()),
            div { class: "_menu", style: "left: {x}px; top: {y}px",

                for group in groups {
                    if let Some(title) = group.title {
                        h3 { class: "_menu-group-header", {title} }
                    }

                    for item in group.items {
                        button {
                            class: "_menu-group-item-button",
                            onclick: move |_| item.on_click.call(()),

                            if let Some(icon) = item.icon {
                                img { class: "_menu-group-item-icon", src: icon }
                            }

                            {item.label}
                        }
                    }
                }
            }
        }
    }
}
