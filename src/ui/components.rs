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

    info!("Build: {}", string_build);

    let markdown = markdown::to_html(&string_build);

    rsx! {
        div { class: "markdown", dangerous_inner_html: markdown, ..attributes }
    }
}
