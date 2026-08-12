use dioxus::prelude::*;

#[component]
pub(crate) fn RichText(
    text: String,
    #[props(extends = GlobalAttributes, extends = div)] attributes: Vec<Attribute>,
) -> Element {
    let markdown = markdown::to_html(&text);

    rsx! {
        div { class: "markdown", dangerous_inner_html: markdown, ..attributes }
    }
}
