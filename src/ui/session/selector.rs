use dioxus::prelude::*;
use uuid::Uuid;

#[component]
pub(crate) fn Selector(
    kv: Vec<(Uuid, String)>,
    title: Option<&'static str>,
    additional_class: Option<&'static str>,
    func: EventHandler<Uuid>,
    on_close: EventHandler,
) -> Element {
    let selector_class = match additional_class {
        Some(add) => format!("selector {}", add),
        None => "selector".to_string(),
    };

    rsx! {
        div { class: selector_class,
            if let Some(title) = title {
                h3 { {title.to_string()} }
            }
            for (k , v) in kv {
                button { key: "{k}", onclick: move |_| func.call(k), {v} }
            }
        }
        div { class: "backdrop", onclick: move |_| on_close.call(()) }
    }
}
