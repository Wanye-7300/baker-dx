use dioxus::prelude::*;
use uuid::Uuid;

#[component]
pub(crate) fn Selector(
    kv: Vec<(Option<Uuid>, String)>,
    title: Option<&'static str>,
    additional_class: Option<&'static str>,
    func: EventHandler<Option<Uuid>>,
    on_close: EventHandler,
) -> Element {
    let selector_class = match additional_class {
        Some(add) => format!("selector {}", add),
        None => "selector".to_string(),
    };

    let kv: Vec<(String, Option<Uuid>, &String)> = kv
        .iter()
        .map(|x| {
            (
                match x.0 {
                    Some(x) => format!("{x}"),
                    None => "endministratorf".to_string(),
                },
                x.0,
                &x.1,
            )
        })
        .collect();

    rsx! {
        div { class: selector_class,
            if let Some(title) = title {
                h3 { {title.to_string()} }
            }
            for (k , uuid , v) in kv {
                button { key: "{k}", onclick: move |_| func.call(uuid), "{v}" }
            }
        }
        div { class: "backdrop", onclick: move |_| on_close.call(()) }
    }
}
