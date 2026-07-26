use dioxus::prelude::*;
use uuid::Uuid;

#[component]
pub(crate) fn Selector(
    kv: Vec<(Option<Uuid>, String)>,
    title: Option<&'static str>,
    optional_kv: Option<Vec<(String, String)>>,
    optional_title: Option<String>,
    additional_class: Option<&'static str>,
    func: EventHandler<(Option<String>, Option<Uuid>)>,
    on_close: EventHandler,
) -> Element {
    let mut selected = use_signal(|| optional_kv.as_ref().and_then(|x| x.first().map(|x| x.0.to_string())));

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
            if let Some(title) = optional_title {
                h3 { {title.to_string()} }
            }
            if let Some(optional_kv) = optional_kv {
                div { class: "optional_wrapper",
                    for (k , v) in optional_kv {
                        label { key: "{k.to_string()}",
                            input {
                                r#type: "radio",
                                name: "optional",
                                value: k.to_string(),
                                checked: selected() == Some(k.to_owned()),
                                onchange: move |_| {
                                    selected.set(Some(k.to_owned()));
                                },
                            }
                            {v}
                        }
                    }
                }
            }
            if let Some(title) = title {
                h3 { {title.to_string()} }
            }
            for (k , uuid , v) in kv {
                button {
                    key: "{k}",
                    onclick: move |_| func.call((selected(), uuid)),
                    {v.to_string()}
                }
            }
        }
        div { class: "backdrop", onclick: move |_| on_close.call(()) }
    }
}
