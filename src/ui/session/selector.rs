use dioxus::prelude::*;
use uuid::Uuid;

#[component]
pub(crate) fn Selector(
    kv: Vec<(Option<Uuid>, String)>,
    title: Option<&'static str>,
    message_type_selector: Option<bool>,
    additional_class: Option<&'static str>,
    func: EventHandler<(Option<super::InputAreaMessageType>, crate::Sender)>,
    on_close: EventHandler,
) -> Element {
    let message_type_selector = message_type_selector.unwrap_or_default();

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
            if message_type_selector {
                h3 { "发送为……" }
                for (k , v) in vec![
                    (super::InputAreaMessageType::HorizontalBreak, "分隔线".to_string()),
                    (super::InputAreaMessageType::State, "“状态”".to_string()),
                    (
                        super::InputAreaMessageType::StateWithHorizontalLine,
                        "“状态”（带分隔符）".to_string(),
                    ),
                ]
                {
                    button {
                        key: "{k:?}",
                        onclick: move |_| func.call((Some(k), crate::Sender::None)),
                        {v.to_string()}
                    }
                }
            }
            if let Some(title) = title {
                h3 { {title.to_string()} }
            }
            for (k , uuid , v) in kv {
                button {
                    key: "{k}",
                    onclick: move |_| func.call((None, crate::Sender::from_optional_uuid(uuid))),
                    {v.to_string()}
                }
            }
        }
        div { class: "backdrop", onclick: move |_| on_close.call(()) }
    }
}
