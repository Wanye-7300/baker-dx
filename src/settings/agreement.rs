use crate::settings::state::SettingsState;
use crate::shared::setting::{SettingItem, SettingItemPage, SettingItemType, SettingPageView, SettingViewModel};
use crate::ui::Dialog;
use dioxus::prelude::*;
use uuid::Uuid;

#[component]
pub(crate) fn AgreementDialog(on_close: EventHandler) -> Element {
    let mut settings = use_context::<SettingsState>();
    let uuid = use_hook(Uuid::new_v4);
    let title = use_signal(|| "/ Baker // 使用协议".to_owned());
    let mut save_error = use_signal(|| None::<String>);
    let vm = use_signal(|| {
        SettingViewModel::new(
            "/ Baker // 使用协议".to_owned(),
            SettingItemPage::new().with_child(SettingItem::new(
                "MIT License".to_owned(),
                Some(include_str!("../../LICENSE").to_owned()),
                SettingItemType::Empty,
                None,
            )),
            false,
        )
    });

    rsx! {
        Dialog {
            title,
            uuid,
            on_close,
            SettingPageView { vm, caption: title }
            if let Some(message) = save_error() {
                p { role: "alert", "{message}" }
            }
            div { class: "dialog-buttons flex flex-row",
                button {
                    class: "dialog-button",
                    r#type: "button",
                    onclick: move |_| on_close.call(()),
                    "不同意"
                }
                button {
                    class: "dialog-buttons-confirm",
                    r#type: "button",
                    onclick: move |_| {
                        match crate::shared::utils::set_item("agreement_accepted", &true) {
                            Ok(()) => settings.agreement_accepted.set(true),
                            Err(_) => save_error.set(Some(
                                "无法保存同意状态，请检查浏览器是否允许本地存储，然后重试。".to_owned(),
                            )),
                        }
                    },
                    "同意并继续"
                }
            }
        }
    }
}
