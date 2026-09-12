use super::state::SettingsState;
use crate::shared::setting::*;
use crate::ui::Dialog;
use crate::{operator::model::Avatar, panic_try, shared::assets};

use dioxus::prelude::*;
use uuid::Uuid;

#[component]
pub(crate) fn Settings(on_close: EventHandler) -> Element {
    let mut settings_state = use_context::<SettingsState>();

    use_effect(move || {
        settings_state.image.read();

        panic_try!(crate::shared::utils::set_item("wallpaper", &(settings_state.image)()));
    });

    use_effect(move || {
        settings_state.endministrator_avatar.read();

        panic_try!(crate::shared::utils::set_item(
            "E_avatar",
            &(settings_state.endministrator_avatar)()
        ));
    });

    use_effect(move || {
        settings_state.endministrator_name.read();

        panic_try!(crate::shared::utils::set_item(
            "E_name",
            &(settings_state.endministrator_name)()
        ));
    });

    let vm = use_signal(move || {
        SettingViewModel::new(
            SETTING_WINDOW_TITLE.to_string(),
            SettingItemPage::new()
                .with_child(SettingItem::new(
                    "墙纸".to_owned(),
                    Some("设置应用背景的墙纸。".to_owned()),
                    SettingItemType::Image {
                        value: (settings_state.image)(),
                    },
                    Some(EventHandler::new(move |value: SettingItemValue| {
                        if let SettingItemValue::Image(uuid) = value {
                            settings_state.image.set(Some(uuid));
                        }
                    })),
                ))
                .with_child(SettingItem::new(
                    "SelfAvatar".to_owned(),
                    Some("设置管理员自己的头像。".to_owned()),
                    SettingItemType::Selection {
                        selections: assets::CHARACTERS_AVATARS.keys().map(|x| (*x).to_owned()).collect(),
                        value: match (settings_state.endministrator_avatar)() {
                            Avatar::Preset(id) => id,
                            _ => "endministratorf".to_owned(),
                        },
                    },
                    Some(EventHandler::new(move |value: SettingItemValue| {
                        if let SettingItemValue::Selection(id) = value {
                            settings_state.endministrator_avatar.set(Avatar::Preset(id));
                        }
                    })),
                ))
                .with_child(SettingItem::new(
                    "名字".to_owned(),
                    Some("设置管理员自己的名字。".to_owned()),
                    SettingItemType::Str {
                        value: (settings_state.endministrator_name)(),
                    },
                    Some(EventHandler::new(move |value: SettingItemValue| {
                        if let SettingItemValue::Str(name) = value {
                            settings_state.endministrator_name.set(name);
                        }
                    })),
                ))
                .with_child(SettingItem::new(
                    "关于项目".to_owned(),
                    None,
                    SettingItemType::Page(SettingItemPage::new().with_child(SettingItem::new(
                        "本项目采用 MIT 协议".to_owned(),
                        Some(include_str!("../../LICENSE").to_owned()),
                        SettingItemType::Empty,
                        None,
                    ))),
                    None,
                )),
            true,
        )
    });

    let uuid = use_hook(Uuid::new_v4);
    let title = use_signal(|| SETTING_WINDOW_TITLE.to_string());

    rsx! {
        Dialog {
            title,
            uuid,
            on_close: move |_| on_close.call(()),
            on_confirm: move |_| on_close.call(()),

            div { id: "settings",
                SettingPageView { vm, caption: title }
            }
        }
    }
}
