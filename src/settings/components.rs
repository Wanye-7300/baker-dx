use super::state::SettingsState;
use crate::shared::setting::*;
use crate::ui::Dialog;
use crate::{operator::model::Avatar, panic_try, shared::assets};

use dioxus::{prelude::*, web::WebFileExt as _};
use uuid::Uuid;

#[component]
pub(crate) fn ImageSetting(object_name: String, on_change: EventHandler<Uuid>, uuid: Option<Uuid>) -> Element {
    let mut uuid_remaining = use_signal(|| uuid);
    let mut with_input_disabled = use_signal(|| false);

    rsx! {
        div { class: "settings-object",
            span { {object_name} }
            label { class: "choose-file",
                "上传文件"
                input {
                    r#type: "file",
                    disabled: with_input_disabled(),
                    hidden: true,
                    onchange: move |evt: Event<FormData>| {
                        if let Some(file_data) = evt.files().first() {
                            let file = file_data.get_web_file().unwrap();

                            spawn(async move {
                                with_input_disabled.set(true);
                                let uuid = Uuid::new_v4();
                                crate::shared::database::save_multimedia(uuid, file.into())
                                    .await
                                    .unwrap();
                                if let Some(uuid) = uuid_remaining() {
                                    crate::shared::database::remove_multimedia(uuid).await.unwrap();
                                }
                                uuid_remaining.set(Some(uuid));
                                with_input_disabled.set(false);
                                on_change.call(uuid);
                            });
                        }
                    },
                }
            }

        }
    }
}

#[component]
pub(crate) fn AvatarSetting(object_name: String, on_change: EventHandler<Avatar>, value: Avatar) -> Element {
    let mut avatar_id = use_signal(|| value.clone());

    rsx! {
        div { class: "settings-object",
            span { {object_name} }

            select {
                name: "avatar",
                id: "avatar-select",
                onchange: move |evt| {
                    avatar_id.set(Avatar::Preset(evt.value()));
                    on_change.call(Avatar::Preset(evt.value()));
                },
                option { value: "", "选择头像" }
                for k in assets::CHARACTERS_AVATARS.keys() {
                    option {
                        selected: if let Avatar::Preset(string) = value.clone() && *k == string { true },
                        value: k,
                        "{assets::CHARACTERS_NAME[k]}"
                    }
                }
            }
        }
    }
}

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

    let vm = use_signal(|| {
        SettingViewModel::new(
            SETTING_WINDOW_TITLE.to_string(),
            SettingItemPage::new()
                .with_child(SettingItem::new(
                    "墙纸".to_owned(),
                    Some("设置应用背景的墙纸。".to_owned()),
                    SettingItemType::Image,
                    None,
                ))
                .with_child(SettingItem::new(
                    "SelfAvatar".to_owned(),
                    Some("设置管理员自己的头像。".to_owned()),
                    SettingItemType::Selection {
                        selections: assets::CHARACTERS_AVATARS.keys().map(|x| (*x).to_owned()).collect(),
                        default: "endministratorf".to_owned(),
                    },
                    None,
                ))
                .with_child(SettingItem::new(
                    "名字".to_owned(),
                    Some("设置管理员自己的名字。".to_owned()),
                    SettingItemType::Str {
                        default: "管理员".to_owned(),
                    },
                    None,
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
