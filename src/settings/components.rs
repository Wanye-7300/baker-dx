use super::state::SettingsState;

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
                                crate::database::save_multimedia(uuid, file.into()).await.unwrap();

                                if let Some(uuid) = uuid_remaining() {
                                    crate::database::remove_multimedia(uuid).await.unwrap();
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
pub(crate) fn Settings(on_close: EventHandler) -> Element {
    let mut settings_state = use_context::<SettingsState>();

    use_effect(move || {
        settings_state.image.read();

        crate::utils::set_item("wallpaper", &(settings_state.image)());
    });

    rsx! {
        div { class: "backdrop centered", onclick: move |_| on_close.call(()),
            div {
                id: "settings",
                onclick: move |evt| {
                    evt.stop_propagation();
                },
                div { id: "settings-title",
                    h2 { "/ Baker // 设置" }
                    button { onclick: move |_| on_close.call(()), "×" }
                }

                div { id: "settings-objects",
                    h3 { "一般" }

                    ImageSetting {
                        object_name: "墙纸",
                        on_change: move |uuid| {
                            settings_state.image.set(Some(uuid));
                            info!("Wallpaper has been set to: {}", uuid);
                        },
                        uuid: (settings_state.image)(),
                    }
                }
            }
        }
    }
}
