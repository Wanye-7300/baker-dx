use crate::shared::setting::SettingPageView;
use crate::ui::Dialog;

use crate::session::view_model::session_view_model::SessionViewModel;
use dioxus::prelude::*;
use uuid::Uuid;

pub(crate) mod session;
pub(crate) mod session_list;

#[component]
pub(crate) fn MessageProperties(
    message: super::model::Message,
    session_uuid: Uuid,
    message_id: u64,
    dialog_uuid: Uuid,
) -> Element {
    let session_view_model = use_context::<SessionViewModel>();
    let repository = session_view_model.message_repository;
    let vm = use_signal(|| message.get_settings_vm(repository, session_uuid, message_id));

    let dialog_uuid = use_hook(|| dialog_uuid);
    let title = use_signal(|| format!("属性 {}:{}", session_uuid, message_id));

    rsx! {
        Dialog { title, uuid: dialog_uuid,

            div {
                SettingPageView { vm, caption: title }
            }
        }
    }
}
