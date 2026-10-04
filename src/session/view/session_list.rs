use crate::{
    panic_try,
    session::{
        repository::MessageRepository,
        view_model::{
            input_view_model::InputViewModel,
            session_view_model::{SessionUIViewModel, SessionViewModel},
        },
    },
    ui::DialogNewSession,
};

use crate::shared::dialogs::{DialogUsage, DialogsManager};
use dioxus::prelude::*;
use fnv::FnvHashSet;
use uuid::Uuid;

#[component]
pub(crate) fn SessionList(session_name: Signal<String>, participants_ids: Signal<FnvHashSet<Uuid>>) -> Element {
    let mut dialogs_manager = use_context::<DialogsManager>();
    let session_view_model = use_context::<SessionViewModel>();

    rsx! {
        div { id: "session-cards", class: "flex flex-column",
            div { id: "cards-wrapper",
                for (uuid , session) in session_view_model.sessions.read().iterator() {
                    Card {
                        key: "{uuid.to_string()}",
                        uuid,
                        name: session.session_name().clone(),
                        avatar: session.avatar(),
                    }
                }
            }
            button {
                id: "cards-new",
                onclick: move |_| {
                    let uuid = Uuid::new_v4();
                    dialogs_manager
                        .append_dialog(uuid, DialogUsage::NewSession, rsx! {
                            DialogNewSession { session_name, participants_ids, uuid }
                        })
                        .unwrap();
                },
                span { "添加新会话" }
                img { src: crate::ICON_NEW_SESSION }
            }
        }
    }
}

#[component]
pub(crate) fn Card(uuid: Uuid, name: String, avatar: Asset) -> Element {
    let session_view_model = use_context::<SessionViewModel>();
    let session_ui_view_model = use_context::<SessionUIViewModel>();
    let input_view_model = use_context::<InputViewModel>();
    let dialogs_manager = use_context::<DialogsManager>();

    let on_choose_card = move |_| {
        let mut session_ui_view_model = session_ui_view_model.clone();
        let mut input_view_model = input_view_model.clone();
        let mut dialogs_manager = dialogs_manager.clone();
        async move {
            panic_try!(MessageRepository::select(session_view_model.message_repository, uuid).await);
            session_ui_view_model.reset();
            input_view_model.reset();
            dialogs_manager.remove_dialog_by_usage(DialogUsage::MessageProperties);
        }
    };

    rsx! {
        div { class: "card flex flex-row", onclick: on_choose_card,

            div { id: "card-img-wrapper",
                img { src: avatar }
            }

            {name}
        }
    }
}
