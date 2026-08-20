use crate::{
    panic_try,
    session::{
        repository::MessageRepository,
        view_model::session_view_model::{SessionUIViewModel, SessionViewModel},
    },
    ui::DialogNewSession,
};

use dioxus::prelude::*;
use fnv::FnvHashSet;
use uuid::Uuid;

#[component]
pub(crate) fn SessionList(session_name: Signal<String>, participants_ids: Signal<FnvHashSet<Uuid>>) -> Element {
    let mut baker_state = use_context::<crate::BakerState>();
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
                    baker_state.dialogs.write().insert(uuid, rsx! {
                        DialogNewSession { session_name, participants_ids, uuid }
                    });
                },
                "添加新会话"
            }
        }
    }
}

#[component]
pub(crate) fn Card(uuid: Uuid, name: String, avatar: Asset) -> Element {
    let session_view_model = use_context::<SessionViewModel>();
    let session_ui_view_model = use_context::<SessionUIViewModel>();

    let mut need_to_scroll_down = session_ui_view_model.need_to_scroll_down;

    let on_choose_card = move |_| async move {
        panic_try!(MessageRepository::select(session_view_model.message_repository, uuid).await);
        need_to_scroll_down.set(true);
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
