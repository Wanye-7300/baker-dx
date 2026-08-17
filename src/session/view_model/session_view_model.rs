use crate::session::repository::*;

use dioxus::prelude::*;
use uuid::Uuid;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SessionViewModel {
    pub(crate) message_repository: Signal<MessageRepository>,
    pub(crate) sessions: Signal<SessionRepository>,
}

impl SessionViewModel {
    pub(crate) fn use_session_view_model_provider() -> anyhow::Result<()> {
        let sessions = SessionRepository::from_local_storage_or_default()?;
        let message_repository = use_signal(MessageRepository::new);
        let sessions = use_signal(|| sessions);

        use_context_provider(|| SessionViewModel {
            message_repository,
            sessions,
        });

        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Action(pub(crate) Uuid, pub(crate) u64, pub(crate) f64, pub(crate) f64);

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SessionUIViewModel {
    pub(crate) with_sender_selector_message_type: Signal<bool>,
    pub(crate) with_sender_selector_open: Signal<bool>,
    pub(crate) with_more_menu_open: Signal<bool>,
    pub(crate) with_stickers_menu_open: Signal<bool>,
    pub(crate) with_message_actions_menu_open: Signal<Option<Action>>,
    pub(crate) with_reaction_menu_open: Signal<Option<Action>>,
    pub(crate) need_to_scroll_down: Signal<bool>,
}

impl SessionUIViewModel {
    pub(crate) fn use_session_ui_view_model_provider() {
        let with_sender_selector_message_type = use_signal(|| true);
        let with_sender_selector_open = use_signal(|| false);
        let with_more_menu_open = use_signal(|| false);
        let with_stickers_menu_open = use_signal(|| false);
        let with_message_actions_menu_open = use_signal(|| None);
        let with_reaction_menu_open = use_signal(|| None);
        let need_to_scroll_down = use_signal(|| false);

        use_context_provider(|| SessionUIViewModel {
            with_sender_selector_message_type,
            with_sender_selector_open,
            with_more_menu_open,
            with_stickers_menu_open,
            with_message_actions_menu_open,
            with_reaction_menu_open,
            need_to_scroll_down,
        });
    }
}
