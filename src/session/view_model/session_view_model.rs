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

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ReplayMode {
    pub(crate) message_id: u64,
    pub(crate) delay_input: i64,
    pub(crate) delay_message: i64,
    pub(crate) delay_reaction: i64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SessionUIViewModel {
    pub(crate) with_sender_selector_message_type: Signal<bool>,
    pub(crate) with_sender_selector_open: Signal<bool>,
    pub(crate) with_more_menu_open: Signal<bool>,
    pub(crate) with_stickers_menu_open: Signal<bool>,
    pub(crate) with_message_actions_menu_open: Signal<Option<Action>>,
    pub(crate) with_reaction_menu_open: Signal<Option<Action>>,
    pub(crate) with_replay_menu_open: Signal<Option<Action>>,
    pub(crate) need_to_scroll_down: Signal<bool>,

    /// 是否在回放模式。如果是，则是 `Some(ReplayMode { .. })`，指示从哪个消息开始回放和延迟。
    pub(crate) replay_mode: Signal<Option<ReplayMode>>,
}

impl SessionUIViewModel {
    pub(crate) fn use_session_ui_view_model_provider() {
        let with_sender_selector_message_type = use_signal(|| true);
        let with_sender_selector_open = use_signal(|| false);
        let with_more_menu_open = use_signal(|| false);
        let with_stickers_menu_open = use_signal(|| false);
        let with_message_actions_menu_open = use_signal(|| None);
        let with_reaction_menu_open = use_signal(|| None);
        let with_replay_menu_open = use_signal(|| None);
        let need_to_scroll_down = use_signal(|| false);
        let replay_mode = use_signal(|| None);

        use_context_provider(|| SessionUIViewModel {
            with_sender_selector_message_type,
            with_sender_selector_open,
            with_more_menu_open,
            with_stickers_menu_open,
            with_message_actions_menu_open,
            with_reaction_menu_open,
            with_replay_menu_open,
            need_to_scroll_down,
            replay_mode,
        });
    }

    pub(crate) fn reset(&mut self) {
        self.with_sender_selector_open.set(false);
        self.with_more_menu_open.set(false);
        self.with_stickers_menu_open.set(false);
        self.with_message_actions_menu_open.set(None);
        self.with_reaction_menu_open.set(None);
        self.need_to_scroll_down.set(true);
        self.replay_mode.set(None);
    }
}
