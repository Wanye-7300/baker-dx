use dioxus::prelude::*;
use uuid::Uuid;

#[derive(Clone, Debug)]
pub(crate) struct SettingsState {
    pub(crate) image: Signal<Option<Uuid>>,
}
