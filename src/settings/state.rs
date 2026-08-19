use crate::operator::model::Avatar;

use dioxus::prelude::*;
use uuid::Uuid;

#[derive(Clone, Debug)]
pub(crate) struct SettingsState {
    pub(crate) image: Signal<Option<Uuid>>,
    pub(crate) endministrator_avatar: Signal<Avatar>,
}
