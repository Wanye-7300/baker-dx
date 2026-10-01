use dioxus::prelude::*;
use uuid::Uuid;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DialogUsage {
    GeneralSettingPage,
    NewSession,
    ManageOperators,
}

#[derive(Clone)]
struct DialogEntry {
    uuid: Uuid,
    usage: DialogUsage,
    dialog: Element,
}

#[derive(Clone)]
pub(crate) struct DialogsManager {
    dialogs: Signal<Vec<DialogEntry>>,
}

impl DialogsManager {
    pub(crate) fn provide_dialog_manager() {
        let dialogs = use_signal(|| vec![]);
        use_context_provider(|| DialogsManager { dialogs });
    }

    pub(crate) fn append_dialog(
        &mut self,
        uuid: Uuid,
        usage: DialogUsage,
        dialog: Element,
    ) -> std::result::Result<Uuid, Element> {
        if !self.dialogs.iter().any(|x| x.usage == usage) {
            self.dialogs.push(DialogEntry { uuid, usage, dialog });
        }
        Ok(uuid)
    }

    pub(crate) fn remove_dialog(&mut self, uuid: Uuid) {
        self.dialogs.retain(|x| x.uuid != uuid);
    }

    pub(crate) fn rendered(&self) -> Element {
        rsx! {
            for dialog in self.dialogs.iter() {
                {dialog.dialog.clone()}
            }
        }
    }
}
