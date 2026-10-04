use dioxus::prelude::*;
use uuid::Uuid;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DialogUsage {
    GeneralSettingPage,
    NewSession,
    ManageOperators,
    MessageProperties,
}

#[derive(Clone)]
struct DialogEntry {
    uuid: Uuid,
    usage: DialogUsage,
    dialog: Element,
    /// 管理器内部的层级，0 为最底层；不改变窗口的渲染顺序。
    layer: usize,
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
            let layer = self.dialogs.len();
            self.dialogs.push(DialogEntry {
                uuid,
                usage,
                dialog,
                layer,
            });
        }
        Ok(uuid)
    }

    pub(crate) fn remove_dialog(&mut self, uuid: Uuid) {
        let Some(layer) = self
            .dialogs
            .peek()
            .iter()
            .find(|entry| entry.uuid == uuid)
            .map(|entry| entry.layer)
        else {
            return;
        };

        let mut dialogs = self.dialogs.write();
        dialogs.retain(|entry| entry.uuid != uuid);
        for entry in dialogs.iter_mut() {
            if entry.layer > layer {
                entry.layer -= 1;
            }
        }
    }

    pub(crate) fn bring_to_front(&mut self, uuid: Uuid) {
        let (index, layer, top) = {
            let dialogs = self.dialogs.peek();
            let Some(index) = dialogs.iter().position(|entry| entry.uuid == uuid) else {
                return;
            };
            let layer = dialogs[index].layer;
            let top = dialogs.len() - 1;
            if layer == top {
                return;
            }
            (index, layer, top)
        };

        let mut dialogs = self.dialogs.write();
        for entry in dialogs.iter_mut() {
            if entry.layer > layer {
                entry.layer -= 1;
            }
        }
        dialogs[index].layer = top;
    }

    pub(crate) fn rendered(&self) -> Element {
        rsx! {
            div { class: "dialogs-layer",
                for dialog in self.dialogs.iter() {
                    div {
                        key: "{dialog.uuid}",
                        class: "dialog-host",
                        z_index: "{dialog.layer}",
                        {dialog.dialog.clone()}
                    }
                }
            }
        }
    }
}
