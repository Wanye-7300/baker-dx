use crate::shared::assets;

use dioxus::prelude::*;
use uuid::Uuid;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub(crate) enum InputAreaMessageType {
    #[default]
    Text,
    Image(Uuid),
    HorizontalBreak,
    State,
    StateWithHorizontalLine,
    Sticker(assets::stickers::Stickers),
}

/// 决定输入框的行为。
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub(crate) enum InputAreaMode {
    /// 正常模式：消息将被正常发送到会话末尾
    #[default]
    Normal,

    /// 插入模式：将在给定的 id 之前插入
    Insert { id: u64 },

    /// 修改模式
    Modify { id: u64 },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct InputViewModel {
    pub(crate) input_area_message_type: Signal<InputAreaMessageType>,
    pub(crate) input_area_text: Signal<String>,
    pub(crate) input_area_mode: Signal<InputAreaMode>,
}

impl InputViewModel {
    pub(crate) fn use_input_view_model_provider() {
        let input_area_message_type = use_signal(InputAreaMessageType::default);
        let input_area_text = use_signal(String::new);
        let input_area_mode = use_signal(InputAreaMode::default);

        use_context_provider(|| InputViewModel {
            input_area_message_type,
            input_area_text,
            input_area_mode,
        });
    }
}
