use self::components::{InputComponent, InputComponentType, InputType};
use crate::operator::model::*;
use crate::operator::view_model::OperatorViewModel;
use crate::session::model::*;
use crate::session::view::session::*;
use crate::session::view::session_list::*;
use crate::session::view_model::session_view_model::SessionViewModel;
use crate::shared::assets;

use crate::shared::dialogs::{DialogUsage, DialogsManager};
use dioxus::prelude::*;
use fnv::FnvHashSet;
use uuid::Uuid;

pub(crate) mod components;
pub(crate) mod selector;

struct ObjectUrl(String);

impl Drop for ObjectUrl {
    fn drop(&mut self) {
        let _ = web_sys::Url::revoke_object_url(&self.0);
    }
}

/// 标题栏高度，与 CSS 中 `.dialog .dialog-title` 保持一致。
const CAPTION_HEIGHT: f64 = 32.0;

/// 窗口被拖出视口时，标题栏至少要留在视口内的宽度与高度，保证窗口还能再次被拖回来。
const CAPTION_VISIBLE_WIDTH: f64 = 120.0;
const CAPTION_VISIBLE_HEIGHT: f64 = 8.0;

/// 窗口拖动状态。
#[derive(Clone, Copy, PartialEq, Debug)]
struct DialogDrag {
    /// 发起拖动的指针，避免其他触点干扰。
    pointer_id: i32,
    /// 按下时指针相对窗口左上角的偏移。
    grab_x: f64,
    grab_y: f64,
    /// 窗口宽度，用于限制窗口横向拖出视口的距离。
    width: f64,
}

/// 读取视口尺寸。
fn viewport_size() -> (f64, f64) {
    let Some(window) = web_sys::window() else {
        return (0.0, 0.0);
    };

    let width = window
        .inner_width()
        .ok()
        .and_then(|value| value.as_f64())
        .unwrap_or(0.0);
    let height = window
        .inner_height()
        .ok()
        .and_then(|value| value.as_f64())
        .unwrap_or(0.0);

    (width, height)
}

/// 获取窗口本体，用于读取位置和捕获拖动指针。
fn dialog_element(uuid: &Uuid) -> Option<web_sys::Element> {
    web_sys::window()?
        .document()?
        .get_element_by_id(&format!("dialog-{uuid}"))
}

/// 允许窗口被拖出视口，但始终保留一部分标题栏可见，保证还能拖回来。
fn clamp_position(left: f64, top: f64, width: f64, viewport: (f64, f64)) -> (f64, f64) {
    let (viewport_width, viewport_height) = viewport;

    // 横向：左右各自最多拖出「窗口宽度 - 120px」，即至少 120px 标题栏留在视口内
    let min_left = CAPTION_VISIBLE_WIDTH - width;
    let max_left = (viewport_width - CAPTION_VISIBLE_WIDTH).max(min_left);

    // 纵向：标题栏上沿最多到 -24px（下沿仍留 8px），下沿最多到视口底部上方 8px
    let min_top = CAPTION_VISIBLE_HEIGHT - CAPTION_HEIGHT;
    let max_top = (viewport_height - CAPTION_VISIBLE_HEIGHT).max(min_top);

    (left.clamp(min_left, max_left), top.clamp(min_top, max_top))
}

#[component]
pub(super) fn Baker() -> Element {
    use_hook(crate::shared::panic::install_panic_hook);

    let dialogs_manager = use_context::<DialogsManager>();
    let settings_state = use_context::<crate::settings::state::SettingsState>();

    let session_name = use_signal(String::new);
    let participants_ids = use_signal(FnvHashSet::default);

    rsx! {
        div { id: "app", class: "flex flex-column",
            div {
                ondoubleclick: {
                    let mut dialogs_manager = dialogs_manager.clone();
                    move |evt| {
                        evt.stop_propagation();
                        let uuid = Uuid::new_v4();
                        dialogs_manager
                            .append_dialog(uuid, DialogUsage::GeneralSettingPage, rsx! {
                                crate::settings::components::Settings { uuid }
                            })
                            .unwrap();
                    }
                },
                id: "title",
                "//BAKER/会话消息"

                img {
                    ondoubleclick: move |evt| {
                        evt.stop_propagation();
                    },
                    onclick: move |evt| {
                        evt.stop_propagation();
                    },
                    id: "title-decoration",
                    src: crate::ACHIEVEMENT_MAIN_DECO05,
                }
            }
            div { id: "main-content", class: "flex flex-row",
                SessionList { session_name, participants_ids }
                SessionUI {}
            }
        }

        {dialogs_manager.rendered()}

        if let Some(uuid) = (settings_state.image)() {
            Image { id: "background-image", uuid }
        }
    }
}

#[component]
pub(crate) fn Image(
    uuid: ReadSignal<Uuid>,
    #[props(extends = GlobalAttributes, extends = img)] attributes: Vec<Attribute>,
) -> Element {
    let image_url = use_resource(move || {
        let uuid = uuid();

        async move {
            let blob = crate::shared::database::get_multimedia(uuid)
                .await
                .map_err(|error| format!("读取图片失败：{error}"))?
                .ok_or_else(|| "图片数据不存在".to_string())?;
            let url = web_sys::Url::create_object_url_with_blob(&blob).map_err(|_| "创建图片地址失败".to_string())?;

            Ok::<ObjectUrl, String>(ObjectUrl(url))
        }
    });

    let image_url = image_url.read();

    match image_url.as_ref() {
        Some(Ok(url)) => rsx! {
            img { src: url.0.clone(), ..attributes }
        },
        Some(Err(error)) => rsx! {
            span { class: "message-image-error", role: "alert", {error.to_string()} }
        },
        None => rsx! {},
    }
}

#[component]
pub(crate) fn ParticipantsSelection(participants_ids: Signal<fnv::FnvHashSet<Uuid>>) -> Element {
    let operator_view_model = use_context::<OperatorViewModel>();
    let operators = operator_view_model.operator_repository;

    let operators = operators
        .read()
        .iterator()
        .filter(|(_, operator)| operator.activity())
        .map(|(id, operator)| (id, operator.clone()))
        .collect::<Vec<_>>();

    rsx! {
        div { class: "participants",
            for (k , v) in operators {
                div { class: "participant",
                    input {
                        r#type: "checkbox",
                        id: k.to_string(),
                        name: k.to_string(),
                        checked: participants_ids.read().get(&k).is_some(),
                        onchange: move |_| {
                            if participants_ids.read().get(&k).is_none() {
                                participants_ids.write().insert(k);
                            } else {
                                participants_ids.write().remove(&k);
                            }
                        },
                    }
                    label { r#for: k.to_string(),
                        {v.name().clone()}
                        {"   "}
                        {k.to_string()}
                    }
                }
            }
        }
    }
}

#[component]
pub(crate) fn Dialog(
    mut title: Signal<String>,
    on_confirm: Option<EventHandler>,
    uuid: Uuid,
    /// 关闭方式：传入时交给外部处理，否则把这个对话框从 dialogs 表里移除
    #[props(default)]
    on_close: Option<EventHandler>,
    #[props(default)] confirm_disabled: bool,
    children: Element,
) -> Element {
    let dialogs_manager = use_context::<DialogsManager>();
    let activate = EventHandler::new({
        let mut dialogs_manager = dialogs_manager.clone();
        move |()| dialogs_manager.bring_to_front(uuid)
    });
    // 仅记录拖动手势；窗口位置由拖动事件直接写入 DOM，初始位置交给 CSS。
    let mut drag = use_signal(|| None::<DialogDrag>);

    let mut end_drag = move |evt: PointerEvent| {
        if drag().is_some_and(|state| state.pointer_id == evt.pointer_id()) {
            drag.set(None);
        }
    };

    rsx! {
        div {
            key: "{uuid}",
            id: "dialog-{uuid}",
            class: "dialog flex flex-column",
            onclick: move |evt| evt.stop_propagation(),
            onpointerdown: move |_| activate.call(()),
            onfocusin: move |_| activate.call(()),
            // 标题栏按下时捕获指针，移出窗口后仍可接收移动和松开事件。
            onpointermove: move |evt| {
                let Some(state) = drag() else {
                    return;
                };
                if state.pointer_id != evt.pointer_id() {
                    return;
                }

                let point = evt.client_coordinates();
                let (left, top) = clamp_position(
                    point.x - state.grab_x,
                    point.y - state.grab_y,
                    state.width,
                    viewport_size(),
                );

                if let Some(element) = dialog_element(&uuid) {
                    // RSX 不管理此 style，避免重渲染覆盖拖动后的位置。
                    let _ = element
                        .set_attribute(
                            "style",
                            &format!(
                                "left: {left}px; top: {top}px; bottom: auto; translate: none;",
                            ),
                        );
                }
            },
            // pointerup / pointercancel 后浏览器自动释放捕获；意外丢失捕获也结束拖动。
            onpointerup: move |evt| end_drag(evt),
            onpointercancel: move |evt| end_drag(evt),
            onlostpointercapture: move |evt| end_drag(evt),
            // 左侧标题 + 右侧关闭按钮；标题栏本身是拖动把手
            div {
                class: "dialog-title flex flex-row",
                onpointerdown: move |evt| {
                    activate.call(());
                    evt.stop_propagation();
                    if drag().is_some() {
                        return;
                    }
                    let Some(element) = dialog_element(&uuid) else {
                        return;
                    };
                    let rect = element.get_bounding_client_rect();
                    let pointer_id = evt.pointer_id();
                    if element.set_pointer_capture(pointer_id).is_err() {
                        return;
                    }

                    let point = evt.client_coordinates();
                    drag.set(
                        Some(DialogDrag {
                            pointer_id,
                            grab_x: point.x - rect.left(),
                            grab_y: point.y - rect.top(),
                            width: rect.width(),
                        }),
                    );
                },
                span { class: "dialog-title-text", "{title}" }
                button {
                    class: "dialog-title-close",
                    r#type: "button",
                    title: "关闭",
                    aria_label: "关闭",
                    // 从关闭按钮上按下不参与拖动
                    onpointerdown: move |evt| {
                        activate.call(());
                        evt.stop_propagation();
                    },
                    onclick: {
                        let mut dialogs_manager = dialogs_manager.clone();
                        move |_| {
                            if let Some(handler) = on_close {
                                handler.call(());
                            } else {
                                dialogs_manager.remove_dialog(uuid);
                            }
                        }
                    },
                    svg {
                        class: "caption-glyph",
                        view_box: "0 0 10 10",
                        width: "10",
                        height: "10",
                        path { d: "M0 0 L10 10 M10 0 L0 10" }
                    }
                }
            }
            div { class: "dialog-content", {children} }
            if let Some(on_confirm) = on_confirm {
                div { class: "dialog-buttons flex flex-row",
                    button {
                        class: "dialog-buttons-confirm",
                        disabled: confirm_disabled,
                        onclick: move |_| on_confirm.call(()),
                        "好"
                    }
                }
            }
        }
    }
}

#[component]
pub(crate) fn DialogNewSession(
    session_name: Signal<String>,
    participants_ids: Signal<fnv::FnvHashSet<Uuid>>,
    uuid: Uuid,
) -> Element {
    let dialogs_manager = use_context::<DialogsManager>();
    let session_view_model = use_context::<SessionViewModel>();
    let mut sessions = session_view_model.sessions;
    let operator_view_model = use_context::<OperatorViewModel>();
    let operators = operator_view_model.operator_repository;

    let title = use_signal(|| "添加新会话".to_string());

    rsx! {
        Dialog {
            title,
            confirm_disabled: session_name.read().trim().is_empty() || participants_ids.read().is_empty(),
            on_confirm: {
                let mut dialogs_manager = dialogs_manager.clone();
                move |_| {
                    let session_name = session_name.read().trim().to_owned();

                    if session_name.is_empty() || participants_ids.read().is_empty() {
                        return;
                    }

                    sessions
                        .write()
                        .push_session(
                            Session::new(
                                session_name,
                                match participants_ids.read().iter().count() {
                                    1 => {
                                        operators
                                            .read()
                                            .get(*participants_ids.read().iter().next().unwrap())
                                            .unwrap()
                                            .get_avatar_originally()
                                            .clone()
                                    }
                                    _ => Avatar::None,
                                },
                                participants_ids.read().iter().cloned().collect::<Vec<Uuid>>(),
                            ),
                        )
                        .unwrap();
                    dialogs_manager.remove_dialog(uuid);
                    participants_ids.clear();
                }
            },
            uuid,
            div { id: "new-sessions-dialog", class: "flex flex-column",
                InputComponent {
                    id: "session-name-{uuid}",
                    label: "会话名",
                    component_type: InputComponentType::Text,
                    value: Some(session_name()),
                    on_value_change: move |value| {
                        if let InputType::Text(value) = value {
                            session_name.set(value);
                        }
                    },
                }

                fieldset { class: "dialog-groupbox",
                    legend { "参与者" }
                    ParticipantsSelection { participants_ids }
                }

                div { class: "dialog-validation-errors",
                    if session_name.read().trim().is_empty() {
                        p { class: "dialog-validation-error", role: "alert", "请填写会话名" }
                    }
                    if participants_ids.read().is_empty() {
                        p { class: "dialog-validation-error", role: "alert",
                            "请至少选择一名参与者"
                        }
                    }
                }
            }
        }
    }
}

#[component]
pub(crate) fn DialogManageOperators(uuid: Uuid) -> Element {
    let mut dialogs_manager = use_context::<DialogsManager>();
    let operator_view_model = use_context::<OperatorViewModel>();
    let mut operators = operator_view_model.operator_repository;
    let session_view_model = use_context::<SessionViewModel>();
    let sessions = session_view_model.sessions;

    let mut name = use_signal(String::new);
    // 若是空，则为未选择头像
    let mut new_operator_avatar_id = use_signal(String::new);

    let mut edit_selected_operator_id = use_signal(|| None);
    let mut edit_selected_operator_name = use_signal(String::new);

    let title = use_signal(|| "管理干员列表".to_string());

    rsx! {
        Dialog {
            title,
            on_confirm: move |_| dialogs_manager.remove_dialog(uuid),
            uuid,

            div { id: "new-operator-dialog", class: "flex flex-column",
                div { class: "menu",
                    h3 { "添加干员" }

                    // 添加新干员的输入区
                    div { class: "new-operator-form",
                        img {
                            class: "new-operator-avatar",
                            src: if new_operator_avatar_id.is_empty() { Avatar::None.to_asset_operator() } else { Avatar::Preset(new_operator_avatar_id()).to_asset_operator() },
                        }
                        div { class: "new-operator-fields",
                            InputComponent {
                                id: "operator-name-{uuid}",
                                label: "干员名",
                                component_type: InputComponentType::Text,
                                value: Some(name()),
                                on_value_change: move |value| {
                                    if let InputType::Text(value) = value {
                                        name.set(value);
                                    }
                                },
                            }

                            div { class: "new-operator-controls",
                                label { r#for: "avatar-select", "头像" }
                                select {
                                    name: "avatar",
                                    id: "avatar-select",
                                    onchange: move |evt| {
                                        new_operator_avatar_id.set(evt.value());
                                    },
                                    option { value: "", "选择头像" }
                                    for k in assets::CHARACTERS_IDS.iter() {
                                        option { value: k, "{assets::CHARACTERS_NAME[k]}" }
                                    }
                                }
                                button {
                                    class: "dialog-buttons-confirm",
                                    onclick: move |_| {
                                        let trimmed = name.read().trim().to_owned();
                                        if !trimmed.is_empty() {
                                            let avatar_id = new_operator_avatar_id();
                                            // 没选头像时用 Avatar::None，别把空 preset 存进去
                                            let avatar = if avatar_id.is_empty() {
                                                Avatar::None
                                            } else {
                                                Avatar::Preset(avatar_id)
                                            };

                                            operators
                                                .write()
                                                .push_operator(Operator::new(trimmed, avatar))
                                                .unwrap();
                                            name.write().clear();
                                        }
                                    },
                                    "添加"
                                }
                            }
                        }
                    }

                    h3 { "干员列表管理" }

                    // 已有干员列表
                    div { class: "participants",
                        for (id , op) in operators.read().iterator() {
                            div { class: "participant participant-setting flex flex-row",
                                span { class: "flex-1", "{op.name()}" }
                                span { class: "actions-participant-setting",
                                    span {
                                        onclick: {
                                            move |_| {
                                                operators.write().deactivate_operator(id, sessions.into()).unwrap();
                                            }
                                        },
                                        "停用干员"
                                    }
                                    span {
                                        onclick: {
                                            move |_| {
                                                edit_selected_operator_id.set(Some(id));
                                            }
                                        },
                                        "改名"
                                    }
                                }
                                if let Some(selected_id) = edit_selected_operator_id() {
                                    if selected_id == id {
                                        div { class: "edit-operator",
                                            InputComponent {
                                                id: "operator-rename-{uuid}-{id}",
                                                label: "新干员名",
                                                component_type: InputComponentType::Text,
                                                value: Some(edit_selected_operator_name()),
                                                on_value_change: move |value| {
                                                    if let InputType::Text(value) = value {
                                                        edit_selected_operator_name.set(value);
                                                    }
                                                },
                                            }
                                            button {
                                                class: "dialog-button",
                                                onclick: move |_| {
                                                    edit_selected_operator_id.set(None);
                                                },
                                                "取消"
                                            }
                                            button {
                                                class: "dialog-button",
                                                onclick: {
                                                    move |_| {
                                                        // TODO: Unicode 规范化
                                                        edit_selected_operator_id.set(None);
                                                        if edit_selected_operator_name.is_empty() {
                                                            return;
                                                        }
                                                        operators.write().rename(id, edit_selected_operator_name()).unwrap();
                                                        edit_selected_operator_name.clear();
                                                    }
                                                },
                                                "确定"
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
