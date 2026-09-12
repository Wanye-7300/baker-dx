use crate::operator::model::*;
use crate::operator::view_model::OperatorViewModel;
use crate::session::model::*;
use crate::session::view::session::*;
use crate::session::view::session_list::*;
use crate::session::view_model::session_view_model::SessionViewModel;
use crate::shared::assets;
use crate::view_try;

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

/// 判定为「拖动」而非「点击」的位移阈值（像素）。
const DRAG_THRESHOLD: f64 = 3.0;

/// 标题栏高度，与 CSS 中 `.win10-dialog .dialog-title` 保持一致。
const CAPTION_HEIGHT: f64 = 32.0;

/// 窗口被拖出视口时，标题栏至少要留在视口内的宽度与高度，保证窗口还能再次被拖回来。
const CAPTION_VISIBLE_WIDTH: f64 = 120.0;
const CAPTION_VISIBLE_HEIGHT: f64 = 8.0;

/// 窗口拖动状态。
#[derive(Clone, Copy, PartialEq, Debug)]
struct DialogDrag {
    /// 按下时指针相对窗口左上角的偏移。
    grab_x: f64,
    grab_y: f64,
    /// 窗口宽度，用于限制窗口横向拖出视口的距离。
    width: f64,
    /// 按下时指针所在的视口坐标，用于判断是否超过拖动阈值。
    start_x: f64,
    start_y: f64,
    /// 本次按下是否已经构成拖动。
    moved: bool,
}

/// 读取视口尺寸。
fn viewport_size() -> (f64, f64) {
    let Some(window) = web_sys::window() else {
        return (0.0, 0.0);
    };

    let width = window.inner_width().ok().and_then(|value| value.as_f64()).unwrap_or(0.0);
    let height = window.inner_height().ok().and_then(|value| value.as_f64()).unwrap_or(0.0);

    (width, height)
}

/// 读取对话框窗口本体在视口中的矩形：left、top、width、height。
fn dialog_rect(uuid: &Uuid) -> Option<(f64, f64, f64, f64)> {
    let element = web_sys::window()?.document()?.get_element_by_id(&format!("win10-dialog-{uuid}"))?;
    let rect = element.get_bounding_client_rect();

    Some((rect.left(), rect.top(), rect.width(), rect.height()))
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

    let baker_state = use_context::<crate::BakerState>();
    let dialogs = baker_state.dialogs.read();

    let mut with_settings_open = use_signal(|| false);

    let session_name = use_signal(String::new);
    let participants_ids = use_signal(FnvHashSet::default);

    rsx! {
        div { id: "app", class: "flex flex-column",
            div {
                ondoubleclick: move |evt| {
                    evt.stop_propagation();
                    with_settings_open.set(true);
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

        for (_uuid , dialog) in dialogs.iter() {
            {dialog}
        }

        if with_settings_open() {
            crate::settings::components::Settings {
                on_close: move |_| {
                    with_settings_open.set(false);
                },
            }
        }

        if let Some(uuid) = view_try!(
            crate ::shared::utils::get_item_or_default("wallpaper", || None::< Uuid >,)
        )
        {
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
    on_confirm: EventHandler,
    uuid: Uuid,
    /// 关闭方式：传入时交给外部处理（例如设置窗口的开关信号），否则把这个对话框从 dialogs 表里移除
    #[props(default)] on_close: Option<EventHandler>,
    #[props(default)] confirm_disabled: bool,
    children: Element,
) -> Element {
    let dialogs = use_context::<crate::BakerState>().dialogs;

    // 拖动相关状态（每个对话框实例各自一份；未拖动过时位置完全交给 CSS）
    let mut drag = use_signal(|| None::<DialogDrag>);
    let mut position = use_signal(|| None::<(f64, f64)>);
    // 拖动结束后需要吞掉随之而来的 backdrop click，否则「拖到空白处松手」会被当成点击背景而关闭对话框
    let mut swallow_click = use_signal(|| false);

    let close = move || {
        if let Some(handler) = on_close {
            handler.call(());
        } else {
            let mut dialogs = dialogs;
            dialogs.write().remove(&uuid);
        }
    };

    rsx! {
        div {
            class: "backdrop",
            key: "{uuid}",
            // 背景铺满视口，指针在窗口内移动/抬起的事件会冒泡到这里，因此无需 pointer capture
            onpointermove: move |evt| {
                let Some(state) = drag() else {
                    return;
                };

                let point = evt.client_coordinates();
                let (left, top) = clamp_position(
                    point.x - state.grab_x,
                    point.y - state.grab_y,
                    state.width,
                    viewport_size(),
                );

                if position() != Some((left, top)) {
                    position.set(Some((left, top)));
                }

                if !state.moved && (point.x - state.start_x).abs() + (point.y - state.start_y).abs() > DRAG_THRESHOLD {
                    drag.set(Some(DialogDrag { moved: true, ..state }));
                }
            },
            onpointerup: move |_| {
                if drag().is_some_and(|state| state.moved) {
                    swallow_click.set(true);
                }
                drag.set(None);
            },
            onpointercancel: move |_| drag.set(None),
            onpointerleave: move |_| drag.set(None),
            // 在背景上按下时清掉残留状态；点击背景关闭对话框的行为保持不变
            onpointerdown: move |_| {
                drag.set(None);
                swallow_click.set(false);
            },
            onclick: move |_| {
                if swallow_click() {
                    swallow_click.set(false);
                    return;
                }
                close();
            },
            div {
                key: "{uuid.to_string()}",
                id: "win10-dialog-{uuid}",
                class: "dialog win10-dialog flex flex-column",
                style: position()
                    .map(|(left, top)| format!("left: {left}px; top: {top}px; bottom: auto;"))
                    .unwrap_or_default(),
                onclick: move |e| {
                    e.stop_propagation();
                },
                // Windows 10 caption：左侧标题 + 右侧关闭按钮；标题栏本身是拖动把手
                div {
                    class: "dialog-title flex flex-row",
                    onpointerdown: move |evt| {
                        let Some((left, top, width, _height)) = dialog_rect(&uuid) else {
                            return;
                        };

                        let point = evt.client_coordinates();
                        drag.set(Some(DialogDrag {
                            grab_x: point.x - left,
                            grab_y: point.y - top,
                            width,
                            start_x: point.x,
                            start_y: point.y,
                            moved: false,
                        }));

                        // 不要冒泡到背景的 pointerdown，否则刚建立的拖动状态会被清掉
                        evt.stop_propagation();
                    },
                    span { class: "dialog-title-text", "{title}" }
                    button {
                        class: "dialog-title-close",
                        r#type: "button",
                        title: "关闭",
                        aria_label: "关闭",
                        // 从关闭按钮上按下不参与拖动
                        onpointerdown: move |evt| evt.stop_propagation(),
                        onclick: move |_| close(),
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
                div { class: "dialog-buttons flex flex-row",
                    button {
                        class: "dialog-buttons-confirm win10-button",
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
    let mut baker_state = use_context::<crate::BakerState>();
    let session_view_model = use_context::<SessionViewModel>();
    let mut sessions = session_view_model.sessions;
    let operator_view_model = use_context::<OperatorViewModel>();
    let operators = operator_view_model.operator_repository;

    let title = use_signal(|| "添加新会话".to_string());

    rsx! {
        Dialog {
            title,
            confirm_disabled: session_name.read().trim().is_empty() || participants_ids.read().is_empty(),
            on_confirm: move |_| {
                let mut baker_state = use_context::<crate::BakerState>();
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
                baker_state.dialogs.write().remove(&uuid);
                participants_ids.clear();
            },
            uuid,
            div { id: "new-sessions-dialog", class: "flex flex-column",
                input {
                    class: "form-input",
                    placeholder: "会话名",
                    value: session_name,
                    onchange: move |evt| {
                        *session_name.write() = evt.value();
                    },
                }

                // Win32 GroupBox：参与者分组
                fieldset { class: "win10-groupbox",
                    legend { "参与者" }
                    ParticipantsSelection { participants_ids }
                }

                div { class: "win10-actions",
                    button {
                        id: "button-new-operator",
                        class: "win10-button",
                        r#type: "button",
                        onclick: move |_| {
                            let uuid_neo = Uuid::new_v4();
                            baker_state.dialogs.write().insert(uuid_neo, rsx! {
                                DialogManageOperators { uuid: uuid_neo }
                            });
                            baker_state.dialogs.write().remove(&uuid);
                        },
                        "添加新干员"
                    }
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
        Dialog { title, on_confirm: move |_| {}, uuid,

            div { id: "new-operator-dialog", class: "flex flex-column",
                div { class: "menu",
                    h3 { "添加干员" }

                    // 添加新干员的输入区
                    span { class: "flex flex-row",
                        img {
                            class: "new-operator-avatar",
                            src: if new_operator_avatar_id.is_empty() { Avatar::None.to_asset_operator() } else { Avatar::Preset(new_operator_avatar_id()).to_asset_operator() },
                        }
                        div { class: "",
                            label { "干员名" }
                            input {
                                r#type: "text",
                                class: "form-input",
                                placeholder: "",
                                value: name,
                                onchange: move |evt| {
                                    *name.write() = evt.value();
                                },
                            }
                        }

                    }

                    label { r#for: "avatar-select", "头像" }
                    select {
                        name: "avatar",
                        id: "avatar-select",
                        onchange: move |evt| {
                            new_operator_avatar_id.set(evt.value());
                        },
                        option { value: "", "选择头像" }
                        for k in assets::CHARACTERS_AVATARS.keys() {
                            option { value: k, "{assets::CHARACTERS_NAME[k]}" }
                        }
                    }
                    br {}
                    button {
                        class: "dialog-buttons-confirm",
                        onclick: move |_| {
                            let trimmed = name.read().trim().to_owned();
                            if !trimmed.is_empty() {
                                operators
                                    .write()
                                    .push_operator(
                                        Operator::new(trimmed, Avatar::Preset(new_operator_avatar_id())),
                                    )
                                    .unwrap();
                                name.write().clear();
                            }
                        },
                        "添加"
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
                                            input {
                                                r#type: "text",
                                                value: "{edit_selected_operator_name()}",
                                                onchange: move |evt| {
                                                    edit_selected_operator_name.set(evt.value());
                                                },
                                            }
                                            button {
                                                onclick: move |_| {
                                                    edit_selected_operator_id.set(None);
                                                },
                                                "取消"
                                            }
                                            button {
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
