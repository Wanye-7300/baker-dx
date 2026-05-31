use dioxus::prelude::*;

const MENU_SURFACE_CLASS: &str =
    "fixed z-[100] border border-black/10 shadow-xl py-1 overflow-hidden text-black";
const MENU_ITEM_CLASS: &str =
    "px-4 py-2 cursor-pointer text-black text-sm transition-colors hover:bg-black/10 select-none";
const MENU_ITEM_STYLE: &str =
    "white-space: normal; overflow-wrap: anywhere; word-break: break-word; line-height: 1.25;";
const MENU_SURFACE_STYLE: &str = "background: rgba(255, 255, 255, 0.78); backdrop-filter: blur(16px) saturate(180%); -webkit-backdrop-filter: blur(16px) saturate(180%); border-radius: 1px; color: #000; box-shadow: 0 12px 32px rgba(0, 0, 0, 0.18);";

fn context_menu_style(x: i32, y: i32, width: i32, height: i32) -> String {
    format!(
        "left: clamp(8px, {x}px, calc(100vw - {width}px - 8px)); top: clamp(8px, {y}px, calc(100vh - {height}px - 8px)); width: {width}px; {MENU_SURFACE_STYLE}"
    )
}

#[component]
pub(crate) fn ContextMenu(x: i32, y: i32, width: i32, height: i32, children: Element) -> Element {
    rsx! {
        div {
            class: MENU_SURFACE_CLASS,
            style: "{context_menu_style(x, y, width, height)}",
            onclick: |e| e.stop_propagation(),
            {children}
        }
    }
}

#[component]
pub(crate) fn ContextMenuItem(
    on_select: EventHandler<MouseEvent>,
    children: Element,
    extra_class: Option<String>,
) -> Element {
    let class = extra_class
        .filter(|class| !class.is_empty())
        .map(|class| format!("{MENU_ITEM_CLASS} {class}"))
        .unwrap_or_else(|| MENU_ITEM_CLASS.to_string());

    rsx! {
        div {
            class,
            style: MENU_ITEM_STYLE,
            onclick: move |evt| on_select.call(evt),
            {children}
        }
    }
}
