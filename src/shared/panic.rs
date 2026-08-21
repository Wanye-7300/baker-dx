use std::panic;

use web_sys::js_sys;
use web_sys::wasm_bindgen::JsValue;

pub(crate) fn install_panic_hook() {
    panic::set_hook(Box::new(move |info| {
        let error = info.payload_as_str().unwrap_or("<UNKNOWN>");
        let location = info.location().map_or(String::new(), |l| {
            format!(" 在于 {}:{},{}", l.file(), l.line(), l.column())
        });

        let stack = js_sys::Error::new("");
        let stack = js_sys::Reflect::get(&stack, &JsValue::from_str("stack"))
            .ok()
            .and_then(|val| val.as_string())
            .unwrap_or_else(|| "<Unsupported>".to_string());

        let Some(window) = web_sys::window() else {
            return;
        };

        let Some(document) = window.document() else {
            return;
        };

        let Some(body) = document.body() else {
            return;
        };

        body.set_inner_html(&format!(
            "<main class=\"panic\">
                程序 BAKER-DX panicked{location}：<br />
                {error}<br /><br />
                - 如果需要恢复，按F5或者浏览器的刷新键刷新页面。<br />
                - 如果要报告问题，请附上这个页面和你刚才做的操作。<br /><br />
                堆栈追踪：<br />{}
            </main>",
            stack.replace("\n", "<br />")
        ));
    }));
}
