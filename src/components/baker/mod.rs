pub mod capture;
pub mod chat_area;
pub mod context_menu;
pub mod input_bar;
pub mod layout;
pub mod locale;
pub mod modals;
pub mod settings;
pub mod sidebar;
pub mod storage;

use crate::components::baker::storage::v2::AppState;
use dioxus::prelude::*;
pub use layout::Route;

/// 创建一个本地 Signal，镜像 AppState 中的某个字段，并在字段变动时自动同步回 AppState。
pub(super) fn use_synced_field<T, G, S>(
    mut app_state: Signal<AppState>,
    get: G,
    set: S,
) -> Signal<T>
where
    T: Clone + PartialEq + 'static,
    G: Fn(&AppState) -> T + Copy + 'static,
    S: Fn(&mut AppState, T) + Copy + 'static,
{
    let signal = use_signal(move || get(&app_state.read()));
    use_effect(move || {
        let current = signal.read().clone();
        if current != get(&app_state.read()) {
            let mut state = app_state.write();
            set(&mut state, current);
        }
    });
    signal
}

pub(super) fn mime_from_filename(name: &str) -> &'static str {
    match name
        .rsplit('.')
        .next()
        .map(|ext| ext.to_lowercase())
        .as_deref()
    {
        Some("png") => "image/png",
        Some("jpg") | Some("jpeg") => "image/jpeg",
        Some("gif") => "image/gif",
        Some("webp") => "image/webp",
        _ => "application/octet-stream",
    }
}

pub(super) fn data_url_from_bytes(mime: &str, bytes: Vec<u8>) -> String {
    use base64::Engine;
    let encoded = base64::engine::general_purpose::STANDARD.encode(bytes);
    format!("data:{mime};base64,{encoded}")
}

#[deprecated]
#[allow(unused)]
pub(super) fn avif_data_url_from_bytes(bytes: Vec<u8>) -> Option<String> {
    use image::ImageEncoder;
    let image = image::load_from_memory(&bytes).ok()?;
    let rgba = image.to_rgba8();
    let (width, height) = rgba.dimensions();
    let mut out = Vec::new();
    let encoder = image::codecs::avif::AvifEncoder::new(&mut out);
    encoder
        .write_image(&rgba, width, height, image::ColorType::Rgba8.into())
        .ok()?;
    Some(data_url_from_bytes("image/avif", out))
}

pub(super) async fn capture(selector: &str, scale: f64) -> Option<String> {
    let eval = document::eval(
        r#"
            const selector = await dioxus.recv();
            const scaleRaw = Number(await dioxus.recv());
            const scale = Number.isFinite(scaleRaw) && scaleRaw > 0 ? scaleRaw : 1;
            const el = document.querySelector(selector);
            if (!el) return null;
            const result = await snapdom(el, { scale });
            const img = await result.toPng();
            return img?.src ?? null;
        "#,
    );
    eval.send(selector.to_owned()).ok()?;
    eval.send(scale).ok()?;
    let value = eval.await.ok()?;
    value.as_str().map(|src| src.to_string())
}

pub(super) async fn download_image(src: &str, format: &str, filename: &str) -> anyhow::Result<()> {
    let eval = document::eval(
        r#"
        let src = await dioxus.recv();
        let format = (await dioxus.recv())?.toLowerCase().trim();
        let filenameInput = (await dioxus.recv())?.trim();
        let hasExt = filenameInput.includes(".");
        let filename = filenameInput.length === 0
            ? `download.${format || "png"}`
            : hasExt
                ? filenameInput
                : `${filenameInput}.${format || "png"}`;
        const link = document.createElement("a");
        link.href = src;
        link.download = filename;
        link.style.display = "none";
        document.body.appendChild(link);
        link.click();
        link.remove();
        return true;
    "#,
    );

    eval.send(src.to_string())?;
    eval.send(format.to_string())?;
    eval.send(filename.to_string())?;

    eval.await.map_err(|err| anyhow::anyhow!(err.to_string()))?;

    Ok(())
}
