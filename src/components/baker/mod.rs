pub mod chat_area;
pub mod input_bar;
pub mod layout;
pub mod modals;
pub mod models;
pub mod settings;
pub mod sidebar;
pub mod storage;

use dioxus::document;
pub use layout::Route;

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

pub(super) fn capture(selector: &str) {
    let selector = selector.to_owned();
    dioxus::core::spawn(async move {
        let eval = document::eval(&format!(
            r#"
            const el = document.querySelector('{}');
            const result = await snapdom(el);

            const img = await result.toPng();
            document.body.appendChild(img);

            await result.download({{ format: 'jpg', filename: 'my-capture' }});
        "#,
            selector
        ));

        let _ = eval.await;
    });
}
