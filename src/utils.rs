pub(crate) fn get_item_or_default<T: serde::de::DeserializeOwned, U: Fn() -> T>(
    key: &str,
    default: U,
) -> T {
    #[cfg(feature = "web")]
    {
        // TODO: 在隐私模式下 unwrap 可能会 panic
        let local_storage = web_sys::window().unwrap().local_storage().unwrap().unwrap();
        match local_storage.get_item(key).unwrap() {
            Some(result) => match serde_json::from_str(&result) {
                Ok(result) => result,
                Err(_) => default(),
            },
            None => default(),
        }
    }

    #[cfg(not(feature = "web"))]
    {
        match std::fs::read_to_string(std::path::Path::new("config/").join(key)) {
            Ok(result) => match serde_json::from_str(&result) {
                Ok(result) => result,
                Err(_) => default(),
            },
            Err(_) => default(),
        }
    }
}

pub(crate) fn set_item<T: serde::Serialize>(key: &str, item: &T) {
    #[cfg(feature = "web")]
    {
        let local_storage = web_sys::window().unwrap().local_storage().unwrap().unwrap();
        let _ = local_storage.set_item(key, &serde_json::to_string(&item).unwrap());
    }

    #[cfg(not(feature = "web"))]
    {
        use std::io::Write;
        std::fs::create_dir_all("config/");
        let mut file = std::fs::File::create(std::path::Path::new("config/").join(key)).unwrap();
        let _ = file.write_all(&serde_json::to_vec(item).unwrap());
    }
}
