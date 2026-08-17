pub(crate) fn get_item_or_default<T: serde::de::DeserializeOwned, U: Fn() -> T>(
    key: &str,
    default: U,
) -> anyhow::Result<T> {
    #[cfg(feature = "web")]
    {
        let local_storage = web_sys::window()
            .ok_or_else(|| anyhow::anyhow!("LocalStorage: Failed to get window"))?
            .local_storage()
            .map_err(|err| anyhow::anyhow!("LocalStorage: Failed to get local storage: {err:?}"))?
            .ok_or_else(|| anyhow::anyhow!("LocalStorage: Failed to get storage"))?;
        match local_storage.get_item(key) {
            Ok(Some(result)) => match serde_json::from_str::<T>(&result) {
                Ok(result) => Ok(result),
                Err(_) => Ok(default()),
            },
            Ok(None) => Ok(default()),
            Err(err) => Err(anyhow::anyhow!("LocalStorage: Failed to get item `{key}`: {err:?}")),
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

pub(crate) fn set_item<T: serde::Serialize>(key: &str, item: &T) -> anyhow::Result<()> {
    #[cfg(feature = "web")]
    {
        let local_storage = web_sys::window()
            .ok_or_else(|| anyhow::anyhow!("LocalStorage: Failed to get window"))?
            .local_storage()
            .map_err(|err| anyhow::anyhow!("LocalStorage: Failed to get local storage: {err:?}"))?
            .ok_or_else(|| anyhow::anyhow!("LocalStorage: Failed to get storage"))?;
        let _ = local_storage.set_item(key, &serde_json::to_string(&item).unwrap());
    }

    #[cfg(not(feature = "web"))]
    {
        use std::io::Write;
        std::fs::create_dir_all("config/");
        let mut file = std::fs::File::create(std::path::Path::new("config/").join(key)).unwrap();
        let _ = file.write_all(&serde_json::to_vec(item).unwrap());
    }

    Ok(())
}

#[macro_export]
macro_rules! view_try {
    ($expr:expr) => {
        match $expr {
            Ok(value) => value,
            Err(err) => {
                dioxus::prelude::error!("Error occurred: {err:#}");
                return rsx! { "Error occurred: {err:#}" };
            }
        }
    };
}

#[macro_export]
macro_rules! panic_try {
    ($expr:expr) => {
        match $expr {
            Ok(value) => value,
            Err(err) => {
                dioxus::prelude::error!("Error occurred: {err:#}");
                panic!("Error occurred: {err:#}");
            }
        }
    };
}
