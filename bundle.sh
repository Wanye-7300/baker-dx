rustup target add wasm32-unknown-unknown
cargo install cargo-binstall
cargo binstall dioxus-cli --force
dx bundle --release --platform web
