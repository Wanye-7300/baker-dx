rustup target add wasm32-unknown-unknown

cargo install dioxus-cli --locked

dx bundle --release --platform web
