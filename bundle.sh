set -euo pipefail

rustup target add wasm32-unknown-unknown
curl -sSL https://dioxus.dev/install.sh | bash
dx bundle --release --platform web
