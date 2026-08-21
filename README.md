# Baker

《明日方舟：终末地》二创制作工具

> [!WARNING]
> 这个分支正在进行项目重构，目前还处在早期开发中。
>
> 目前仅支持 Web 平台。

## 贡献

本项目随时欢迎你的贡献（包括代码、文档和反馈）！

作者完全支持你使用 AI 撰写代码，但是请在提交 PR 之前自行审查代码质量。

## MSRV

最低支持 Rust 版本（MSRV）为 `1.91.0`。

## 运行

```bash
rustup target add wasm32-unknown-unknown

# 如果尚未安装 cargo-binstall
cargo install cargo-binstall

cargo binstall dioxus-cli --force
dx serve --platform web
```
