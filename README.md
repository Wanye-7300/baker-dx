# Baker

基于 Dioxus 0.7 的会话编辑与回放界面，面向聊天场景的 UI 原型与交互演示。
这个项目旨在还原《明日方舟：终末地》中 Baker 的聊天功能，当然还有一点功能没实现的……留作以后再完善。

## 功能概览

- 会话与联系人管理：选择会话、发起新会话、配置干员列表
- 消息编辑能力：发送、编辑、删除、在指定位置插入
- 状态行支持：以独立状态行展示，可编辑、删除、插入
- 回放能力：从指定消息起开始回放
- 主题与资料：会话头样式切换、背景模式设置、用户资料配置
- 本地持久化存储：Web 使用 LocalStorage，非 wasm 目标使用本地 JSON 文件

## TO-DO LIST

- [] 移动端有限宽度的布局支持
- [] 应用配置导出
- [] “任务”消息支持
- [] 会话的离屏渲染并导出
- [] 群组会话时控制哪些干员消息位于会话右方

## 使用的技术

- Rust 2021
- Dioxus 0.7

## 运行

```bash
cargo install dioxus-cli
dx serve --platform web
# ...If you want to run it on desktop platform, you can use the following command:
dx serve --platform desktop
```

## 问题、建议、Pull Request

如果你在使用这个软件时遇到任何 bug 或者对软件有啥建议，请在 Github 上提交 Issue，谢谢！

如果有什么功能你想实现，也欢迎提交 Pull Request，我会尽快合并。

---

_……_

_我是管理员，这个项目的作者请我帮忙带个话：“**请注意，作者在天师考校上课，由于天师考校复课（高中开学），因此作者在工作日可能不能及时回复问题或者查看拉取请求。**”谢谢你的理解！_

## 展示

![展示-Perlica](./readme/perlica.png)
