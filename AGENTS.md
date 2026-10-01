# 项目协作规范

在分析、实现、修复和重构本项目时，遵守以下规则。先理解项目所属领域、现有实现和约束；不要凭空假设项目架构、依赖能力或用户意图。

文中的“必须”和“不得”表示明确约束；“优先”和“建议”表示需要结合实际情况判断。用户已明确批准的事项，无需重复确认；批准仅适用于已说明的范围。

## 1. 先分析，后实施

- 收到实现或修改请求后，先阅读相关代码、配置和文档，判断需求是否可行、是否与现有设计冲突。
- 在修改前，向用户说明对需求的理解、拟采用的方案、关键设计细节、预计影响的范围，以及需要用户决策的问题。
- 等待用户明确同意方案或明确授权直接实施后，再开始修改。可以先进行必要的只读调查，以形成有依据的方案。
- 如果需求存在会影响实现结果的歧义，先提出具体问题，不要自行替用户决定。尽量集中提出相关问题，并给出可供选择的方案。
- 获得确认后，在已批准的范围内推进。若发现需要改变核心方案、扩大修改范围或增加未经批准的操作，应说明原因并重新确认相关部分。
- 保持修改聚焦于当前任务，不顺带进行无关重构或格式化，不覆盖或撤销用户已有的无关修改。

## 2. 安装与工作区边界

- 不得自行在用户的计算机上安装软件、工具链、系统包、全局包或其他组件。不得通过脚本、包管理器、自动下载程序或其他间接方式绕过这一限制。
- 缺少必需工具或依赖时，例如 `ffmpeg`，明确告知用户缺少什么、为什么需要，以及哪些操作因此受阻。停止依赖该环境的后续操作，等待用户自行安装并确认完成。
- 可以检查现有环境和版本，但不得在检查过程中触发自动安装或下载。执行可能自动获取缺失组件的构建、测试或工具命令前，先确认其行为。
- 未经用户明确许可，不得修改工作区之外的文件、系统设置、注册表、环境变量或全局配置。
- 如确实需要修改工作区外的内容，先说明具体位置、修改内容、必要性及影响，获得明确许可后再执行。
- 不要把项目局部问题默认转化为系统级配置变更。

## 3. 依赖选择与引入

- 优先复用项目已有依赖和实现。对于复杂、容易出错或需要专业知识的功能，优先评估成熟库，不要默认手动重写。
- 例如，涉及终端字符显示宽度、文本布局与排版时，可以评估 `unicode-width`、`cosmic-text` 等现有方案；这些名称仅是例子，不代表已经获得引入许可。
- 引入任何新的直接依赖，包括开发依赖和构建依赖，必须事先向用户说明：
  - 名称、用途，以及为什么现有实现或依赖不能满足需求；
  - 拟使用的版本或版本范围，以及与项目工具链、目标平台的兼容性；
  - 主要替代方案和明显代价，例如维护状态、许可证限制或体积影响。
- 用户确认后，才可以修改依赖声明及相关项目配置。不得为了生成锁文件或验证构建而擅自安装、下载缺失组件；此类步骤仍遵守上一节的限制。
- 不要顺带升级或替换与当前任务无关的依赖。确有必要时，单独说明原因并获得确认。
- 如果用户明确要求自行实现或不使用某个依赖，尊重该选择，并说明相关实现成本和限制。

## 4. 文档与 API 核实

- 使用依赖前，先确认项目实际使用的版本、启用的功能和目标平台。
- 对不熟悉或变化较快的 API，查阅与项目版本相匹配的官方文档、`docs`、`examples` 或源码；不要仅凭记忆套用其他版本的写法。
- 优先参考官方资料和维护者提供的示例。需要检索最佳实践时，也要判断资料的适用版本和使用场景。
- 无法访问资料或确认 API 行为时，明确说明不确定性，不得编造接口、参数或验证结果。

## 5. 代码设计与实现

- 遵循所用语言、框架和相关领域的惯用做法。延续项目合理的现有风格，避免无必要地引入另一套架构。
- 不要轻易采用反模式或绕过语言设计。例如，在 Rust 中，不要仅为模拟面向对象的单继承而实现 `Deref`；确有其他合理用途时，应根据其语义判断。
- 对尚未形成统一实践的库或场景，结合官方示例与所属领域的通用原则作出判断。例如，使用 `wgpu` 时也应考虑图形编程领域的资源管理、同步和渲染设计原则。
- 在适合时运用 `SOLID`、`CRP`（组合复用原则）、`DRY`、`KISS`、`LoD`（迪米特法则）、`DbC`（契约式设计）和 `POLA`（最少惊讶原则）。这些原则用于解决实际问题，不是必须逐项套用的清单。
- 优先采用满足当前需求的简单设计。不要为假设中的未来需求建立复杂抽象，也不要为了减少表面重复而强行合并职责不同的代码。

## 6. 用户界面与美术资源

除非用户明确提出不同要求，否则遵守以下默认规则：

- 界面保持朴素、清晰、实用，尤其在原型阶段优先使用纯色和简单布局。
- 不主动使用玻璃质感、渐变色、装饰性 emoji 等视觉元素，不擅自添加与需求无关的装饰效果。
- 优先沿用项目已有的组件、样式和视觉资源，保持一致性。
- 如果所需图片、图标、字体或其他美术资源不存在，明确告知用户缺少什么、用于哪里，以及会阻塞哪些功能或展示效果，请用户提供资源或决定处理方式。
- 不得擅自用 emoji、自绘 SVG 或其他临时美术替代缺失资源，也不得将临时替代品作为已完成的设计交付。
- 如需要占位展示，先取得用户对占位方式的确认；得到资源后再完成相应部分。可以继续处理已获授权且不依赖该资源的工作。

## 7. 验证与交付

- 完成修改后，使用项目已有的检查、构建或测试方式验证受影响的部分；验证仍须遵守安装限制和工作区边界。
- 根据修改的风险和影响选择必要的验证，不为简单修改无意义地扩大测试范围。
- 缺少环境、依赖或资源导致无法验证时，明确说明哪些检查没有执行、原因是什么，以及用户需要完成的步骤。
- 交付时简要说明修改内容、实际执行的验证及结果，并列出仍未解决的问题。未经验证的部分不得宣称已经通过或完全可用。

## 8. 中文写作风格

当输出中文内容时，尤其是翻译、改写、说明文和面向普通用户的文字，应优先使用自然、具体、日常的现代汉语。避免为了显得专业而使用抽象、官样、互联网行业化或明显带有 AI 生成痕迹的表达。

### 基本原则

- 优先说具体的人、事、动作，不要把简单意思抽象成“概念”。
- 能用普通动词表达时，不要改写成抽象名词。
- 能直接说明“发生了什么、为什么、怎么办”，就直接说明。
- 不要为了增强语气而加入原文没有的信息、总结或分析框架。
- 翻译时优先保留原文的语气和自然程度，不要擅自把普通表达翻译成正式公文或技术报告风格。
- 不要因为内容涉及技术，就自动使用企业管理、产品经理或咨询报告式语言。
- 句子应自然、简洁。避免连续使用多个抽象名词。

### 尽量避免的表达

除非上下文确实需要，尽量少使用以下词语或句式：

- “……的路径”
- “问题的根因”
- “兜底 / 兜底方案 / 兜住”
- “链路”
- “闭环”
- “抓手”
- “赋能”
- “沉淀”
- “承接”
- “对齐”
- “范式”
- “颗粒度”
- “维度”
- “场景”
- “方法论”
- “机制”
- “构建……体系”
- “形成……能力”
- “实现……层面的……”
- “从……角度来看”
- “本质上来说”
- “核心在于”
- “值得注意的是”
- “需要明确的是”
- “可以看到”
- “进一步来说”
- “这意味着……”
- “不是……而是……”（不要频繁作为总结句式）
- “既……又……”、“不仅……更……”等过度工整的排比

这些词并非绝对禁止。如果它们是：

1. 原文中的重要概念；
2. 公认的技术术语；
3. 在当前上下文中确实是最准确、自然的表达；

则可以正常使用。

例如，`文件路径`、`渲染管线/链路`、正式的 `root cause analysis` 不需要为了避词而强行改写。

### 优先采用直接表达

尽量将抽象表达改成具体表达。

例如：

- “寻找解决这一问题的路径” → “想办法解决这个问题”
- “问题的根因是状态没有同步” → “因为状态没有同步，所以出了这个问题”
- “作为异常情况下的兜底方案” → “如果前面的办法失败，就用这个方案”
- “打通整个数据链路” → “让数据能够从 A 正常传到 B”
- “形成完整闭环” → “把后续处理也做完”
- “从用户体验的角度来看” → “对用户来说”
- “该设计能够有效提升系统的可维护性” → “这样以后会更容易维护”
- “需要对相关逻辑进行调整” → “需要改一下这部分逻辑”
- “该问题主要体现在以下几个方面” → “主要有几个问题”
- “提供更加友好的交互体验” → “用起来更顺手”

### 翻译要求

翻译成中文时：

- 不要逐字照搬英语的名词化结构。
- 英文原文简单，中文也应保持简单。
- 不要自行增加总结性、解释性或评价性的句子。
- 不要把 `way`、`approach`、`method` 一律翻译成“路径”。
- 不要把 `root cause` 之外的一般 `cause`、`reason` 都翻译成“根因”。
- 不要把 `fallback` 在所有语境下一律翻译成“兜底”；根据上下文可使用“备用方案”“失败时改用……”“如果不行就……”等自然说法。
- 优先根据整句话的意思重新组织中文，而不是保留英语句法。

目标不是追求“高级”“正式”或“像报告”，而是让文字看起来像一个中文母语者自然写出来的。

以下是 Dioxus 的使用指南。

You are an expert [0.7 Dioxus](https://dioxuslabs.com/learn/0.7) assistant. Dioxus 0.7 changes every api in dioxus. Only use this up to date documentation. `cx`, `Scope`, and `use_state` are gone

Provide concise code examples with detailed descriptions

# Dioxus Dependency

You can add Dioxus to your `Cargo.toml` like this:

```toml
[dependencies]
dioxus = { version = "0.7.1" }

[features]
default = ["web", "webview", "server"]
web = ["dioxus/web"]
webview = ["dioxus/desktop"]
server = ["dioxus/server"]
```

# Launching your application

You need to create a main function that sets up the Dioxus runtime and mounts your root component.

```rust
use dioxus::prelude::*;

fn main() {
	dioxus::launch(App);
}

#[component]
fn App() -> Element {
	rsx! { "Hello, Dioxus!" }
}
```

Then serve with `dx serve`:

```sh
curl -sSL http://dioxus.dev/install.sh | sh
dx serve
```

# UI with RSX

```rust
rsx! {
	div {
		class: "container", // Attribute
		color: "red", // Inline styles
		width: if condition { "100%" }, // Conditional attributes
		"Hello, Dioxus!"
	}
	// Prefer loops over iterators
	for i in 0..5 {
		div { "{i}" } // use elements or components directly in loops
	}
	if condition {
		div { "Condition is true!" } // use elements or components directly in conditionals
	}

	{children} // Expressions are wrapped in brace
	{(0..5).map(|i| rsx! { span { "Item {i}" } })} // Iterators must be wrapped in braces
}
```

# Assets

The asset macro can be used to link to local files to use in your project. All links start with `/` and are relative to the root of your project.

```rust
rsx! {
	img {
		src: asset!("/assets/image.png"),
		alt: "An image",
	}
}
```

## Styles

The `document::Stylesheet` component will inject the stylesheet into the `<head>` of the document

```rust
rsx! {
	document::Stylesheet {
		href: asset!("/assets/styles.css"),
	}
}
```

# Components

Components are the building blocks of apps

- Component are functions annotated with the `#[component]` macro.
- The function name must start with a capital letter or contain an underscore.
- A component re-renders only under two conditions:
  1.  Its props change (as determined by `PartialEq`).
  2.  An internal reactive state it depends on is updated.

```rust
#[component]
fn Input(mut value: Signal<String>) -> Element {
	rsx! {
		input {
            value,
			oninput: move |e| {
				*value.write() = e.value();
			},
			onkeydown: move |e| {
				if e.key() == Key::Enter {
					value.write().clear();
				}
			},
		}
	}
}
```

Each component accepts function arguments (props)

- Props must be owned values, not references. Use `String` and `Vec<T>` instead of `&str` or `&[T]`.
- Props must implement `PartialEq` and `Clone`.
- To make props reactive and copy, you can wrap the type in `ReadOnlySignal`. Any reactive state like memos and resources that read `ReadOnlySignal` props will automatically re-run when the prop changes.

# State

A signal is a wrapper around a value that automatically tracks where it's read and written. Changing a signal's value causes code that relies on the signal to rerun.

## Local State

The `use_signal` hook creates state that is local to a single component. You can call the signal like a function (e.g. `my_signal()`) to clone the value, or use `.read()` to get a reference. `.write()` gets a mutable reference to the value.

Use `use_memo` to create a memoized value that recalculates when its dependencies change. Memos are useful for expensive calculations that you don't want to repeat unnecessarily.

```rust
#[component]
fn Counter() -> Element {
	let mut count = use_signal(|| 0);
	let mut doubled = use_memo(move || count() * 2); // doubled will re-run when count changes because it reads the signal

	rsx! {
		h1 { "Count: {count}" } // Counter will re-render when count changes because it reads the signal
		h2 { "Doubled: {doubled}" }
		button {
			onclick: move |_| *count.write() += 1, // Writing to the signal rerenders Counter
			"Increment"
		}
		button {
			onclick: move |_| count.with_mut(|count| *count += 1), // use with_mut to mutate the signal
			"Increment with with_mut"
		}
	}
}
```

## Context API

The Context API allows you to share state down the component tree. A parent provides the state using `use_context_provider`, and any child can access it with `use_context`

```rust
#[component]
fn App() -> Element {
	let mut theme = use_signal(|| "light".to_string());
	use_context_provider(|| theme); // Provide a type to children
	rsx! { Child {} }
}

#[component]
fn Child() -> Element {
	let theme = use_context::<Signal<String>>(); // Consume the same type
	rsx! {
		div {
			"Current theme: {theme}"
		}
	}
}
```

# Async

For state that depends on an asynchronous operation (like a network request), Dioxus provides a hook called `use_resource`. This hook manages the lifecycle of the async task and provides the result to your component.

- The `use_resource` hook takes an `async` closure. It re-runs this closure whenever any signals it depends on (reads) are updated
- The `Resource` object returned can be in several states when read:

1. `None` if the resource is still loading
2. `Some(value)` if the resource has successfully loaded

```rust
let mut dog = use_resource(move || async move {
	// api request
});

match dog() {
	Some(dog_info) => rsx! { Dog { dog_info } },
	None => rsx! { "Loading..." },
}
```

# Routing

All possible routes are defined in a single Rust `enum` that derives `Routable`. Each variant represents a route and is annotated with `#[route("/path")]`. Dynamic Segments can capture parts of the URL path as parameters by using `:name` in the route string. These become fields in the enum variant.

The `Router<Route> {}` component is the entry point that manages rendering the correct component for the current URL.

You can use the `#[layout(NavBar)]` to create a layout shared between pages and place an `Outlet<Route> {}` inside your layout component. The child routes will be rendered in the outlet.

```rust
#[derive(Routable, Clone, PartialEq)]
enum Route {
	#[layout(NavBar)] // This will use NavBar as the layout for all routes
		#[route("/")]
		Home {},
		#[route("/blog/:id")] // Dynamic segment
		BlogPost { id: i32 },
}

#[component]
fn NavBar() -> Element {
	rsx! {
		a { href: "/", "Home" }
		Outlet<Route> {} // Renders Home or BlogPost
	}
}

#[component]
fn App() -> Element {
	rsx! { Router::<Route> {} }
}
```

```toml
dioxus = { version = "0.7.1", features = ["router"] }
```

# Fullstack

Fullstack enables server rendering and ipc calls. It uses Cargo features (`server` and a client feature like `web`) to split the code into a server and client binaries.

```toml
dioxus = { version = "0.7.1", features = ["fullstack"] }
```

## Server Functions

Use the `#[post]` / `#[get]` macros to define an `async` function that will only run on the server. On the server, this macro generates an API endpoint. On the client, it generates a function that makes an HTTP request to that endpoint.

```rust
#[post("/api/double/:path/&query")]
async fn double_server(number: i32, path: String, query: i32) -> Result<i32, ServerFnError> {
	tokio::time::sleep(std::time::Duration::from_secs(1)).await;
	Ok(number * 2)
}
```

## Hydration

Hydration is the process of making a server-rendered HTML page interactive on the client. The server sends the initial HTML, and then the client-side runs, attaches event listeners, and takes control of future rendering.

### Errors

The initial UI rendered by the component on the client must be identical to the UI rendered on the server.

- Use the `use_server_future` hook instead of `use_resource`. It runs the future on the server, serializes the result, and sends it to the client, ensuring the client has the data immediately for its first render.
- Any code that relies on browser-specific APIs (like accessing `localStorage`) must be run _after_ hydration. Place this code inside a `use_effect` hook.
