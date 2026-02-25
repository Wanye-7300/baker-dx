use futures_util::StreamExt;
use serde::Serialize;

const DEEPSEEK_API_CHAT_COMPLETIONS_URL: &str = "https://api.deepseek.com/chat/completions";

#[derive(Debug, Clone, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessage>,
    stream: bool,
}

#[derive(Debug, Clone)]
pub struct DeepseekLLMProvider {
    api_key: String,
    client: reqwest::Client,
}

impl DeepseekLLMProvider {
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            client: reqwest::Client::new(),
        }
    }

    pub async fn stream_chat_completions<F>(
        &self,
        messages: Vec<ChatMessage>,
        mut on_delta: F,
    ) -> anyhow::Result<()>
    where
        F: FnMut(String),
    {
        let request = ChatCompletionRequest {
            model: "deepseek-chat".to_string(),
            messages,
            stream: true,
        };
        let response = self
            .client
            .post(DEEPSEEK_API_CHAT_COMPLETIONS_URL)
            .header("Accept", "application/json")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&request)
            .send()
            .await?;

        let mut buffer = String::new();
        let mut stream = response.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            let text = String::from_utf8_lossy(&chunk);
            buffer.push_str(&text);
            while let Some(pos) = buffer.find('\n') {
                let line = buffer[..pos].trim().to_string();
                buffer.replace_range(..=pos, "");
                if let Some(delta) = parse_stream_line(&line)? {
                    on_delta(delta);
                }
            }
        }
        if !buffer.is_empty() {
            if let Some(delta) = parse_stream_line(buffer.trim())? {
                on_delta(delta);
            }
        }
        Ok(())
    }
}

fn parse_stream_line(line: &str) -> anyhow::Result<Option<String>> {
    if !line.starts_with("data:") {
        return Ok(None);
    }
    let data = line.trim_start_matches("data:").trim();
    if data.is_empty() || data == "[DONE]" {
        return Ok(None);
    }
    let value: serde_json::Value = serde_json::from_str(data)?;
    let content = value["choices"][0]["delta"]["content"]
        .as_str()
        .unwrap_or("")
        .to_string();
    if content.is_empty() {
        return Ok(None);
    }
    Ok(Some(content))
}
