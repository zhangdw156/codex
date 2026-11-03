//! Fail-closed evaluation boundaries for the Glitter native ARE driver.
//!
//! This is opt-in per process. The private socket is an acknowledged local
//! control channel, never a model transport or an alternative agent loop.
use crate::client_common::Prompt;
use crate::session::session::Session;
use crate::session::step_context::StepContext;
use codex_protocol::error::CodexErr;
use codex_protocol::error::Result;
use codex_protocol::models::ResponseItem;
use codex_tools::ResponsesApiNamespaceTool;
use codex_tools::ToolName;
use codex_tools::ToolSpec;
use serde_json::Value;
use serde_json::json;
use std::collections::HashSet;
use std::sync::OnceLock;
use tokio::io::AsyncBufReadExt;
use tokio::io::AsyncWriteExt;
use tokio::io::BufReader;

pub(crate) fn enabled() -> bool {
    std::env::var_os("GLITTER_CODEX_SOCKET").is_some()
}

fn allowed_names() -> &'static HashSet<String> {
    static NAMES: OnceLock<HashSet<String>> = OnceLock::new();
    NAMES.get_or_init(|| {
        std::env::var("GLITTER_CODEX_TOOLS")
            .ok()
            .and_then(|value| serde_json::from_str(&value).ok())
            .unwrap_or_default()
    })
}

pub(crate) fn allowed(name: &ToolName) -> bool {
    !enabled()
        || (name.namespace.as_deref() == Some("mcp__gaia2") && allowed_names().contains(&name.name))
}

pub(crate) fn restrict_specs(specs: Vec<ToolSpec>) -> Vec<ToolSpec> {
    if !enabled() {
        return specs;
    }
    restrict_specs_to(specs, allowed_names())
}

fn restrict_specs_to(specs: Vec<ToolSpec>, names: &HashSet<String>) -> Vec<ToolSpec> {
    specs
        .into_iter()
        .filter_map(|spec| match spec {
            ToolSpec::Namespace(mut namespace) if namespace.name == "mcp__gaia2" => {
                namespace.tools.retain(|tool| match tool {
                    ResponsesApiNamespaceTool::Function(tool) => {
                        allowed_names().contains(&tool.name)
                    }
                    ResponsesApiNamespaceTool::Custom(_) => false,
                });
                (!namespace.tools.is_empty()).then_some(ToolSpec::Namespace(namespace))
            }
            _ => None,
        })
        .collect()
}

fn failure(error: impl std::fmt::Display) -> CodexErr {
    CodexErr::Fatal(format!("Glitter boundary failed: {error}"))
}

pub(crate) async fn call(value: Value) -> Result<Value> {
    #[cfg(unix)]
    {
        let path = std::env::var("GLITTER_CODEX_SOCKET").map_err(failure)?;
        let mut stream = tokio::net::UnixStream::connect(path)
            .await
            .map_err(failure)?;
        let mut payload = serde_json::to_vec(&value).map_err(failure)?;
        payload.push(b'\n');
        stream.write_all(&payload).await.map_err(failure)?;
        let mut line = String::new();
        BufReader::new(stream)
            .read_line(&mut line)
            .await
            .map_err(failure)?;
        let response: Value = serde_json::from_str(&line).map_err(failure)?;
        if let Some(error) = response.get("error") {
            return Err(failure(error));
        }
        Ok(response)
    }
    #[cfg(not(unix))]
    {
        let _ = value;
        Err(failure("Glitter requires a Unix control socket"))
    }
}

pub(crate) async fn before_request(
    session: &Session,
    step: &StepContext,
    prompt: &mut Prompt,
) -> Result<bool> {
    if !enabled() {
        return Ok(false);
    }
    let prepared = call(json!({"operation": "prepare"})).await?;
    if prepared.get("stop").and_then(Value::as_bool) == Some(true) {
        session.flush_rollout().await.map_err(failure)?;
        return Ok(true);
    }
    let items: Vec<ResponseItem> = serde_json::from_value(
        prepared
            .get("messages")
            .cloned()
            .unwrap_or_else(|| json!([])),
    )
    .map_err(failure)?;
    if !items.is_empty() {
        session
            .record_conversation_items(&step.turn, &step.settings.model_info, &items)
            .await;
        prompt.input.extend(items);
    }
    prompt.parallel_tool_calls = false;
    session.flush_rollout().await.map_err(failure)?;
    let path = session
        .current_rollout_path()
        .await
        .map_err(failure)?
        .ok_or_else(|| failure("a persistent local rollout is required"))?;
    call(
        json!({"operation": "checkpoint", "session_id": session.thread_id.to_string(),
        "session_path": path}),
    )
    .await?;
    Ok(false)
}

pub(crate) async fn turn_ended(session: &Session) -> Result<bool> {
    if !enabled() {
        return Ok(false);
    }
    let status = call(json!({"operation": "status"})).await?;
    let ended = status.get("stop").and_then(Value::as_bool) == Some(true);
    if ended {
        session.flush_rollout().await.map_err(failure)?;
    }
    Ok(ended)
}

#[cfg(test)]
mod tests {
    use super::*;
    use codex_tools::ResponsesApiNamespace;
    use codex_tools::ResponsesApiTool;

    fn tool(name: &str) -> ResponsesApiTool {
        ResponsesApiTool {
            name: name.into(),
            description: String::new(),
            strict: false,
            defer_loading: None,
            output_schema: None,
            parameters: serde_json::from_value(json!({"type": "object", "properties": {}}))
                .unwrap(),
        }
    }

    #[test]
    fn scenario_filter_removes_host_tools_and_unlisted_mcp_tools() {
        let allowed = HashSet::from(["echo".into()]);
        let specs = vec![
            ToolSpec::Function(tool("exec_command")),
            ToolSpec::Namespace(ResponsesApiNamespace {
                name: "other_mcp".into(),
                description: String::new(),
                tools: vec![ResponsesApiNamespaceTool::Function(tool("echo"))],
            }),
            ToolSpec::Namespace(ResponsesApiNamespace {
                name: "mcp__gaia2".into(),
                description: String::new(),
                tools: vec![
                    ResponsesApiNamespaceTool::Function(tool("echo")),
                    ResponsesApiNamespaceTool::Function(tool("hidden")),
                ],
            }),
        ];
        let filtered = restrict_specs_to(specs, &allowed);
        assert_eq!(filtered.len(), 1);
        let ToolSpec::Namespace(namespace) = &filtered[0] else {
            panic!("expected namespace")
        };
        assert_eq!(namespace.name, "mcp__gaia2");
        assert_eq!(namespace.tools.len(), 1);
        assert!(restrict_specs_to(filtered, &HashSet::new()).is_empty());
    }
}
