#![allow(dead_code)] // rustc 1.95 ICE: emitting dead_code diagnostics for tools::search panics in the renderer (slice-index in warn_dead_code -> track_diagnostic); suppress emission crate-wide until the pinned toolchain moves past it. Do NOT remove without testing cargo check -p calyx-mcp.
//! MCP interface for agent-facing Calyx operations.
//!
//! The wire stack is split across modules: [`jsonrpc`] decodes inbound requests,
//! [`protocol`] frames responses and MCP descriptors, [`schema`] builds tool
//! input schemas, and [`server`] holds the tool registry and dispatch.

pub mod jsonrpc;
pub mod protocol;
pub mod schema;
pub mod server;
pub mod tools;

pub use jsonrpc::{
    CALYX_MCP_JSONRPC_INVALID, JsonRpcId, JsonRpcRequest, JsonRpcWire, decode_jsonrpc_request,
    decode_jsonrpc_wire,
};
pub use protocol::{
    ContentBlock, JSONRPC_CALYX_ERROR, JSONRPC_INTERNAL_ERROR, JSONRPC_INVALID_PARAMS,
    JSONRPC_METHOD_NOT_FOUND, JsonRpcError, JsonRpcResponse, ToolCallResult, ToolDef,
};
pub use server::{
    CALYX_MCP_TOOL_DUPLICATE, MCP_PROTOCOL_VERSION, McpServer, SERVER_NAME, Tool, ToolError,
    ToolResult,
};

#[cfg(test)]
mod tests {
    #[test]
    fn crate_metadata_is_present() {
        assert_eq!(env!("CARGO_PKG_NAME"), "calyx-mcp");
    }
}
