// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::Serialize;
use serde_json::json;

use crate::state::AppState;

#[derive(Serialize)]
pub(crate) struct VersionResponse {
    version: String,
    rust_frontend_version: &'static str,
}

/// Get engine and Rust frontend version metadata, or 403 unless
/// `VLLM_EXPOSE_DEPLOYMENT_INFO` exposes it, with Python's body.
pub async fn version(State(state): State<Arc<AppState>>) -> Response {
    if !state.api_server_options.expose_deployment_info {
        return (
            StatusCode::FORBIDDEN,
            Json(json!({"error": "Deployment information is not exposed"})),
        )
            .into_response();
    }
    let version = state.llm().info().vllm_version.clone();

    Json(VersionResponse {
        version,
        rust_frontend_version: vllm_build_info::VERSION,
    })
    .into_response()
}
