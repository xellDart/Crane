//! VLM (Vision-Language Model) handlers for PaddleOCR-VL.
//!
//! These handlers bypass the text-only engine and use PaddleOcrVL directly
//! for image+text inference. The model processes images and generates OCR
//! results as streaming or non-streaming text.

use std::sync::Arc;

use axum::{
    http::StatusCode,
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Json, Response,
    },
};

use crane_core::models::paddleocr_vl::OcrTask;

use futures::future::join_all;

use crate::openai_api::*;
use crate::sglang_api::*;
use crate::{make_error, now_epoch, AppState};

// ─────────────────────────────────────────────────────────────
//  VLM Request Channel Structure
// ─────────────────────────────────────────────────────────────

pub enum VlmRequest {
    /// Non-streaming PaddleOCR request
    Recognize {
        img_path: std::path::PathBuf,
        task: OcrTask,
        max_tokens: usize,
        tx: tokio::sync::oneshot::Sender<Result<String, String>>,
    },
    /// Streaming PaddleOCR request
    RecognizeStream {
        img_path: std::path::PathBuf,
        task: OcrTask,
        max_tokens: usize,
        token_tx: tokio::sync::mpsc::UnboundedSender<String>,
        done_tx: tokio::sync::oneshot::Sender<Result<(), String>>,
    },
    /// Non-streaming Qwen3-VL request
    Qwen3VlRecognize {
        img_paths: Vec<std::path::PathBuf>,
        prompt: String,
        max_tokens: usize,
        tx: tokio::sync::oneshot::Sender<Result<String, String>>,
    },
    /// Streaming Qwen3-VL request
    Qwen3VlRecognizeStream {
        img_paths: Vec<std::path::PathBuf>,
        prompt: String,
        max_tokens: usize,
        token_tx: tokio::sync::mpsc::UnboundedSender<String>,
        done_tx: tokio::sync::oneshot::Sender<Result<(), String>>,
    },
}

// ─────────────────────────────────────────────────────────────
//  Image downloading
// ─────────────────────────────────────────────────────────────

/// Resolve an image URL to a local file path.
/// Supports:
///   - data:image/jpeg;base64,...  → decode base64 to temp file (offloaded to blocking thread)
///   - http(s)://...               → downloads to temp file (async)
async fn download_image(url: &str) -> Result<(tempfile::TempDir, std::path::PathBuf), String> {
    // Handle base64 data URIs: offload CPU-bound decode to blocking thread pool
    if url.starts_with("data:") {
        let url_owned = url.to_string();
        return tokio::task::spawn_blocking(move || decode_base64_image(&url_owned))
            .await
            .map_err(|e| format!("Base64 decode task failed: {e}"))?;
    }

    let resp = reqwest::get(url)
        .await
        .map_err(|e| format!("Failed to download image from '{}': {e}", url))?;

    if !resp.status().is_success() {
        return Err(format!(
            "Image download failed (HTTP {}): {}",
            resp.status(),
            url
        ));
    }

    let content_type = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
        .unwrap_or_default();

    let bytes = resp
        .bytes()
        .await
        .map_err(|e| format!("Failed to read image bytes: {e}"))?;

    // Write to temp file on blocking thread pool
    let url_owned = url.to_string();
    tokio::task::spawn_blocking(move || {
        let dir = tempfile::TempDir::new()
            .map_err(|e| format!("Failed to create temp dir: {e}"))?;

        let ext = detect_image_ext_from_content_type(&content_type, &url_owned);
        let img_path = dir.path().join(format!("image.{ext}"));
        std::fs::write(&img_path, &bytes)
            .map_err(|e| format!("Failed to write image to temp file: {e}"))?;
        Ok((dir, img_path))
    })
    .await
    .map_err(|e| format!("Image write task failed: {e}"))?
}

/// Decode a base64 data URI to a temp file (runs on blocking thread).
fn decode_base64_image(url: &str) -> Result<(tempfile::TempDir, std::path::PathBuf), String> {
    use base64::Engine;

    let dir = tempfile::TempDir::new()
        .map_err(|e| format!("Failed to create temp dir: {e}"))?;

    let parts: Vec<&str> = url.splitn(2, ',').collect();
    if parts.len() != 2 {
        return Err("Invalid data URI: missing comma separator".into());
    }
    let header = parts[0];
    let b64_data = parts[1];

    let ext = if header.contains("image/png") {
        "png"
    } else if header.contains("image/webp") {
        "webp"
    } else {
        "jpg"
    };

    let bytes = base64::engine::general_purpose::STANDARD
        .decode(b64_data)
        .map_err(|e| format!("Failed to decode base64 image: {e}"))?;

    let dest = dir.path().join(format!("image.{ext}"));
    std::fs::write(&dest, &bytes)
        .map_err(|e| format!("Failed to write decoded image: {e}"))?;
    Ok((dir, dest))
}

/// Determine image file extension from content-type header or URL.
fn detect_image_ext_from_content_type<'a>(content_type: &str, url: &str) -> &'a str {
    if content_type.contains("image/png") {
        "png"
    } else if content_type.contains("image/webp") {
        "webp"
    } else if content_type.contains("image/jpeg") || content_type.contains("image/jpg") {
        "jpg"
    } else {
        let url_lower = url.to_lowercase();
        if url_lower.contains(".png") {
            "png"
        } else if url_lower.contains(".webp") {
            "webp"
        } else {
            "jpg"
        }
    }
}

/// Determine the OCR task from the text prompt.
fn detect_ocr_task(text: &str) -> OcrTask {
    let text_lower = text.to_lowercase();
    if text_lower.contains("table") {
        OcrTask::Table
    } else if text_lower.contains("formula") {
        OcrTask::Formula
    } else if text_lower.contains("chart") {
        OcrTask::Chart
    } else {
        OcrTask::Ocr
    }
}

// ─────────────────────────────────────────────────────────────
//  Chat Completions (VLM)
// ─────────────────────────────────────────────────────────────

/// VLM-aware chat completions handler.
///
/// Extracts image URLs and text from multimodal messages, downloads
/// images, and runs inference. For Qwen3-VL with engine, routes via
/// `submit_vlm()`. For PaddleOCR-VL, uses the VLM thread.
pub async fn vlm_chat_completions(
    state: Arc<AppState>,
    req: ChatCompletionRequest,
) -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
    // Qwen3-VL with engine: route via submit_vlm for multi-request concurrency.
    if state.is_qwen3_vl && state.engine.is_some() {
        return vlm_chat_via_engine(state, req).await;
    }

    let vlm_tx = state.vlm_tx.as_ref().ok_or_else(|| {
        make_error(StatusCode::INTERNAL_SERVER_ERROR, "VLM model not loaded")
    })?;

    // Extract image URLs and text from messages.
    let mut image_urls = Vec::new();
    let mut text_prompt = String::new();

    for msg in &req.messages {
        if msg.role == "user" {
            let urls = msg.image_urls();
            image_urls.extend(urls);
            let text = msg.text_content();
            if !text.is_empty() {
                text_prompt = text;
            }
        }
    }

    if image_urls.is_empty() {
        return Err(make_error(
            StatusCode::BAD_REQUEST,
            "No image_url found in messages. VLM requires at least one image.",
        ));
    }

    // Download all images concurrently.
    let download_futures: Vec<_> = image_urls.iter()
        .map(|url| download_image(url))
        .collect();
    let download_results = join_all(download_futures).await;
    let mut temp_dirs = Vec::new();
    let mut img_paths = Vec::new();
    for result in download_results {
        let (td, ip) = result.map_err(|e| make_error(StatusCode::BAD_REQUEST, &e))?;
        temp_dirs.push(td);
        img_paths.push(ip);
    }

    let max_tokens = req.max_tokens;
    let request_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    let is_qwen3_vl = state.is_qwen3_vl;

    if req.stream {
        // Streaming mode
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();
        let (done_tx, _done_rx) = tokio::sync::oneshot::channel::<Result<(), String>>();

        let send_result = if is_qwen3_vl {
            vlm_tx.send(VlmRequest::Qwen3VlRecognizeStream {
                img_paths,
                prompt: text_prompt.clone(),
                max_tokens,
                token_tx: tx,
                done_tx,
            })
        } else {
            let task = detect_ocr_task(&text_prompt);
            vlm_tx.send(VlmRequest::RecognizeStream {
                img_path: img_paths.into_iter().next().unwrap(),
                task,
                max_tokens,
                token_tx: tx,
                done_tx,
            })
        };
        if send_result.is_err() {
            return Err(make_error(StatusCode::INTERNAL_SERVER_ERROR, "VLM engine thread crashed"));
        }

        let model_name = state.model_name.clone();
        let created = now_epoch();

        let stream = async_stream::stream! {
            // Role announcement chunk.
            let first_chunk = ChatCompletionChunk {
                id: request_id.clone(),
                object: "chat.completion.chunk".into(),
                created,
                model: model_name.clone(),
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: ChunkDelta {
                        role: Some("assistant".into()),
                        content: None,
                    },
                    finish_reason: None,
                }],
                usage: None,
            };
            yield Ok::<_, std::convert::Infallible>(Event::default().json_data(&first_chunk).unwrap());

            // Stream tokens.
            while let Some(text) = rx.recv().await {
                let chunk = ChatCompletionChunk {
                    id: request_id.clone(),
                    object: "chat.completion.chunk".into(),
                    created,
                    model: model_name.clone(),
                    choices: vec![ChunkChoice {
                        index: 0,
                        delta: ChunkDelta {
                            role: None,
                            content: Some(text),
                        },
                        finish_reason: None,
                    }],
                    usage: None,
                };
                yield Ok(Event::default().json_data(&chunk).unwrap());
            }

            // Finish chunk.
            let finish_chunk = ChatCompletionChunk {
                id: request_id.clone(),
                object: "chat.completion.chunk".into(),
                created,
                model: model_name.clone(),
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: ChunkDelta {
                        role: None,
                        content: None,
                    },
                    finish_reason: Some("stop".into()),
                }],
                usage: None,
            };
            yield Ok(Event::default().json_data(&finish_chunk).unwrap());
            yield Ok(Event::default().data("[DONE]"));
        };

        Ok(Sse::new(stream)
            .keep_alive(KeepAlive::default())
            .into_response())
    } else {
        // Non-streaming mode
        let (tx, rx) = tokio::sync::oneshot::channel();
        let send_result = if is_qwen3_vl {
            vlm_tx.send(VlmRequest::Qwen3VlRecognize {
                img_paths,
                prompt: text_prompt,
                max_tokens,
                tx,
            })
        } else {
            let task = detect_ocr_task(&text_prompt);
            vlm_tx.send(VlmRequest::Recognize {
                img_path: img_paths.into_iter().next().unwrap(),
                task,
                max_tokens,
                tx,
            })
        };
        if send_result.is_err() {
            return Err(make_error(StatusCode::INTERNAL_SERVER_ERROR, "VLM engine thread crashed"));
        }

        let result = rx.await
            .map_err(|e| make_error(StatusCode::INTERNAL_SERVER_ERROR, &format!("VLM task dropped: {e}")))?
            .map_err(|e| make_error(StatusCode::INTERNAL_SERVER_ERROR, &format!("VLM inference failed: {e}")))?;

        let response = ChatCompletionResponse {
            id: request_id,
            object: "chat.completion".into(),
            created: now_epoch(),
            model: state.model_name.clone(),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".into(),
                    content: ChatMessageContent::Text(result),
                },
                finish_reason: Some("stop".into()),
            }],
            usage: Usage {
                prompt_tokens: 0,
                completion_tokens: 0,
                total_tokens: 0,
            },
        };
        Ok(Json(response).into_response())
    }
}

// ─────────────────────────────────────────────────────────────
//  /generate (VLM)
// ─────────────────────────────────────────────────────────────

/// VLM-aware generate handler for SGLang-style `/generate`.
pub async fn vlm_generate(
    state: Arc<AppState>,
    req: GenerateRequest,
) -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
    let vlm_tx = state.vlm_tx.as_ref().ok_or_else(|| {
        make_error(StatusCode::INTERNAL_SERVER_ERROR, "VLM model not loaded")
    })?;

    let image_url = req.image_url.as_deref().ok_or_else(|| {
        make_error(
            StatusCode::BAD_REQUEST,
            "PaddleOCR-VL requires 'image_url' in the generate request",
        )
    })?;

    // Download image.
    let (_temp_dir, img_path) = download_image(image_url)
        .await
        .map_err(|e| make_error(StatusCode::BAD_REQUEST, &e))?;

    let text_prompt = req.text.as_deref().unwrap_or("OCR:");
    let task = detect_ocr_task(text_prompt);
    let max_tokens = req.sampling_params.max_new_tokens;
    let request_id = req
        .rid
        .unwrap_or_else(|| format!("gen-{}", uuid::Uuid::new_v4()));

    if req.stream {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();
        let (done_tx, _done_rx) = tokio::sync::oneshot::channel::<Result<(), String>>();

        if vlm_tx.send(VlmRequest::RecognizeStream {
            img_path,
            task,
            max_tokens,
            token_tx: tx,
            done_tx,
        }).is_err() {
            return Err(make_error(StatusCode::INTERNAL_SERVER_ERROR, "VLM engine thread crashed"));
        }

        let rid = request_id.clone();
        let stream = async_stream::stream! {
            while let Some(text) = rx.recv().await {
                let chunk = GenerateStreamChunk {
                    text,
                    meta_info: None,
                };
                yield Ok::<_, std::convert::Infallible>(Event::default().json_data(&chunk).unwrap());
            }

            // Final chunk with meta.
            let final_chunk = GenerateStreamChunk {
                text: String::new(),
                meta_info: Some(GenerateMetaInfo {
                    id: rid,
                    prompt_tokens: 0,
                    completion_tokens: 0,
                    finish_reason: "stop".into(),
                }),
            };
            yield Ok(Event::default().json_data(&final_chunk).unwrap());
            yield Ok(Event::default().data("[DONE]"));
        };

        Ok(Sse::new(stream)
            .keep_alive(KeepAlive::default())
            .into_response())
    } else {
        let (tx, rx) = tokio::sync::oneshot::channel();
        if vlm_tx.send(VlmRequest::Recognize {
            img_path,
            task,
            max_tokens,
            tx,
        }).is_err() {
            return Err(make_error(StatusCode::INTERNAL_SERVER_ERROR, "VLM engine thread crashed"));
        }

        let result = rx.await
            .map_err(|e| make_error(StatusCode::INTERNAL_SERVER_ERROR, &format!("VLM task dropped: {e}")))?
            .map_err(|e| make_error(StatusCode::INTERNAL_SERVER_ERROR, &format!("VLM inference failed: {e}")))?;

        let response = GenerateResponse {
            text: result,
            meta_info: GenerateMetaInfo {
                id: request_id,
                prompt_tokens: 0,
                completion_tokens: 0,
                finish_reason: "stop".into(),
            },
        };

        Ok(Json(response).into_response())
    }
}

// ─────────────────────────────────────────────────────────────
//  Qwen3-VL via Engine (multi-request concurrency)
// ─────────────────────────────────────────────────────────────

/// Download image URL to raw bytes (no temp files).
async fn download_image_bytes(url: &str) -> Result<Vec<u8>, String> {
    use base64::Engine;

    if url.starts_with("data:") {
        let url_owned = url.to_string();
        return tokio::task::spawn_blocking(move || {
            let parts: Vec<&str> = url_owned.splitn(2, ',').collect();
            if parts.len() != 2 {
                return Err("Invalid data URI: missing comma separator".into());
            }
            base64::engine::general_purpose::STANDARD
                .decode(parts[1])
                .map_err(|e| format!("Failed to decode base64 image: {e}"))
        })
        .await
        .map_err(|e| format!("Base64 decode task failed: {e}"))?;
    }

    let resp = reqwest::get(url)
        .await
        .map_err(|e| format!("Failed to download image from '{}': {e}", url))?;

    if !resp.status().is_success() {
        return Err(format!("Image download failed (HTTP {}): {}", resp.status(), url));
    }

    resp.bytes()
        .await
        .map(|b| b.to_vec())
        .map_err(|e| format!("Failed to read image bytes: {e}"))
}

/// Qwen3-VL chat completions routed through the inference engine.
async fn vlm_chat_via_engine(
    state: Arc<AppState>,
    req: ChatCompletionRequest,
) -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
    use crate::engine::EngineResponse;

    let engine = state.engine.as_ref().ok_or_else(|| {
        make_error(StatusCode::INTERNAL_SERVER_ERROR, "Engine not loaded")
    })?;

    // Extract image URLs and text from messages.
    let mut image_urls = Vec::new();
    let mut text_prompt = String::new();

    for msg in &req.messages {
        if msg.role == "user" {
            image_urls.extend(msg.image_urls());
            let text = msg.text_content();
            if !text.is_empty() {
                text_prompt = text;
            }
        }
    }

    if image_urls.is_empty() {
        return Err(make_error(
            StatusCode::BAD_REQUEST,
            "No image_url found in messages. VLM requires at least one image.",
        ));
    }

    // Download all images to bytes concurrently.
    let download_futures: Vec<_> = image_urls.iter()
        .map(|url| download_image_bytes(url))
        .collect();
    let download_results = join_all(download_futures).await;
    let mut images_bytes = Vec::new();
    for result in download_results {
        let bytes = result.map_err(|e| make_error(StatusCode::BAD_REQUEST, &e))?;
        images_bytes.push(bytes);
    }

    let max_tokens = req.max_tokens;
    let request_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    let eos_token_id = state.eos_token_id.clone();

    // Submit VLM request to engine.
    let mut rx = engine.submit_vlm(
        request_id.clone(),
        images_bytes,
        text_prompt,
        max_tokens,
        req.temperature,
        req.top_p,
        req.top_k,
        req.repetition_penalty.unwrap_or(1.0),
        eos_token_id,
    ).map_err(|e| make_error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()))?;

    if req.stream {
        let model_name = state.model_name.clone();
        let created = now_epoch();

        let stream = async_stream::stream! {
            // Role announcement chunk.
            let first_chunk = ChatCompletionChunk {
                id: request_id.clone(),
                object: "chat.completion.chunk".into(),
                created,
                model: model_name.clone(),
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: ChunkDelta {
                        role: Some("assistant".into()),
                        content: None,
                    },
                    finish_reason: None,
                }],
                usage: None,
            };
            yield Ok::<_, std::convert::Infallible>(Event::default().json_data(&first_chunk).unwrap());

            while let Some(resp) = rx.recv().await {
                match resp {
                    EngineResponse::Token { text, .. } => {
                        let chunk = ChatCompletionChunk {
                            id: request_id.clone(),
                            object: "chat.completion.chunk".into(),
                            created,
                            model: model_name.clone(),
                            choices: vec![ChunkChoice {
                                index: 0,
                                delta: ChunkDelta {
                                    role: None,
                                    content: Some(text),
                                },
                                finish_reason: None,
                            }],
                            usage: None,
                        };
                        yield Ok(Event::default().json_data(&chunk).unwrap());
                    }
                    EngineResponse::Finished { finish_reason, prompt_tokens, completion_tokens, .. } => {
                        let finish_chunk = ChatCompletionChunk {
                            id: request_id.clone(),
                            object: "chat.completion.chunk".into(),
                            created,
                            model: model_name.clone(),
                            choices: vec![ChunkChoice {
                                index: 0,
                                delta: ChunkDelta {
                                    role: None,
                                    content: None,
                                },
                                finish_reason: Some(finish_reason),
                            }],
                            usage: Some(Usage {
                                prompt_tokens,
                                completion_tokens,
                                total_tokens: prompt_tokens + completion_tokens,
                            }),
                        };
                        yield Ok(Event::default().json_data(&finish_chunk).unwrap());
                        yield Ok(Event::default().data("[DONE]"));
                        break;
                    }
                    EngineResponse::Error(msg) => {
                        let err_chunk = ChatCompletionChunk {
                            id: request_id.clone(),
                            object: "chat.completion.chunk".into(),
                            created,
                            model: model_name.clone(),
                            choices: vec![ChunkChoice {
                                index: 0,
                                delta: ChunkDelta {
                                    role: None,
                                    content: Some(format!("[Error: {}]", msg)),
                                },
                                finish_reason: Some("error".into()),
                            }],
                            usage: None,
                        };
                        yield Ok(Event::default().json_data(&err_chunk).unwrap());
                        yield Ok(Event::default().data("[DONE]"));
                        break;
                    }
                }
            }
        };

        Ok(Sse::new(stream)
            .keep_alive(KeepAlive::default())
            .into_response())
    } else {
        // Non-streaming: collect all tokens until Finished.
        let mut full_text = String::new();
        let mut prompt_tokens = 0;
        let mut completion_tokens = 0;
        let mut finish_reason = "stop".to_string();

        while let Some(resp) = rx.recv().await {
            match resp {
                EngineResponse::Token { text, .. } => {
                    full_text.push_str(&text);
                }
                EngineResponse::Finished { full_text: ft, prompt_tokens: pt, completion_tokens: ct, finish_reason: fr } => {
                    full_text = ft;
                    prompt_tokens = pt;
                    completion_tokens = ct;
                    finish_reason = fr;
                    break;
                }
                EngineResponse::Error(msg) => {
                    return Err(make_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        &format!("VLM inference failed: {msg}"),
                    ));
                }
            }
        }

        let response = ChatCompletionResponse {
            id: request_id,
            object: "chat.completion".into(),
            created: now_epoch(),
            model: state.model_name.clone(),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".into(),
                    content: ChatMessageContent::Text(full_text),
                },
                finish_reason: Some(finish_reason),
            }],
            usage: Usage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
            },
        };
        Ok(Json(response).into_response())
    }
}
