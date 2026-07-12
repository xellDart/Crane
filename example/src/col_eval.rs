//! Generic retrieval eval over any ColEmbedder model (ops / argus / colqwen3_5),
//! dispatched by config.json model_type. Runs the same pages/queries the ops
//! run used and dumps ranked top-k per query for side-by-side comparison.
//!
//! Usage: col_eval <model_path> <eval_dir>
//!   <eval_dir>/queries.json  (ops output: documents[].entities[] {query,k,pages_obtained})
//!   writes <eval_dir>/<model_kind>_results.json

use anyhow::{Context, Result};
use crane_core::models::col_embedder::ColEmbedder;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::time::Instant;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: col_eval <model_path> <eval_dir>");
        std::process::exit(1);
    }
    let model_path = &args[1];
    let eval_dir = PathBuf::from(&args[2]);

    let root: Value = serde_json::from_str(&std::fs::read_to_string(eval_dir.join("queries.json"))?)?;
    let documents = root["documents"].as_array().context("no documents")?;

    println!("Loading model from {} ...", model_path);
    let mut model = ColEmbedder::from_local(model_path, false, true)?;
    let kind = model.model_kind().to_string();
    println!("model_kind = {}", kind);

    let mut out_docs = Vec::new();
    let t_all = Instant::now();
    let mut total_pages = 0usize;
    for (di, doc) in documents.iter().enumerate() {
        let doc_id = doc["document_id"].as_str().unwrap_or("?").to_string();
        let doc_type = doc["document_type"].as_str().unwrap_or("?").to_string();

        let mut pages: Vec<(i64, PathBuf)> = doc["pages"]
            .as_array()
            .context("no pages")?
            .iter()
            .map(|p| {
                let n = p["page"].as_i64().unwrap_or(0);
                let f = p["file"].as_str().unwrap_or("");
                (n, eval_dir.join(f))
            })
            .collect();
        pages.sort_by_key(|(n, _)| *n);
        let page_nums: Vec<i64> = pages.iter().map(|(n, _)| *n).collect();
        let page_paths: Vec<&Path> = pages.iter().map(|(_, p)| p.as_path()).collect();

        println!(
            "[{}/{}] {} ({}, {} pages) encoding...",
            di + 1, documents.len(), &doc_id[..doc_id.len().min(8)], doc_type, pages.len()
        );
        let t0 = Instant::now();
        let page_embs = model.encode_images(&page_paths)?;
        total_pages += pages.len();

        let entities = doc["entities"].as_array().context("no entities")?;
        let queries: Vec<&str> = entities.iter().map(|e| e["query"].as_str().unwrap_or("")).collect();
        let query_embs = model.encode_queries(&queries)?;
        let dt = t0.elapsed().as_secs_f64();

        let scores = ColEmbedder::score(&query_embs, &page_embs, 128)?;
        let smat: Vec<Vec<f32>> = scores.to_vec2()?;

        let mut out_ents = Vec::new();
        for (ei, ent) in entities.iter().enumerate() {
            let k = ent["k"].as_u64().unwrap_or(3) as usize;
            let row = &smat[ei];
            let mut idx: Vec<usize> = (0..row.len()).collect();
            idx.sort_by(|&a, &b| row[b].partial_cmp(&row[a]).unwrap_or(std::cmp::Ordering::Equal));
            let topk: Vec<i64> = idx.iter().take(k).map(|&i| page_nums[i]).collect();
            let topk_scores: Vec<f32> =
                idx.iter().take(k).map(|&i| (row[i] * 1000.0).round() / 1000.0).collect();
            out_ents.push(json!({
                "name": ent["name"],
                "query": ent["query"],
                "k": k,
                "ops_pages": ent["pages_obtained"].clone(),
                "pred_pages": topk,
                "pred_scores": topk_scores,
            }));
        }
        println!("     encoded {} pages + {} queries in {:.2}s", pages.len(), queries.len(), dt);
        out_docs.push(json!({
            "document_id": doc_id,
            "document_type": doc_type,
            "total_pages": pages.len(),
            "entities": out_ents,
        }));
    }

    let out = json!({ "model_kind": kind, "record_id": root["record_id"], "documents": out_docs });
    let out_path = eval_dir.join(format!("{}_results.json", kind));
    std::fs::write(&out_path, serde_json::to_string_pretty(&out)?)?;
    println!(
        "wrote {} | {} docs, {} pages in {:.1}s",
        out_path.display(), out_docs.len(), total_pages, t_all.elapsed().as_secs_f64()
    );
    Ok(())
}
