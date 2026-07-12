//! Retrieval eval: run Argus over the same pages/queries ops used and dump the
//! ranked top-k page numbers per query, for side-by-side comparison with ops.
//!
//! Usage: argus_eval <model_path> <eval_dir>
//!   <eval_dir>/queries.json  (ops output: documents[].entities[] {query,k,pages_obtained})
//!   writes <eval_dir>/argus_results.json

use anyhow::{Context, Result};
use crane_core::models::argus_colqwen35::ArgusColqwen35Emb;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: argus_eval <model_path> <eval_dir>");
        std::process::exit(1);
    }
    let model_path = &args[1];
    let eval_dir = PathBuf::from(&args[2]);

    let root: Value = serde_json::from_str(&std::fs::read_to_string(eval_dir.join("queries.json"))?)?;
    let documents = root["documents"].as_array().context("no documents")?;

    println!("Loading Argus-Colqwen3.5 ...");
    let mut model = ArgusColqwen35Emb::from_local(model_path, false, true)?;

    let mut out_docs = Vec::new();
    for (di, doc) in documents.iter().enumerate() {
        let doc_id = doc["document_id"].as_str().unwrap_or("?").to_string();
        let doc_type = doc["document_type"].as_str().unwrap_or("?").to_string();

        // Pages sorted by page number.
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

        println!("[{}/{}] {} ({}, {} pages) encoding...", di + 1, documents.len(), &doc_id[..8], doc_type, pages.len());
        let page_embs = model.encode_images(&page_paths)?;

        let entities = doc["entities"].as_array().context("no entities")?;
        let queries: Vec<&str> = entities.iter().map(|e| e["query"].as_str().unwrap_or("")).collect();
        let query_embs = model.encode_queries(&queries)?;

        // (n_q, n_p) score matrix.
        let scores = ArgusColqwen35Emb::score(&query_embs, &page_embs, 128)?;
        let smat: Vec<Vec<f32>> = scores.to_vec2()?;

        let mut out_ents = Vec::new();
        for (ei, ent) in entities.iter().enumerate() {
            let k = ent["k"].as_u64().unwrap_or(3) as usize;
            let row = &smat[ei];
            let mut idx: Vec<usize> = (0..row.len()).collect();
            idx.sort_by(|&a, &b| row[b].partial_cmp(&row[a]).unwrap_or(std::cmp::Ordering::Equal));
            let topk: Vec<i64> = idx.iter().take(k).map(|&i| page_nums[i]).collect();
            let topk_scores: Vec<f32> = idx.iter().take(k).map(|&i| (row[i] * 1000.0).round() / 1000.0).collect();
            let ops_pages = ent["pages_obtained"].clone();
            out_ents.push(json!({
                "name": ent["name"],
                "query": ent["query"],
                "k": k,
                "ops_pages": ops_pages,
                "argus_pages": topk,
                "argus_scores": topk_scores,
            }));
        }
        out_docs.push(json!({
            "document_id": doc_id,
            "document_type": doc_type,
            "total_pages": pages.len(),
            "entities": out_ents,
        }));
    }

    let out = json!({ "record_id": root["record_id"], "documents": out_docs });
    let out_path = eval_dir.join("argus_results.json");
    std::fs::write(&out_path, serde_json::to_string_pretty(&out)?)?;
    println!("wrote {}", out_path.display());
    Ok(())
}
