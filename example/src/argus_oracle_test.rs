//! Parity harness: validate the Rust Argus-Colqwen3.5 path against the Python
//! oracle (modeling_argus.py via transformers 5.x).
//!
//! Encodes the same images + queries as `argus_oracle.py` and writes the
//! embeddings and score matrix to disk as raw f32.
//!
//! Usage:
//!   argus_oracle_test <model_path> <test_dir>
//!
//! Reads:  <test_dir>/images/00_white_32.png, 01_black_16.png
//! Writes: <test_dir>/crane_argus_query{0,1}.bin  (Sq, 1024) f32 row-major
//!         <test_dir>/crane_argus_image{0,1}.bin  (Sp, 1024)
//!         <test_dir>/crane_argus_scores.bin      (2, 2)
//!         <test_dir>/crane_argus_meta.txt

use anyhow::Result;
use crane_core::models::argus_colqwen35::ArgusColqwen35Emb;
use crane_core::models::candle_core;
use crane_core::models::candle_core::DType;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

fn write_tensor_f32(path: &PathBuf, t: &candle_core::Tensor) -> Result<(usize, usize)> {
    let t = t.to_dtype(DType::F32)?.contiguous()?;
    let dims = t.dims().to_vec();
    let (rows, cols) = match dims.len() {
        2 => (dims[0], dims[1]),
        1 => (1, dims[0]),
        _ => anyhow::bail!("expected 1D or 2D tensor, got {:?}", dims),
    };
    let flat: Vec<f32> = t.flatten_all()?.to_vec1()?;
    let mut f = File::create(path)?;
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(flat.as_ptr() as *const u8, flat.len() * 4) };
    f.write_all(bytes)?;
    Ok((rows, cols))
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: argus_oracle_test <model_path> <test_dir>");
        std::process::exit(1);
    }
    let model_path = &args[1];
    let test_dir = PathBuf::from(&args[2]);
    let img_dir = test_dir.join("images");

    let queries = [
        "Is attention really all you need?",
        "What is the amount of bananas farmed in Salvador?",
    ];
    let image_paths = [img_dir.join("00_white_32.png"), img_dir.join("01_black_16.png")];

    println!("Loading Argus-Colqwen3.5 (bf16=true)...");
    let mut model = ArgusColqwen35Emb::from_local(model_path, false, true)?;

    println!("Encoding queries...");
    let query_embs = model.encode_queries(&queries)?;
    println!("Encoding images...");
    let image_embs: Vec<_> = image_paths
        .iter()
        .map(|p| model.encode_images(&[p]).map(|mut v| v.remove(0)))
        .collect::<Result<Vec<_>>>()?;

    let mut meta = String::new();
    let (r, c) = write_tensor_f32(&test_dir.join("crane_argus_query0.bin"), &query_embs[0])?;
    meta.push_str(&format!("query0: ({}, {})\n", r, c));
    let (r, c) = write_tensor_f32(&test_dir.join("crane_argus_query1.bin"), &query_embs[1])?;
    meta.push_str(&format!("query1: ({}, {})\n", r, c));
    let (r, c) = write_tensor_f32(&test_dir.join("crane_argus_image0.bin"), &image_embs[0])?;
    meta.push_str(&format!("image0: ({}, {})\n", r, c));
    let (r, c) = write_tensor_f32(&test_dir.join("crane_argus_image1.bin"), &image_embs[1])?;
    meta.push_str(&format!("image1: ({}, {})\n", r, c));

    println!("Computing scores...");
    let scores = ArgusColqwen35Emb::score(&query_embs, &image_embs, 128)?;
    let (r, c) = write_tensor_f32(&test_dir.join("crane_argus_scores.bin"), &scores)?;
    meta.push_str(&format!("scores: ({}, {})\n", r, c));
    let scores_vec: Vec<f32> = scores.flatten_all()?.to_vec1()?;
    meta.push_str(&format!("scores_values: {:?}\n", scores_vec));

    File::create(test_dir.join("crane_argus_meta.txt"))?.write_all(meta.as_bytes())?;
    print!("{}", meta);
    Ok(())
}
