//! Test harness para validar Crane contra el oráculo Python.
//!
//! Encodea las mismas 2 imágenes y 2 queries que el reference Python,
//! escribe los embeddings y la matriz de scores a disco como bytes raw f32.
//!
//! Uso:
//!   embedding_oracle_test <model_path> <test_dir>
//!
//! Lee:
//!   <test_dir>/images/00_white_32.png
//!   <test_dir>/images/01_black_16.png
//! Escribe:
//!   <test_dir>/crane_query0.bin     (Sq, 2560)  f32 row-major
//!   <test_dir>/crane_query1.bin
//!   <test_dir>/crane_image0.bin     (Sp, 2560)
//!   <test_dir>/crane_image1.bin
//!   <test_dir>/crane_scores.bin     (2, 2)
//!   <test_dir>/crane_meta.txt        ← shapes legibles

use anyhow::Result;
use crane_core::models::candle_core;
use crane_core::models::candle_core::DType;
use crane_core::models::colqwen3_emb::ColQwen3Emb;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

fn write_tensor_f32(path: &PathBuf, t: &candle_core::Tensor) -> Result<(usize, usize)> {
    let t = t.to_dtype(DType::F32)?.contiguous()?;
    let dims = t.dims().to_vec();
    let (rows, cols) = if dims.len() == 2 {
        (dims[0], dims[1])
    } else if dims.len() == 1 {
        (1, dims[0])
    } else {
        anyhow::bail!("expected 1D or 2D tensor, got {:?}", dims);
    };
    let flat: Vec<f32> = t.flatten_all()?.to_vec1()?;
    let mut f = File::create(path)?;
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(flat.as_ptr() as *const u8, flat.len() * 4)
    };
    f.write_all(bytes)?;
    Ok((rows, cols))
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: embedding_oracle_test <model_path> <test_dir>");
        std::process::exit(1);
    }
    let model_path = &args[1];
    let test_dir = PathBuf::from(&args[2]);
    let img_dir = test_dir.join("images");

    let queries = [
        "Is attention really all you need?",
        "What is the amount of bananas farmed in Salvador?",
    ];
    let image_paths = [
        img_dir.join("00_white_32.png"),
        img_dir.join("01_black_16.png"),
    ];

    println!("Loading ColQwen3 (bf16=true)...");
    let mut model = ColQwen3Emb::from_local(model_path, false, true)?;

    println!("Encoding queries...");
    let query_embs = model.encode_queries(&queries)?;
    println!("Encoding images...");
    let image_embs: Vec<_> = image_paths.iter().map(|p| {
        model.encode_images(&[p]).map(|mut v| v.remove(0))
    }).collect::<Result<Vec<_>>>()?;

    let mut meta = String::new();
    let (r, c) = write_tensor_f32(&test_dir.join("crane_query0.bin"), &query_embs[0])?;
    meta.push_str(&format!("query0: ({}, {})\n", r, c));
    let (r, c) = write_tensor_f32(&test_dir.join("crane_query1.bin"), &query_embs[1])?;
    meta.push_str(&format!("query1: ({}, {})\n", r, c));
    let (r, c) = write_tensor_f32(&test_dir.join("crane_image0.bin"), &image_embs[0])?;
    meta.push_str(&format!("image0: ({}, {})\n", r, c));
    let (r, c) = write_tensor_f32(&test_dir.join("crane_image1.bin"), &image_embs[1])?;
    meta.push_str(&format!("image1: ({}, {})\n", r, c));

    println!("Computing scores...");
    let scores = ColQwen3Emb::score(&query_embs, &image_embs, 128)?;
    let (r, c) = write_tensor_f32(&test_dir.join("crane_scores.bin"), &scores)?;
    meta.push_str(&format!("scores: ({}, {})\n", r, c));

    let scores_vec: Vec<f32> = scores.flatten_all()?.to_vec1()?;
    meta.push_str(&format!("scores_values: {:?}\n", scores_vec));

    File::create(test_dir.join("crane_meta.txt"))?.write_all(meta.as_bytes())?;
    print!("{}", meta);
    Ok(())
}
