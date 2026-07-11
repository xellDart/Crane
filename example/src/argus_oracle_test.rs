//! Parity harness: Rust Argus-Colqwen3.5 vs the Python oracle.
//!
//! Encodes every image in <test_dir>/images/ (sorted) and every line in
//! <test_dir>/queries.txt, writing raw little-endian f32 (row-major):
//!   <test_dir>/crane_argus_image{i}.bin   (Sp, 1024)
//!   <test_dir>/crane_argus_query{i}.bin   (Sq, 1024)
//!   <test_dir>/crane_argus_scores.bin     (n_q, n_p)
//!   <test_dir>/crane_argus_meta.txt
//!
//! Usage: argus_oracle_test <model_path> <test_dir>

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

    // Sorted image list.
    let mut image_paths: Vec<PathBuf> = std::fs::read_dir(test_dir.join("images"))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            matches!(
                p.extension().and_then(|s| s.to_str()).map(|s| s.to_ascii_lowercase()),
                Some(ref e) if e == "jpg" || e == "jpeg" || e == "png"
            )
        })
        .collect();
    image_paths.sort();

    let queries_raw = std::fs::read_to_string(test_dir.join("queries.txt"))?;
    let queries: Vec<&str> = queries_raw.lines().map(|l| l.trim()).filter(|l| !l.is_empty()).collect();

    println!("Loading Argus-Colqwen3.5 (bf16=true)... {} images, {} queries", image_paths.len(), queries.len());
    let mut model = ArgusColqwen35Emb::from_local(model_path, false, true)?;

    println!("Encoding queries...");
    let query_embs = model.encode_queries(&queries)?;
    println!("Encoding images...");
    let image_embs: Vec<_> = image_paths
        .iter()
        .map(|p| model.encode_images(&[p]).map(|mut v| v.remove(0)))
        .collect::<Result<Vec<_>>>()?;

    let mut meta = String::new();
    for (i, q) in query_embs.iter().enumerate() {
        let (r, c) = write_tensor_f32(&test_dir.join(format!("crane_argus_query{i}.bin")), q)?;
        meta.push_str(&format!("query{i}: ({r}, {c})\n"));
    }
    for (i, im) in image_embs.iter().enumerate() {
        let (r, c) = write_tensor_f32(&test_dir.join(format!("crane_argus_image{i}.bin")), im)?;
        meta.push_str(&format!("image{i}: ({r}, {c})\n"));
    }

    println!("Computing scores...");
    let scores = ArgusColqwen35Emb::score(&query_embs, &image_embs, 128)?;
    let (r, c) = write_tensor_f32(&test_dir.join("crane_argus_scores.bin"), &scores)?;
    meta.push_str(&format!("scores: ({r}, {c})\n"));
    let scores_vec: Vec<f32> = scores.flatten_all()?.to_vec1()?;
    meta.push_str(&format!("scores_values: {:?}\n", scores_vec));

    File::create(test_dir.join("crane_argus_meta.txt"))?.write_all(meta.as_bytes())?;
    print!("{}", meta);
    Ok(())
}
