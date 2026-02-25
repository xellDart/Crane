use anyhow::{Context, Result};
use crane_core::models::colqwen3_emb::ColQwen3Emb;
use std::path::PathBuf;
use std::time::Instant;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("Usage: embedding_simple <model_path> <images_dir> <query> [--top-k N] [--bf16]");
        eprintln!();
        eprintln!("Example:");
        eprintln!("  embedding_simple ./checkpoints/colqwen3 ./images \"tabla de accionistas\" --top-k 4 --bf16");
        std::process::exit(1);
    }

    let model_path = &args[1];
    let images_dir = &args[2];
    let query = &args[3];
    let use_bf16 = args.iter().any(|a| a == "--bf16");
    let top_k: usize = args
        .windows(2)
        .find(|w| w[0] == "--top-k")
        .and_then(|w| w[1].parse().ok())
        .unwrap_or(4);

    // Collect image paths
    let image_paths = collect_images(images_dir)?;
    if image_paths.is_empty() {
        anyhow::bail!("No images found in {}", images_dir);
    }
    println!("Found {} images in {}", image_paths.len(), images_dir);

    // Load model
    println!("Loading ColQwen3 model from: {} (bf16={})", model_path, use_bf16);
    let t0 = Instant::now();
    let mut model = ColQwen3Emb::from_local(model_path, false, use_bf16)?;
    println!("Model loaded in {:.2}s", t0.elapsed().as_secs_f64());

    // Encode images one by one (to handle variable sizes and avoid OOM)
    println!("\nEncoding {} images...", image_paths.len());
    let t0 = Instant::now();
    let mut image_embeddings = Vec::new();
    for (i, path) in image_paths.iter().enumerate() {
        let embs = model.encode_images(&[path])?;
        image_embeddings.extend(embs);
        if (i + 1) % 10 == 0 || i + 1 == image_paths.len() {
            println!("  [{}/{}] encoded", i + 1, image_paths.len());
        }
    }
    let encode_images_time = t0.elapsed().as_secs_f64();
    println!(
        "Images encoded in {:.2}s ({:.2}s/image)",
        encode_images_time,
        encode_images_time / image_paths.len() as f64
    );

    // Encode query
    println!("\nEncoding query: \"{}\"", query);
    let t0 = Instant::now();
    let query_embeddings = model.encode_queries(&[query.as_str()])?;
    println!("Query encoded in {:.2}s", t0.elapsed().as_secs_f64());

    // Score
    println!("\nComputing MaxSim scores...");
    let t0 = Instant::now();
    let scores = ColQwen3Emb::score(&query_embeddings, &image_embeddings, 128)?;
    let scores_vec: Vec<f32> = scores.squeeze(0)?.to_vec1()?;
    println!("Scoring done in {:.4}s", t0.elapsed().as_secs_f64());

    // Rank and display top-k
    let mut indexed: Vec<(usize, f32)> = scores_vec.into_iter().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let k = top_k.min(image_paths.len());
    println!("\n{}", "=".repeat(60));
    println!("Top {} results for: \"{}\"", k, query);
    println!("{}", "=".repeat(60));
    for (rank, (idx, score)) in indexed.iter().take(k).enumerate() {
        let name = image_paths[*idx]
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        println!("{}. {} (score: {:.4})", rank + 1, name, score);
        println!("   {}", image_paths[*idx].display());
    }

    Ok(())
}

fn collect_images(dir: &str) -> Result<Vec<PathBuf>> {
    let extensions = ["jpg", "jpeg", "png", "webp", "bmp"];
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
        .context(format!("Cannot read directory: {}", dir))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .and_then(|ext| ext.to_str())
                .is_some_and(|ext| extensions.contains(&ext.to_lowercase().as_str()))
        })
        .collect();
    paths.sort();
    Ok(paths)
}
