use anyhow::{Context, Result};
use crane_core::models::qwen3_vl::Qwen3VL;
use std::io::Write;
use std::path::{Path, PathBuf};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage:");
        eprintln!("  Image mode:   qwen3_vl_simple <model_path> <img1[,img2,...]> [prompt]");
        eprintln!("  Dataset mode: qwen3_vl_simple <model_path> --entry <index> [--dataset <train.json>]");
        eprintln!();
        eprintln!("Examples:");
        eprintln!("  qwen3_vl_simple ./checkpoints/qwen3_vl_2b_merged ./front.jpg \"Extract MRZ\"");
        eprintln!("  qwen3_vl_simple ./checkpoints/qwen3_vl_2b_merged --entry 0");
        std::process::exit(1);
    }

    let model_path = &args[1];

    // Dataset mode
    if args.len() >= 4 && args[2] == "--entry" {
        let index: usize = args[3].parse().context("--entry requires a number")?;
        let dataset_path = if args.len() >= 6 && args[4] == "--dataset" {
            PathBuf::from(&args[5])
        } else {
            PathBuf::from("train.json")
        };
        return run_dataset_entry(model_path, &dataset_path, index);
    }

    // Image mode: pass user text, model builds prompt internally
    let image_paths: Vec<&str> = args[2].split(',').collect();
    let user_text = if args.len() > 3 {
        args[3..].join(" ")
    } else {
        "Describe this image in detail.".to_string()
    };

    println!("Loading model from: {}", model_path);
    let mut model = Qwen3VL::from_local(model_path, false, false)?;

    println!("Processing {} image(s): {}", image_paths.len(), args[2]);
    println!("User text: {}", user_text);
    println!("---");

    let result = model.recognize_stream(
        &image_paths,
        &user_text,
        512,
        |token| {
            print!("{}", token);
            let _ = std::io::stdout().flush();
        },
    )?;

    println!("\n---");
    println!(
        "Generated {} tokens in {:.2}s ({:.1} tok/s)",
        result.tokens_generated,
        result.duration_secs,
        result.tokens_generated as f32 / result.duration_secs
    );
    Ok(())
}

fn run_dataset_entry(model_path: &str, dataset_path: &Path, index: usize) -> Result<()> {
    let dataset_dir = dataset_path
        .parent()
        .unwrap_or(Path::new("."))
        .to_path_buf();

    println!("Loading dataset: {}", dataset_path.display());
    let raw = std::fs::read_to_string(dataset_path)
        .context(format!("Cannot read {}", dataset_path.display()))?;
    let entries: Vec<serde_json::Value> = serde_json::from_str(&raw)?;

    if index >= entries.len() {
        anyhow::bail!("Index {} out of range (dataset has {} entries)", index, entries.len());
    }

    let entry = &entries[index];
    println!("Entry #{} of {}", index, entries.len());

    let images: Vec<String> = entry["images"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default();

    let image_paths: Vec<PathBuf> = images
        .iter()
        .map(|p| {
            let path = PathBuf::from(p);
            if path.is_absolute() { path } else { dataset_dir.join(p) }
        })
        .collect();

    let conversations = entry["conversations"].as_array();
    let human_text = conversations
        .and_then(|convs| convs.iter().find(|c| c["from"].as_str() == Some("human")))
        .and_then(|c| c["value"].as_str())
        .unwrap_or("");
    let expected_gpt = conversations
        .and_then(|convs| {
            convs.iter().find(|c| {
                let f = c["from"].as_str().unwrap_or("");
                f == "gpt" || f == "assistant"
            })
        })
        .and_then(|c| c["value"].as_str())
        .unwrap_or("");

    println!(
        "Images ({}): {}",
        image_paths.len(),
        image_paths.iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join(", ")
    );
    println!("Prompt: {}...", &human_text[..human_text.len().min(120)]);
    println!();

    // recognize_stream now takes the raw user text and builds the prompt internally
    println!("Loading model from: {}", model_path);
    let mut model = Qwen3VL::from_local(model_path, false, false)?;
    println!("---");
    println!("MODEL OUTPUT:");

    model.recognize_stream(
        &image_paths,
        human_text,
        1024,
        |token| {
            print!("{}", token);
            let _ = std::io::stdout().flush();
        },
    )?;

    println!("\n---");
    println!("EXPECTED OUTPUT:");
    println!("{}", expected_gpt);
    println!("---");
    Ok(())
}
