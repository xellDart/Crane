//! Dump one image's multi-vector embedding to a flat f32 file, so the two GDN
//! scans (recurrent vs chunked, selected via CRANE_GDN_RECURRENT) can be
//! compared for numerical parity.
//!
//! Usage: gdn_parity <model_path> <image_path> <out.bin>

use anyhow::Result;
use crane_core::models::colqwen3_5::ColQwen3_5Emb;
use std::io::Write;
use std::path::Path;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let model_path = &args[1];
    let image_path = &args[2];
    let out = &args[3];

    let mut model = ColQwen3_5Emb::from_local(model_path, false, true)?;
    let embs = model.encode_images(&[Path::new(image_path)])?;
    let e = &embs[0]; // (seq, 320)
    let (s, d) = e.dims2()?;
    let flat: Vec<f32> = e.flatten_all()?.to_dtype(crane_core::models::DType::F32)?.to_vec1()?;
    let mut f = std::fs::File::create(out)?;
    for x in &flat {
        f.write_all(&x.to_le_bytes())?;
    }
    println!("wrote {} values ({}x{}) to {}", flat.len(), s, d, out);
    Ok(())
}
