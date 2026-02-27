#!/usr/bin/env python3
"""
Full sequential vs batch comparison for Qwen3-VL.

1. Start server with --max-concurrent 1 → send entries one at a time → save as baseline
2. Start server with --max-concurrent 4 → send entries concurrently → save as batch
3. Compare outputs character-by-character

Usage:
  python scripts/test_seq_vs_batch.py --entries 0,1,2,3,4 --dataset train.json
"""

import argparse
import base64
import json
import mimetypes
import os
import subprocess
import sys
import time
import concurrent.futures

try:
    import requests
except ImportError:
    print("pip install requests")
    sys.exit(1)


URL = "http://localhost:8080"
MODEL_PATH = "checkpoints/qwen3_vl_2b_merged"
BINARY = "./target/release/crane-oai"


def image_to_data_uri(img_path: str) -> str:
    mime, _ = mimetypes.guess_type(img_path)
    if mime is None:
        mime = "image/jpeg"
    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def load_dataset_entry(dataset_path: str, index: int):
    with open(dataset_path) as f:
        entries = json.load(f)
    entry = entries[index]
    dataset_dir = os.path.dirname(os.path.abspath(dataset_path))
    images = entry.get("images", [])
    image_paths = [p if os.path.isabs(p) else os.path.join(dataset_dir, p) for p in images]
    convs = entry.get("conversations", [])
    human_text = ""
    for c in convs:
        if c.get("from") == "human":
            human_text = c.get("value", "")
            break
    return image_paths, human_text


def build_request(image_paths, prompt, max_tokens):
    content = []
    for img_path in image_paths:
        if os.path.isfile(img_path):
            content.append({"type": "image_url", "image_url": {"url": image_to_data_uri(img_path)}})
    content.append({"type": "text", "text": prompt})
    return {
        "model": "qwen3-vl",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0.01,
        "stream": False,
    }


def send_request(payload):
    resp = requests.post(f"{URL}/v1/chat/completions", json=payload, timeout=300)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def wait_for_server(timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = requests.get(f"{URL}/v1/stats", timeout=2)
            if r.ok:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def start_server(max_concurrent, decode_tokens=8):
    proc = subprocess.Popen(
        [BINARY,
         "--model-path", MODEL_PATH,
         "--model-type", "qwen3_vl",
         "--max-concurrent", str(max_concurrent),
         "--decode-tokens-per-seq", str(decode_tokens),
         "--port", "8080"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if not wait_for_server():
        proc.kill()
        raise RuntimeError("Server failed to start")
    return proc


def stop_server(proc):
    proc.kill()
    proc.wait()
    time.sleep(2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entries", required=True, help="0,1,2 or 0-4")
    parser.add_argument("--dataset", default="/home/miguel/Documentos/unsloth/train.json")
    parser.add_argument("--max-tokens", type=int, default=2048)
    args = parser.parse_args()

    if "-" in args.entries and not args.entries.startswith("-"):
        start, end = args.entries.split("-", 1)
        indices = list(range(int(start), int(end) + 1))
    elif "," in args.entries:
        indices = [int(x.strip()) for x in args.entries.split(",")]
    else:
        indices = [int(args.entries)]

    # Prepare payloads
    payloads = {}
    for idx in indices:
        image_paths, prompt = load_dataset_entry(args.dataset, idx)
        payloads[idx] = build_request(image_paths, prompt, args.max_tokens)
    print(f"Prepared {len(indices)} entries: {indices}")

    # ── Phase 1: Sequential ──
    print(f"\n{'='*60}")
    print("  PHASE 1: Sequential decode (max-concurrent=1)")
    print(f"{'='*60}")
    proc = start_server(max_concurrent=1)
    try:
        seq_results = {}
        for idx in indices:
            t0 = time.time()
            text = send_request(payloads[idx])
            elapsed = time.time() - t0
            seq_results[idx] = text
            preview = text[:60].replace('\n', ' ')
            print(f"  Entry {idx}: {elapsed:.2f}s, {len(text)} chars — {preview}...")
    finally:
        stop_server(proc)

    # ── Phase 2: Batch ──
    print(f"\n{'='*60}")
    print("  PHASE 2: Batch decode (max-concurrent=4)")
    print(f"{'='*60}")
    proc = start_server(max_concurrent=4)
    try:
        batch_results = {}
        t0 = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(indices)) as executor:
            futures = {executor.submit(send_request, payloads[idx]): idx for idx in indices}
            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                text = future.result()
                elapsed = time.time() - t0
                batch_results[idx] = text
                preview = text[:60].replace('\n', ' ')
                print(f"  Entry {idx}: {elapsed:.2f}s, {len(text)} chars — {preview}...")
        total = time.time() - t0
        print(f"  Total batch time: {total:.2f}s")
    finally:
        stop_server(proc)

    # ── Compare ──
    print(f"\n{'='*60}")
    print("  COMPARISON: Sequential vs Batch")
    print(f"{'='*60}")
    all_match = True
    for idx in indices:
        s = seq_results.get(idx, "")
        b = batch_results.get(idx, "")
        if s == b:
            print(f"  Entry {idx}: IDENTICAL ({len(s)} chars)")
        else:
            all_match = False
            print(f"  Entry {idx}: DIFFER!")
            print(f"    Sequential: {len(s)} chars")
            print(f"    Batch:      {len(b)} chars")
            # Find first diff
            for pos in range(min(len(s), len(b))):
                if s[pos] != b[pos]:
                    ctx_s = s[max(0,pos-20):pos+20]
                    ctx_b = b[max(0,pos-20):pos+20]
                    print(f"    First diff at char {pos}:")
                    print(f"      Seq:   ...{ctx_s}...")
                    print(f"      Batch: ...{ctx_b}...")
                    break

    if all_match:
        print(f"\n  ALL {len(indices)} ENTRIES ARE BIT-IDENTICAL!")
    else:
        print(f"\n  WARNING: Some entries differ!")

    return 0 if all_match else 1


if __name__ == "__main__":
    sys.exit(main())
