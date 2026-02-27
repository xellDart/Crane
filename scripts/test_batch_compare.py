#!/usr/bin/env python3
"""
Compare batch decode vs sequential decode outputs for Qwen3-VL.

Sends N concurrent requests (to trigger batch decode) and then
N sequential requests (one at a time). Compares the outputs token-by-token.

Usage:
  # Start server first:
  #   ./target/release/crane-oai --model-path <path> --model-type qwen3_vl \
  #     --max-concurrent 4 --decode-tokens-per-seq 8 --port 8080

  python scripts/test_batch_compare.py --entries 0,1,2 --dataset train.json
  python scripts/test_batch_compare.py --entries 0-4 --max-tokens 128
"""

import argparse
import base64
import json
import mimetypes
import os
import sys
import time
import concurrent.futures
from pathlib import Path

try:
    import requests
except ImportError:
    print("pip install requests")
    sys.exit(1)


DEFAULT_URL = "http://localhost:8080"


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
    if index >= len(entries):
        print(f"Entry {index} out of range ({len(entries)} entries)")
        sys.exit(1)
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


def build_request(image_paths, prompt, max_tokens, temperature=0.01):
    content = []
    for img_path in image_paths:
        if os.path.isfile(img_path):
            content.append({
                "type": "image_url",
                "image_url": {"url": image_to_data_uri(img_path)}
            })
    content.append({"type": "text", "text": prompt})

    return {
        "model": "qwen3-vl",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }


def send_request(url, payload, entry_idx):
    t0 = time.time()
    resp = requests.post(f"{url}/v1/chat/completions", json=payload, timeout=300)
    elapsed = time.time() - t0
    if not resp.ok:
        return entry_idx, None, elapsed, f"HTTP {resp.status_code}: {resp.text[:200]}"
    data = resp.json()
    text = data["choices"][0]["message"]["content"]
    return entry_idx, text, elapsed, None


def parse_entry_range(entry_arg: str) -> list:
    if "-" in entry_arg and not entry_arg.startswith("-"):
        start, end = entry_arg.split("-", 1)
        return list(range(int(start), int(end) + 1))
    elif "," in entry_arg:
        return [int(x.strip()) for x in entry_arg.split(",")]
    else:
        return [int(entry_arg)]


def main():
    parser = argparse.ArgumentParser(description="Compare batch vs sequential decode")
    parser.add_argument("--entries", required=True, help="Entry indices: 0,1,2 or 0-4")
    parser.add_argument("--dataset", default="train.json", help="Dataset JSON file")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--runs", type=int, default=2, help="Number of batch runs to compare")
    args = parser.parse_args()

    indices = parse_entry_range(args.entries)
    print(f"Entries: {indices}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Server: {args.url}")

    # Load and prepare requests
    payloads = {}
    for idx in indices:
        image_paths, prompt = load_dataset_entry(args.dataset, idx)
        payloads[idx] = build_request(image_paths, prompt, args.max_tokens)
        print(f"  Entry {idx}: {len(image_paths)} images, prompt len {len(prompt)}")

    all_run_results = []

    for run_num in range(args.runs):
        print(f"\n{'='*60}")
        print(f"  RUN {run_num + 1}/{args.runs} — Concurrent ({len(indices)} requests)")
        print(f"{'='*60}")

        t0 = time.time()
        results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(indices)) as executor:
            futures = {
                executor.submit(send_request, args.url, payloads[idx], idx): idx
                for idx in indices
            }
            for future in concurrent.futures.as_completed(futures):
                idx, text, elapsed, err = future.result()
                if err:
                    print(f"  Entry {idx}: ERROR — {err}")
                    results[idx] = None
                else:
                    preview = text[:80].replace('\n', ' ') + ("..." if len(text) > 80 else "")
                    print(f"  Entry {idx}: {elapsed:.2f}s — {preview}")
                    results[idx] = text

        total = time.time() - t0
        print(f"  Total: {total:.2f}s")
        all_run_results.append(results)

    # Compare runs
    print(f"\n{'='*60}")
    print("  COMPARISON")
    print(f"{'='*60}")

    all_match = True
    for idx in indices:
        texts = [r.get(idx) for r in all_run_results]
        if any(t is None for t in texts):
            print(f"  Entry {idx}: SKIP (some runs failed)")
            continue

        # Compare all runs
        if all(t == texts[0] for t in texts):
            print(f"  Entry {idx}: MATCH (all {args.runs} runs identical)")
            print(f"    Length: {len(texts[0])} chars")
        else:
            all_match = False
            print(f"  Entry {idx}: MISMATCH!")
            for i, t in enumerate(texts):
                print(f"    Run {i+1}: {len(t)} chars — {t[:100]}...")
            # Find first difference
            min_len = min(len(t) for t in texts)
            for pos in range(min_len):
                chars = [t[pos] for t in texts]
                if len(set(chars)) > 1:
                    print(f"    First diff at char {pos}: {chars}")
                    break

    if all_match:
        print(f"\n  ALL ENTRIES MATCH across {args.runs} runs!")
    else:
        print(f"\n  WARNING: Some entries differ between runs!")

    return 0 if all_match else 1


if __name__ == "__main__":
    sys.exit(main())
