#!/usr/bin/env python3
"""
Qwen3-VL extraction tester — sends images + JSON schema to the Crane server
and prints the extracted result.

Usage:
  # From a JSON schema file + images
  python test_qwen3_vl.py --schema schema.json --images img1.jpg img2.jpg

  # From a JSON schema string + images
  python test_qwen3_vl.py --schema '{"nombre":"","fecha":""}' --images doc.jpg

  # From train.json dataset entry (for comparison)
  python test_qwen3_vl.py --entry 0
  python test_qwen3_vl.py --entry 0 --dataset train.json

  # Batch: test entries 0-9 and save results
  python test_qwen3_vl.py --entry 0-9 --output results.json

  # Custom server URL
  python test_qwen3_vl.py --entry 0 --url http://localhost:8080

  # Stream output
  python test_qwen3_vl.py --schema schema.json --images doc.jpg --stream
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    print("Error: requests library required. Install with: pip install requests")
    sys.exit(1)


PROMPT_TEMPLATE = """Based on the provided images, please fill out the following JSON structure. Analyze each image carefully and extract all relevant information. If any field cannot be determined from the images, leave it empty.

JSON structure to fill:
{schema}"""

DEFAULT_URL = "http://localhost:8080"
DEFAULT_MAX_TOKENS = 1024


def load_schema(schema_arg: str) -> str:
    """Load JSON schema from file path or inline string."""
    if os.path.isfile(schema_arg):
        with open(schema_arg) as f:
            data = json.load(f)
        return json.dumps(data, indent=2, ensure_ascii=False)

    # Try parsing as inline JSON
    try:
        data = json.loads(schema_arg)
        return json.dumps(data, indent=2, ensure_ascii=False)
    except json.JSONDecodeError:
        print(f"Error: '{schema_arg}' is not a valid JSON file or JSON string")
        sys.exit(1)


def build_content(image_paths: list[str], schema_text: str) -> list[dict]:
    """Build the OpenAI-style content array with images + text."""
    content = []

    for img_path in image_paths:
        abs_path = os.path.abspath(img_path)
        if not os.path.isfile(abs_path):
            print(f"Warning: image not found: {abs_path}")
            continue
        content.append({
            "type": "image_url",
            "image_url": {"url": f"file://{abs_path}"}
        })

    prompt = PROMPT_TEMPLATE.format(schema=schema_text)
    content.append({"type": "text", "text": prompt})

    return content


def call_server(url: str, content: list[dict], max_tokens: int, stream: bool = False) -> str:
    """Send request to crane-oai server and return the response text."""
    payload = {
        "model": "qwen3-vl",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "stream": stream,
    }

    if stream:
        return call_server_stream(url, payload)

    resp = requests.post(
        f"{url}/v1/chat/completions",
        json=payload,
        timeout=300,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def call_server_stream(url: str, payload: dict) -> str:
    """Stream tokens from the server, printing as they arrive."""
    resp = requests.post(
        f"{url}/v1/chat/completions",
        json=payload,
        timeout=300,
        stream=True,
    )
    resp.raise_for_status()

    full_text = ""
    for line in resp.iter_lines():
        if not line:
            continue
        line = line.decode("utf-8")
        if not line.startswith("data: "):
            continue
        data_str = line[6:]
        if data_str.strip() == "[DONE]":
            break
        try:
            chunk = json.loads(data_str)
            delta = chunk["choices"][0].get("delta", {})
            token = delta.get("content", "")
            if token:
                print(token, end="", flush=True)
                full_text += token
        except (json.JSONDecodeError, KeyError, IndexError):
            pass

    print()
    return full_text


def load_dataset_entry(dataset_path: str, index: int) -> tuple[list[str], str, str]:
    """Load an entry from the dataset. Returns (image_paths, schema_text, expected_output)."""
    with open(dataset_path) as f:
        entries = json.load(f)

    if index >= len(entries):
        print(f"Error: entry {index} out of range (dataset has {len(entries)} entries)")
        sys.exit(1)

    entry = entries[index]
    dataset_dir = os.path.dirname(os.path.abspath(dataset_path))

    # Resolve image paths
    images = entry.get("images", [])
    image_paths = []
    for p in images:
        full = p if os.path.isabs(p) else os.path.join(dataset_dir, p)
        image_paths.append(full)

    # Extract human prompt (contains the JSON schema)
    convs = entry.get("conversations", [])
    human_text = ""
    expected = ""
    for c in convs:
        if c.get("from") == "human":
            human_text = c.get("value", "")
        elif c.get("from") in ("gpt", "assistant"):
            expected = c.get("value", "")

    return image_paths, human_text, expected


def parse_entry_range(entry_arg: str) -> list[int]:
    """Parse entry argument: '5' -> [5], '0-9' -> [0..9], '1,3,5' -> [1,3,5]."""
    if "-" in entry_arg and not entry_arg.startswith("-"):
        start, end = entry_arg.split("-", 1)
        return list(range(int(start), int(end) + 1))
    elif "," in entry_arg:
        return [int(x.strip()) for x in entry_arg.split(",")]
    else:
        return [int(entry_arg)]


def compare_json(expected: str, actual: str) -> tuple[int, int]:
    """Compare two JSON outputs field by field. Returns (matching, total)."""
    try:
        exp = json.loads(expected)
        act = json.loads(actual)
    except json.JSONDecodeError:
        return 0, 1

    def flatten(obj, prefix=""):
        items = {}
        if isinstance(obj, dict):
            for k, v in obj.items():
                items.update(flatten(v, f"{prefix}.{k}" if prefix else k))
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                items.update(flatten(v, f"{prefix}[{i}]"))
        else:
            items[prefix] = str(obj)
        return items

    exp_flat = flatten(exp)
    act_flat = flatten(act)

    total = len(exp_flat)
    matching = sum(1 for k in exp_flat if exp_flat.get(k) == act_flat.get(k))
    return matching, total


def main():
    parser = argparse.ArgumentParser(
        description="Test Qwen3-VL extraction via Crane server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_qwen3_vl.py --schema schema.json --images img1.jpg img2.jpg
  python test_qwen3_vl.py --entry 0
  python test_qwen3_vl.py --entry 0-9 --output results.json
  python test_qwen3_vl.py --schema '{"name":"","date":""}' --images doc.jpg --stream
        """,
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--schema", help="JSON schema file or inline JSON string")
    mode.add_argument("--entry", help="Dataset entry index, range (0-9), or list (1,3,5)")

    parser.add_argument("--images", nargs="+", help="Image file paths (for --schema mode)")
    parser.add_argument("--dataset", default="train.json", help="Dataset JSON file (default: train.json)")
    parser.add_argument("--url", default=DEFAULT_URL, help=f"Server URL (default: {DEFAULT_URL})")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help=f"Max tokens (default: {DEFAULT_MAX_TOKENS})")
    parser.add_argument("--stream", action="store_true", help="Stream output tokens")
    parser.add_argument("--output", help="Save results to JSON file (batch mode)")
    parser.add_argument("--quiet", action="store_true", help="Only print the extracted JSON")

    args = parser.parse_args()

    # ── Schema + Images mode ──
    if args.schema:
        if not args.images:
            parser.error("--schema requires --images")

        schema_text = load_schema(args.schema)
        content = build_content(args.images, schema_text)

        if not args.quiet:
            print(f"Images: {', '.join(args.images)}")
            print(f"Server: {args.url}")
            print("---")

        t0 = time.time()
        result = call_server(args.url, content, args.max_tokens, args.stream)
        elapsed = time.time() - t0

        if not args.stream:
            print(result)

        if not args.quiet:
            print("---")
            print(f"Time: {elapsed:.2f}s")

        return

    # ── Dataset entry mode ──
    indices = parse_entry_range(args.entry)
    results = []

    for idx in indices:
        image_paths, human_text, expected = load_dataset_entry(args.dataset, idx)

        if not args.quiet:
            print(f"{'='*50}")
            print(f"Entry #{idx}")
            print(f"Images: {', '.join(os.path.basename(p) for p in image_paths)}")
            print(f"---")

        # Build content using the original human prompt (already has schema inside)
        content = []
        for img_path in image_paths:
            abs_path = os.path.abspath(img_path)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"file://{abs_path}"}
            })
        content.append({"type": "text", "text": human_text})

        t0 = time.time()
        try:
            result = call_server(args.url, content, args.max_tokens, args.stream and len(indices) == 1)
        except requests.exceptions.ConnectionError:
            print(f"Error: cannot connect to {args.url}. Is the server running?")
            print(f"Start it with: ./target/release/crane-oai --model-path <model_path> --port 8080")
            sys.exit(1)
        except Exception as e:
            print(f"Error: {e}")
            result = ""
        elapsed = time.time() - t0

        if not (args.stream and len(indices) == 1):
            print("MODEL OUTPUT:")
            print(result)

        # Compare with expected
        if expected:
            matching, total = compare_json(expected, result)
            accuracy = (matching / total * 100) if total > 0 else 0

            if not args.quiet:
                print("---")
                print(f"EXPECTED OUTPUT:")
                print(expected)
                print(f"---")
                print(f"Accuracy: {matching}/{total} fields ({accuracy:.1f}%)")
                print(f"Time: {elapsed:.2f}s")

            results.append({
                "entry": idx,
                "accuracy": round(accuracy, 1),
                "matching": matching,
                "total": total,
                "time_s": round(elapsed, 2),
                "output": result,
                "expected": expected,
            })
        else:
            results.append({
                "entry": idx,
                "time_s": round(elapsed, 2),
                "output": result,
            })

    # ── Summary for batch ──
    if len(indices) > 1:
        print(f"\n{'='*50}")
        print("BATCH SUMMARY")
        print(f"{'='*50}")
        total_acc = []
        for r in results:
            acc = r.get("accuracy", None)
            acc_str = f"{acc}%" if acc is not None else "N/A"
            print(f"  Entry {r['entry']:>4d}:  {acc_str:>7s}  ({r['time_s']:.1f}s)")
            if acc is not None:
                total_acc.append(acc)
        if total_acc:
            avg = sum(total_acc) / len(total_acc)
            print(f"\n  Average accuracy: {avg:.1f}%")
            print(f"  Total time: {sum(r['time_s'] for r in results):.1f}s")

    # ── Save results ──
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
