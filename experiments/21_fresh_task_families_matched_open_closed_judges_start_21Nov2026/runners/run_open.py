"""Run one stateless local Qwen chat per frozen prompt and preserve raw receipts."""
import argparse
import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path

SERVER = "http://127.0.0.1:11434"
MODEL = "qwen3.5:9b-q4_K_M"
MODEL_DIGEST = "56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"


def digest(contents):
    return hashlib.sha256(contents).hexdigest()


def encoded(document):
    return json.dumps(document, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


def api(path, document=None, timeout=30):
    request = urllib.request.Request(
        SERVER + path, data=None if document is None else encoded(document),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def prompt_rows(path, expected_digest):
    source = path.read_bytes()
    if digest(source) != expected_digest:
        raise ValueError("Prompt bundle differs from the frozen SHA-256.")
    rows = []
    for line in source.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if set(row) == {"id", "system", "user"}:
            messages = [{"role": role, "content": row[role]} for role in ("system", "user")]
        elif set(row) == {"id", "messages"}:
            messages = row["messages"]
        else:
            raise ValueError("Expected prompt-only rows containing id and system/user or messages.")
        if ([m["role"] for m in messages] != ["system", "user"] or
            any(set(m) != {"role", "content"} or not isinstance(m["content"], str) for m in messages)):
            raise ValueError("Each prompt must contain exactly one system message and one user message.")
        rows.append((row["id"], messages))
    if not rows or len({row_id for row_id, _ in rows}) != len(rows):
        raise ValueError("Expected nonempty prompt rows with unique ids.")
    return rows


def save_receipt(path, data):
    partial = path.with_suffix(".partial")
    with partial.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    partial.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("requests", type=Path)
    parser.add_argument("receipts", type=Path)
    parser.add_argument("--requests-sha256", required=True)
    parser.add_argument("--context", type=int, default=16384)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--timeout", type=int, default=1800, help="Per-request transport timeout in seconds.")
    parser.add_argument("--seed", type=int, default=20261121)
    parser.add_argument("--json-mode", action="store_true", help="Enable only if frozen in the protocol.")
    args = parser.parse_args()
    rows = prompt_rows(args.requests, args.requests_sha256)
    actual = {row["name"]: row["digest"] for row in api("/api/tags")["models"]}
    if actual.get(MODEL) != MODEL_DIGEST:
        raise ValueError("Local model digest differs from the pinned checkpoint.")
    version = api("/api/version")["version"]
    args.receipts.mkdir(mode=0o700, parents=True, exist_ok=True)
    options = {"temperature": 0, "seed": args.seed, "num_ctx": args.context, "num_predict": args.max_tokens}
    for index, (row_id, messages) in enumerate(rows):
        request = {"model": MODEL, "messages": messages, "stream": False, "think": False,
                   "options": options, "keep_alive": "10m"}
        if args.json_mode:
            request["format"] = "json"
        fingerprint = {"id": row_id, "messages_sha256": digest(encoded(messages)),
                       "request_sha256": digest(encoded(request)),
                       "prompts_sha256": args.requests_sha256, "model_digest": MODEL_DIGEST,
                       "server_version": version}
        receipt = args.receipts / f"{index:05d}.json"
        if receipt.exists():
            prior = json.loads(receipt.read_text())
            if any(prior[key] != value for key, value in fingerprint.items()):
                raise ValueError(f"Existing receipt does not match frozen request {index}.")
            continue
        started = time.monotonic()
        response = api("/api/chat", request, timeout=args.timeout)
        if response.get("model") != MODEL or not response.get("done") or "content" not in response.get("message", {}):
            raise RuntimeError(f"Invalid Ollama response for request {index}.")
        elapsed = time.monotonic() - started
        save_receipt(receipt, {**fingerprint, "options": options, "think": False, "json_mode": args.json_mode,
                               "response": response, "wall_seconds": elapsed})
        print(f"Completed {index + 1}/{len(rows)} with {response['done_reason']}, "
              f"{response.get('prompt_eval_count')} input / {response.get('eval_count')} output tokens, "
              f"{elapsed:.2f}s.", flush=True)
    print(f"All {len(rows)} frozen requests have receipts under {args.receipts}.", flush=True)


if __name__ == "__main__":
    main()
