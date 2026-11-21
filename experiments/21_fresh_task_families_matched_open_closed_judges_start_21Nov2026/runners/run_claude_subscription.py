"""Isolated one-turn Claude Code judge with authenticated subscription receipts."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from uuid import uuid4
import time


def subscription_environment():
    """Use the native login, with no provider keys or model-routing overrides."""
    return {key: os.environ[key] for key in ("PATH", "HOME", "LANG", "TMPDIR", "USER")
            if key in os.environ}


def check_included_usage():
    """Require first-party subscription authentication through the native CLI."""
    cli = shutil.which("claude")
    result = subprocess.run([cli, "auth", "status"], check=True, text=True, capture_output=True,
                            env=subscription_environment())
    login = json.loads(result.stdout)
    if not (login.get("loggedIn") and login.get("authMethod") == "claude.ai"
            and login.get("apiProvider") == "firstParty" and login.get("subscriptionType")):
        raise RuntimeError("A native first-party Claude subscription login is required.")
    return {"auth_method": login["authMethod"], "subscription_type": login["subscriptionType"],
            "api_provider": login["apiProvider"]}


def ask(model, effort, system, user, timeout=120):
    """Run a new conversation in an empty directory, accepting only one tool-free response."""
    arguments = [shutil.which("claude"), "-p", "--safe-mode", "--setting-sources", "",
                 "--system-prompt", system, "--tools", "", "--strict-mcp-config",
                 "--disable-slash-commands", "--no-session-persistence", "--model", model,
                 "--effort", effort, "--output-format", "stream-json", "--verbose"]
    start = time.perf_counter()
    with tempfile.TemporaryDirectory() as empty:
        result = subprocess.run(arguments, cwd=empty, input=user, capture_output=True,
                                text=True, env=subscription_environment(), timeout=timeout)
    events = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    init = next((event for event in events if event.get("subtype") == "init"), {})
    last = next((event for event in events if event.get("type") == "result"), {})
    if result.returncode or last.get("is_error") or last.get("subtype") != "success":
        raise RuntimeError(f"Claude call failed (exit {result.returncode}, status {last.get('subtype')}).")
    if (init.get("model") != model or set(last.get("modelUsage", {})) != {model}
            or init.get("apiKeySource") != "none" or init.get("tools") != []
            or init.get("mcp_servers") != [] or last.get("num_turns") != 1):
        raise RuntimeError("Subscription/model/tool/isolation receipt failed.")
    if any(block.get("type") in {"tool_use", "server_tool_use"}
           for event in events if event.get("type") == "assistant"
           for block in event["message"]["content"]):
        raise RuntimeError("Unexpected tool call.")
    if any(event.get("rate_limit_info", {}).get("isUsingOverage") for event in events):
        raise RuntimeError("Extra usage was reported.")
    return {"raw_response": last["result"], "model": model, "effort": effort,
            "num_turns": last["num_turns"], "tools": init["tools"],
            "mcp_servers": init["mcp_servers"], "api_key_source": init["apiKeySource"],
            "cli_version": init.get("claude_code_version"), "usage": last.get("usage"),
            "elapsed_seconds": round(time.perf_counter() - start, 3)}


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def score_one(item, model, effort, timeout):
    identity = {"id": item["id"], "system_sha256": digest(item["system"]),
                "user_sha256": digest(item["user"]), "model": model, "effort": effort}
    try:
        receipt = ask(model, effort, item["system"], item["user"], timeout)
    except subprocess.TimeoutExpired:
        return {**identity, "status": "timeout"}
    except (RuntimeError, ValueError, OSError) as exc:
        # RuntimeError contains only locally constructed status, never input text.
        public_error = str(exc) if isinstance(exc, RuntimeError) else type(exc).__name__
        return {**identity, "status": "failed", "error": public_error}
    return {**identity, "status": "ok", **receipt}


def run(args):
    payload = args.requests.read_bytes()
    request_sha = hashlib.sha256(payload).hexdigest()
    if request_sha != args.request_sha256:
        raise ValueError("Request manifest SHA-256 mismatch.")
    items = [json.loads(line) for line in payload.splitlines()]
    if not items or any(set(row) != {"id", "system", "user"} or
                        any(not isinstance(value, str) or not value for value in row.values())
                        for row in items):
        raise ValueError("Requests must contain exactly nonempty id, system and user strings.")
    ids = [row["id"] for row in items]
    if len(set(ids)) != len(ids) or args.workers < 1 or args.timeout <= 0:
        raise ValueError("Request IDs must be unique and worker/timeout budgets positive.")
    out = args.output.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    # A single writer stores every finished call, replacing the checkpoint atomically.
    with out.with_suffix(out.suffix + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        old = [json.loads(line) for line in out.read_bytes().splitlines()] if out.exists() else []
        seen = {row["id"]: row for row in old}
        if (len(seen) != len(old) or not set(seen) <= set(ids)
                or any(r["requests_sha256"] != request_sha or r["model"] != args.model or
                       r["effort"] != args.effort for r in old)):
            raise ValueError("Existing receipt conflicts with this manifest or model.")
        todo = [item for item in items if item["id"] not in seen]
        if not todo:
            print(f"All {len(items)} requests already have receipts.")
            return
        auth = check_included_usage()
        with ThreadPoolExecutor(max_workers=args.workers) as workers:
            pending = [workers.submit(score_one, item, args.model, args.effort, args.timeout)
                       for item in todo]
            for future in as_completed(pending):
                row = future.result()
                row.update(requests_sha256=request_sha, auth=auth)
                seen[row["id"]] = row
                tmp = out.with_name(out.name + "." + str(uuid4()) + ".tmp")
                with tmp.open("x", encoding="utf-8") as stream:
                    for item_id in ids:
                        if item_id in seen:
                            stream.write(json.dumps(seen[item_id], ensure_ascii=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                tmp.replace(out)
                print(f"{len(seen)}/{len(items)} {row['id']} {row['status']}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True, help="Label-free request JSONL.")
    parser.add_argument("--request-sha256", required=True, help="Pinned input manifest SHA-256.")
    parser.add_argument("--output", type=Path, required=True,
                        help="Local checkpoint for raw model text and receipts; do not publish unaudited.")
    parser.add_argument("--model", required=True, help="Exact native subscription model ID.")
    parser.add_argument("--effort", required=True, choices=["low", "medium", "high", "max"])
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=120, help="Wall seconds per request.")
    run(parser.parse_args())
