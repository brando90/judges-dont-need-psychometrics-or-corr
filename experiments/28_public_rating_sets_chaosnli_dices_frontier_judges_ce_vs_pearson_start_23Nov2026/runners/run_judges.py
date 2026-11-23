"""Run frozen label-free prompts through the Stanford gateway or the personal Codex plan and keep raw receipts.

Dated 11-23-2026. Reads no label file. One call per prompt, no content retries; a request that failed in
transport (no response received) is retried only on a later invocation. Keys are read from files by name and
never printed or written to a receipt.
"""
import argparse, hashlib, json, os, subprocess, tempfile, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

GATEWAY = "https://aiapi-prod.stanford.edu/v1/chat/completions"
KEY_FILE = Path.home() / "keys" / "stanford_stair_bm_gateway_key.txt"


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def gateway_call(model, system, user, max_tokens, timeout):
    body = json.dumps({"model": model, "temperature": 0, "max_tokens": max_tokens,
                       "messages": [{"role": "system", "content": system},
                                    {"role": "user", "content": user}]}).encode()
    req = urllib.request.Request(GATEWAY, data=body, headers={
        "Content-Type": "application/json", "Authorization": "Bearer " + KEY_FILE.read_text().strip()})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.load(r)
    choice = data["choices"][0]
    msg = choice["message"]
    return {"raw_response": msg.get("content") or "", "finish_reason": choice.get("finish_reason"),
            "served_model": data.get("model"), "usage": data.get("usage"),
            "had_reasoning_field": bool(msg.get("reasoning_content") or msg.get("reasoning"))}


def codex_call(model, effort, system, user, timeout):
    prompt = system + "\n\n" + user
    cmd = ["codex", "--ask-for-approval", "never", "exec", "--json", "--skip-git-repo-check",
           "--sandbox", "read-only", "--model", model, "--ignore-user-config",
           "-c", "forced_login_method=chatgpt", "-c", f'model_reasoning_effort="{effort}"', "-"]
    env = {k: os.environ[k] for k in ("PATH", "HOME", "LANG", "USER") if k in os.environ}
    env["CODEX_HOME"] = str(Path.home() / ".codex")
    with tempfile.TemporaryDirectory() as empty:
        r = subprocess.run(cmd, input=prompt, capture_output=True, text=True, cwd=empty, env=env, timeout=timeout)
    text, usage, tools = "", None, 0
    for line in r.stdout.splitlines():
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if ev.get("type") == "item.completed":
            item = ev["item"]
            if item.get("type") == "agent_message":
                text = item.get("text", "")
            elif item.get("type") not in ("reasoning",):
                tools += 1
        if ev.get("type") == "turn.completed":
            usage = ev.get("usage")
    if r.returncode or not text:
        raise RuntimeError(f"codex exit {r.returncode}")
    if tools:
        raise RuntimeError("unexpected tool item")
    return {"raw_response": text, "usage": usage, "served_model": model}


def claude_call(model, effort, system, user, timeout):
    """Mirror of the experiment 21 isolated one-turn Claude Code call, via the personal-plan wrapper."""
    import shutil
    cmd = [shutil.which("claude"), "-p", "--safe-mode", "--setting-sources", "", "--system-prompt", system,
           "--tools", "", "--strict-mcp-config", "--disable-slash-commands", "--no-session-persistence",
           "--model", model, "--effort", effort, "--output-format", "stream-json", "--verbose"]
    env = {k: os.environ[k] for k in ("PATH", "HOME", "LANG", "TMPDIR", "USER") if k in os.environ}
    with tempfile.TemporaryDirectory() as empty:
        r = subprocess.run(cmd, cwd=empty, input=user, capture_output=True, text=True, env=env, timeout=timeout)
    events = [json.loads(l) for l in r.stdout.splitlines() if l.strip().startswith("{")]
    init = next((e for e in events if e.get("subtype") == "init"), {})
    last = next((e for e in events if e.get("type") == "result"), {})
    if r.returncode or last.get("is_error") or last.get("subtype") != "success":
        raise RuntimeError(f"claude exit {r.returncode} status {last.get('subtype')}")
    if (init.get("model") != model or set(last.get("modelUsage", {})) != {model} or init.get("tools") != []
            or init.get("mcp_servers") != [] or last.get("num_turns") != 1):
        raise RuntimeError("isolation or model receipt failed")
    if any(e.get("rate_limit_info", {}).get("isUsingOverage") for e in events):
        raise RuntimeError("extra usage reported")
    return {"raw_response": last["result"], "usage": last.get("usage"), "served_model": model,
            "api_key_source": init.get("apiKeySource"), "cli_version": init.get("claude_code_version")}


def one(args, row):
    ident = {"id": row["id"], "system_sha256": sha(row["system"]), "user_sha256": sha(row["user"]),
             "model": args.model, "route": args.route}
    t = time.perf_counter()
    try:
        if args.route == "gateway":
            out = gateway_call(args.model, row["system"], row["user"], args.max_tokens, args.timeout)
        elif args.route == "claude":
            out = claude_call(args.model, args.effort, row["system"], row["user"], args.timeout)
        else:
            out = codex_call(args.model, args.effort, row["system"], row["user"], args.timeout)
        return {**ident, "status": "ok", **out, "elapsed_seconds": round(time.perf_counter() - t, 2)}
    except (urllib.error.URLError, TimeoutError, subprocess.TimeoutExpired, RuntimeError, OSError, KeyError) as e:
        kind = "http_%s" % e.code if isinstance(e, urllib.error.HTTPError) else type(e).__name__
        return {**ident, "status": "transport_error", "error": kind + (": " + str(e)[:120] if isinstance(e, RuntimeError) else "")}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("requests", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--requests-sha256", required=True)
    ap.add_argument("--route", choices=("gateway", "codex", "claude"), required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--effort", default="low")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--timeout", type=int, default=240)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    data = a.requests.read_bytes()
    assert hashlib.sha256(data).hexdigest() == a.requests_sha256, "request bundle differs from frozen hash"
    rows = [json.loads(l) for l in data.splitlines()]
    if a.limit:
        rows = rows[:a.limit]
    done = {}
    if a.output.exists():
        for l in a.output.read_text().splitlines():
            r = json.loads(l)
            if r["status"] == "ok" or r["status"] != "transport_error":
                done[r["id"]] = r
    todo = [r for r in rows if r["id"] not in done]
    print(f"{a.route}/{a.model}: {len(done)} done, {len(todo)} to run", flush=True)
    with a.output.open("a") as out, ThreadPoolExecutor(a.workers) as ex:
        for fut in as_completed([ex.submit(one, a, r) for r in todo]):
            res = fut.result()
            res["requests_sha256"] = a.requests_sha256
            out.write(json.dumps(res, ensure_ascii=False) + "\n")
            out.flush()
            print(res["id"], res["status"], flush=True)


if __name__ == "__main__":
    main()
