"""P0.8 model screening: probe the gateway for the five hard criteria.

Reads base URL + key from Qwen.txt. Never prints the key.
Checks per model: thinking toggle, json_object mode, temperature=0,
returned model id (pinning), usage (token counting).
"""
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CRED = ROOT / "Qwen.txt"


def load_cred():
    vals = {}
    for line in CRED.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            vals[k.strip()] = v.strip()
    return vals["URL_OPENAI"].rstrip("/"), vals["API"]


BASE, KEY = load_cred()


def call(model, *, thinking=None, temperature=0.0, max_tokens=96,
         json_mode=False, timeout=90):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You reply with JSON only. No prose."},
            {"role": "user", "content": 'Output exactly {"ok": true, "tag": "probe"} in JSON format.'},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    if thinking is not None:
        payload.update(thinking)
    req = urllib.request.Request(
        f"{BASE}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = json.loads(r.read().decode())
        dt = int((time.time() - t0) * 1000)
        ch = body["choices"][0]
        msg = ch.get("message", {})
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or ""
        usage = body.get("usage") or {}
        return {
            "ok": True,
            "model_returned": body.get("model"),
            "content": content,
            "content_len": len(content),
            "reasoning_len": len(reasoning),
            "finish_reason": ch.get("finish_reason"),
            "usage": usage,
            "latency_ms": dt,
            "json_parsed": _try_json(content),
        }
    except urllib.error.HTTPError as e:
        detail = e.read().decode()[:400]
        return {"ok": False, "http": e.code, "error": detail, "latency_ms": int((time.time() - t0) * 1000)}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def _try_json(s):
    try:
        return json.loads(s)
    except Exception:  # noqa: BLE001
        return None


MODELS = [
    "qwen3.7-plus",
    "qwen3.7-max",
    "qwen3.8-flash",
    "qwen3.8-max",
    "deepseek-v4-flash-0731",
    "deepseek-v4-pro-0813",
    "glm-5.3",
]

FLAGS = {
    "none": None,
    "enable_thinking_false": {"enable_thinking": False},
    "thinking_disabled": {"thinking": {"type": "disabled"}},
}

results = {}
for m in MODELS:
    print(f"\n=== {m} ===", flush=True)
    rec = {}
    for name, flag in FLAGS.items():
        r = call(m, thinking=flag)
        rec[name] = r
        if r.get("ok"):
            print(f"  {name:24s} ok  model={r['model_returned']!r:32s} "
                  f"reason_len={r['reasoning_len']:5d} json={r['json_parsed'] is not None} "
                  f"usage={'Y' if r['usage'] else 'N'}", flush=True)
        else:
            print(f"  {name:24s} FAIL http={r.get('http')} {str(r.get('error'))[:110]}", flush=True)
    # json_object + best available thinking flag
    worked = None
    for name in ("enable_thinking_false", "thinking_disabled"):
        if rec.get(name, {}).get("ok"):
            worked = FLAGS[name]
            break
    rj = call(m, thinking=worked, json_mode=True)
    rec["json_object"] = rj
    if rj.get("ok"):
        print(f"  {'json_object':24s} ok  json={rj['json_parsed'] is not None} "
              f"finish={rj['finish_reason']} usage={rj['usage'].get('prompt_tokens', '?')}"
              f"/{rj['usage'].get('completion_tokens', '?')}", flush=True)
    else:
        print(f"  {'json_object':24s} FAIL http={rj.get('http')} {str(rj.get('error'))[:110]}", flush=True)
    results[m] = rec

out = ROOT / "model_probe_results.json"
out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\nwrote {out.name}")

# ---- summary ----
print("\n\n=== SUMMARY ===")
hdr = f"{'model':28s} {'thinkOff':8s} {'jsonObj':8s} {'temp0':7s} {'pin':10s} {'usage':6s}"
print(hdr)
print("-" * len(hdr))
for m, rec in results.items():
    off = any(rec.get(k, {}).get("ok") for k in ("enable_thinking_false", "thinking_disabled"))
    # does reasoning actually vanish?
    base_r = rec.get("none", {})
    off_r = rec.get("enable_thinking_false", {})
    truly_off = off_r.get("ok") and off_r.get("reasoning_len", 0) == 0
    jo = rec.get("json_object", {}).get("ok") and rec.get("json_object", {}).get("json_parsed") is not None
    pin = rec.get("json_object", {}).get("model_returned", "-")
    pin_exact = "exact" if pin == m else "MISMATCH"
    usage = "Y" if rec.get("json_object", {}).get("usage") else "N"
    print(f"{m:28s} {str(truly_off):8s} {str(bool(jo)):8s} {'?':7s} {pin_exact:10s} {usage:6s}")
