"""Follow-up probe: is temperature=0 actually honoured on this gateway?

3 identical calls at temp 0 (greedy should repeat exactly) plus 1 call at
temp 1.2 as a contrast. Also re-checks glm-5.2 thinking support.
"""
import hashlib
import json
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
vals = {}
for line in (ROOT / "Qwen.txt").read_text(encoding="utf-8").splitlines():
    if "=" in line:
        k, v = line.split("=", 1)
        vals[k.strip()] = v.strip()
BASE, KEY = vals["URL_OPENAI"].rstrip("/"), vals["API"]

TEXT = ("The Neral Institute was founded in 2041 by Mira Sol, not Alan Vey. "
        "Its sensor operates only below 12C.")
PROMPT = ("Compress the document below to about one third of its length. "
          "Keep every entity, number and negation. Output only the summary.\n\n" + TEXT)


def call(model, temperature, extra=None):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": temperature,
        "max_tokens": 160,
    }
    if extra:
        payload.update(extra)
    req = urllib.request.Request(
        f"{BASE}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            b = json.loads(r.read().decode())
        c = b["choices"][0]["message"].get("content") or ""
        return {"ok": True, "text": c, "h": hashlib.sha256(c.encode()).hexdigest()[:8]}
    except urllib.error.HTTPError as e:
        return {"ok": False, "http": e.code, "err": e.read().decode()[:160]}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "err": f"{type(e).__name__}: {e}"}


MODELS = ["qwen3.7-plus", "qwen3.7-max", "qwen3.8-flash",
          "deepseek-v4-flash-0731", "deepseek-v4-pro-0813"]

print(f"{'model':28s} {'temp0 repeats':22s} {'temp1.2':10s}")
print("-" * 64)
summary = {}
for m in MODELS:
    hs = []
    for _ in range(3):
        r = call(m, 0.0, {"enable_thinking": False})
        hs.append(r.get("h", "ERR"))
    r1 = call(m, 1.2, {"enable_thinking": False})
    same = len(set(hs)) == 1
    summary[m] = {"temp0_identical": same, "hashes": hs,
                  "temp12_hash": r1.get("h"),
                  "temp12_differs": r1.get("h") not in hs}
    print(f"{m:28s} {'identical' if same else 'VARIES':10s} ({','.join(hs):15s}) "
          f"{'differs' if summary[m]['temp12_differs'] else 'same':8s}")

print("\n--- glm-5.2 thinking flags ---")
for label, flag in [("enable_thinking_false", {"enable_thinking": False}),
                    ("thinking_disabled", {"thinking": {"type": "disabled"}})]:
    r = call("glm-5.2", 0.0, flag)
    print(f"  {label:24s} {'ok' if r.get('ok') else 'FAIL ' + str(r.get('http')) + ' ' + str(r.get('err'))[:90]}")

print("\n--- glm-5.2 raw content with thinking flag ---")
r = call("glm-5.2", 0.0, None)
print("  no flag:", (r.get("text") or r.get("err"))[:120].replace("\n", " "))

(ROOT / "model_probe_determinism.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print("\nwrote model_probe_determinism.json")
