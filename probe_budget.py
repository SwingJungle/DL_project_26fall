import json, urllib.request, urllib.error, sys
from pathlib import Path

vals = {}
for line in Path("Qwen.txt").read_text(encoding="utf-8").splitlines():
    if "=" in line:
        k, v = line.split("=", 1); vals[k.strip()] = v.strip()
BASE, KEY = vals["URL_OPENAI"].rstrip("/"), vals["API"]

DOC = ("The Neral Institute was founded in 2041 by Mira Sol, not by Alan Vey, despite earlier "
"records naming Vey. Its primary sensor array operates only when the ambient temperature stays "
"below 12 degrees Celsius, and it shuts down automatically above that threshold. According to "
"director Iven Marrow, the planned 2045 expansion remains uncertain because the funding council "
"has not approved the second tranche. The Institute relocated from Tesk to Varn in 2039, two years "
"before the founding charter was signed. A separate observatory in Varn, operated by the same "
"council, reports readings on a twelve-hour cycle. Marrow denied that the 2039 relocation "
"influenced the 2041 charter, although the Varn facility already held the necessary permits.")

SB = len(DOC.encode())
budget = int(SB * 0.5)
PROMPT = (f"Compress the document to between {int(budget*0.96)} and {int(budget*1.04)} UTF-8 bytes "
          f"(target {budget} bytes). Preserve entities, numbers, negations, conditions and attributions. "
          f"Do not add information. Output only the compressed text.\n\nDOCUMENT:\n{DOC}")

def call(model, timeout=45):
    p = {"model": model, "messages": [{"role": "user", "content": PROMPT}],
         "temperature": 0.0, "max_tokens": 600, "enable_thinking": False}
    req = urllib.request.Request(f"{BASE}/chat/completions",
        data=json.dumps(p).encode(),
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
        method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        b = json.loads(r.read().decode())
    return b["choices"][0]["message"].get("content") or ""

print(f"source_bytes={SB} budget={budget}", flush=True)
print(f"{'model':26s} {'bytes':>6s} {'ratio':>7s} {'ok':>5s}", flush=True)
for m in sys.argv[1:]:
    try:
        out = call(m)
        nb = len(out.encode()); ratio = nb / SB
        print(f"{m:26s} {nb:6d} {ratio:7.2f} {str(0.48 <= ratio <= 0.52):>5s}", flush=True)
    except Exception as e:
        print(f"{m:26s} ERROR {type(e).__name__}: {str(e)[:60]}", flush=True)
