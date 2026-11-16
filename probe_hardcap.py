import json, urllib.request, sys
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
SB = len(DOC.encode()); B = int(SB*0.5); LO, HI = int(B*0.96), int(B*1.04)

def chat(model, msgs, timeout=45):
    p = {"model": model, "messages": msgs, "temperature": 0.0,
         "max_tokens": 600, "enable_thinking": False}
    req = urllib.request.Request(f"{BASE}/chat/completions", data=json.dumps(p).encode(),
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return (json.loads(r.read().decode()))["choices"][0]["message"].get("content") or ""

PROMPT = (f"HARD LIMIT: your output MUST be between {LO} and {HI} UTF-8 bytes. "
"Length compliance outranks completeness: if every fact cannot fit, drop the least important "
"ones and keep entities, numbers and negations first. Never exceed the limit. "
f"Output only the compressed text.\n\nDOCUMENT:\n{DOC}")

print(f"source={SB} target={B} window=[{LO},{HI}]", flush=True)
for m in sys.argv[1:]:
    try:
        out = chat(m, [{"role": "user", "content": PROMPT}])
        nb = len(out.encode())
        print(f"{m:26s} {nb:4d}  ratio={nb/SB:.2f}  {'OK' if LO<=nb<=HI else 'out'}  {out[:90]!r}", flush=True)
    except Exception as e:
        print(f"{m:26s} ERROR {type(e).__name__}", flush=True)
