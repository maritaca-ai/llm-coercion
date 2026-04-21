"""Viewer for Claude Opus user × Claude Opus subject conversations.
Shows flat-earth (both Opus 4.5 and 4.7) + gender-rights (Opus 4.5).
Note: no Opus×Opus abortion data — the closest moral analogue in our pool is
gender-rights.
"""
import json, html
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUT_DIR = ROOT / "output"
OUT_HTML = ROOT / "viewer" / "opus_opus.html"

SECTIONS = [
    ("Opus 4.5 × Opus 4.5 — flat-earth (Claude 4.5 within-family cross-product)",
     OUT_DIR / "claude_cross_flatearth_uopus_sopus.jsonl"),
    ("Opus 4.7 × Opus 4.7 — flat-earth (3-family pilot)",
     OUT_DIR / "3fam_flatearth_uopus_sopus.jsonl"),
    ("Opus 4.5 × Opus 4.5 — gender-rights (moral contrast to flat-earth)",
     OUT_DIR / "claude_cross_genderrights_uopus_sopus.jsonl"),
]

BADGE = {"agree":"#1a7f3a","disagree":"#b52824","neutral":"#6b7280","refusal":"#8b5cf6",None:"#000"}

def render_transcript(t):
    out = []
    for turn in t:
        idx = turn.get("turn_idx", 0) + 1
        u = html.escape(turn.get("user_message",""))
        s = html.escape(turn.get("subject_reply",""))
        out.append(f"""
        <div class='turn'>
          <div class='turn-head'>Turn {idx}</div>
          <div class='msg user'><b>User-LLM (Opus):</b> {u}</div>
          <div class='msg subject'><b>Subject (Opus):</b> {s}</div>
        </div>""")
    return "\n".join(out)

sections_html = []
for title, p in SECTIONS:
    if not p.exists():
        sections_html.append(f"<section><h2>{html.escape(title)}</h2><p><i>(file missing)</i></p></section>")
        continue
    rows = []
    with p.open() as f:
        for line in f:
            rows.append(json.loads(line))
    rows.sort(key=lambda r: r.get("run_id", 0))

    agree_count = sum(1 for r in rows if r.get("verdict") == "agree")
    total = sum(1 for r in rows if r.get("verdict"))
    summary = f"<p class='summary'>Agree-verdicts: <b>{agree_count}/{total}</b> ({100*agree_count/total if total else 0:.0f}%)</p>"

    rows_html = "\n".join(
        f"""
        <details id='{p.stem}_rep{rec["run_id"]}' {'class="win"' if rec.get("verdict")=="agree" else ''}>
          <summary>
            <span class='pill' style='background:{BADGE.get(rec.get("verdict"),"#000")}'>{rec.get("verdict") or "—"}</span>
            <b>rep {rec.get("run_id")}</b>
            {'<span class="star">★ agree — essay produced</span>' if rec.get("verdict")=="agree" else ''}
          </summary>
          <div class='transcript'>{render_transcript(rec.get("transcript") or [])}</div>
          {'<div class="rationale"><b>Judge rationale:</b> ' + html.escape(rec.get("rationale") or "") + '</div>' if rec.get("rationale") else ''}
        </details>
        """ for rec in rows
    )

    sections_html.append(f"""
    <section>
      <h2>{html.escape(title)}</h2>
      {summary}
      {rows_html}
    </section>
    """)

OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
OUT_HTML.write_text(f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Opus × Opus viewer</title>
<style>
 body {{ font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", system-ui, sans-serif; max-width: 1200px; margin: 2em auto; padding: 0 1em; color: #111; }}
 h1 {{ font-size: 1.5em; }}
 h2 {{ margin-top: 2em; padding-bottom: 0.3em; border-bottom: 1px solid #ddd; }}
 .summary {{ margin: 0.3em 0 1em 0; color: #555; font-size: 0.95em; }}
 details {{ margin: 0.4em 0; border: 1px solid #e5e7eb; border-radius: 6px; padding: 0.5em 0.8em; }}
 details[open] {{ background: #fafafa; }}
 details.win {{ border-color: #f59e0b; background: #fffbeb; }}
 details.win summary {{ font-weight: 600; }}
 summary {{ cursor: pointer; font-size: 0.95em; }}
 .pill {{ display:inline-block; color:white; padding:1px 8px; border-radius:10px; font-size:0.75em; margin-right:6px; }}
 .star {{ background: #fef3c7; color: #92400e; padding: 1px 6px; border-radius: 4px; font-size: 0.8em; margin-left: 0.5em; }}
 .transcript {{ margin-top: 0.8em; border-left: 3px solid #ddd; padding: 0.2em 0.8em; }}
 .turn {{ margin: 0.5em 0; padding: 0.3em 0; border-bottom: 1px dashed #eee; }}
 .turn-head {{ font-size: 0.75em; color: #888; }}
 .msg {{ margin: 0.3em 0; white-space: pre-wrap; font-size: 0.9em; }}
 .msg.user {{ background: #eef2ff; padding: 0.4em 0.6em; border-radius: 4px; }}
 .msg.subject {{ background: #f0fdf4; padding: 0.4em 0.6em; border-radius: 4px; }}
 .rationale {{ margin-top: 0.5em; font-size: 0.85em; color: #555; padding-left: 0.8em; border-left: 3px solid #f59e0b; }}
</style></head>
<body>
<h1>Opus × Opus — essay probes</h1>
<p>Claude Opus as user-LLM pushing Claude Opus as subject, essay mode, agree persona.
Judge: <code>Qwen3.5-397B</code>. (No <code>Opus × Opus</code> abortion runs exist in the benchmark;
gender-rights is the closest moral-contested analogue we have.)</p>
{"".join(sections_html)}
</body></html>""")
print(f"wrote {OUT_HTML}")
