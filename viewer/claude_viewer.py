"""Viewer for Claude 4.5 cross-product: 3 tiers × 3 tiers × 2 topics × 10 reps."""
import json, html
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUT_DIR = ROOT / "output"
OUT_HTML = ROOT / "viewer" / "claude_cross.html"

TIERS = ["haiku", "sonnet", "opus"]
LABEL = {"haiku":"Haiku 4.5", "sonnet":"Sonnet 4.5", "opus":"Opus 4.5"}
TOPICS = [("flatearth","Flat-earth"), ("genderrights","Gender-rights")]

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
          <div class='msg user'><b>User:</b> {u}</div>
          <div class='msg subject'><b>Subject:</b> {s}</div>
        </div>""")
    return "\n".join(out)

sections = []
for topic_key, topic_name in TOPICS:
    # Grid summary table
    grid_rows = ""
    for u in TIERS:
        row = f"<tr><th>u={LABEL[u]}</th>"
        for s in TIERS:
            p = OUT_DIR / f"claude_cross_{topic_key}_u{u}_s{s}.jsonl"
            verdicts = []
            if p.exists():
                with p.open() as f:
                    for line in f:
                        r = json.loads(line)
                        if r.get("verdict") is not None:
                            verdicts.append(r["verdict"])
            agr = sum(1 for v in verdicts if v == "agree")
            n = len(verdicts)
            rate = (agr/n*100) if n else 0
            color = "#fffbeb" if agr > 0 else "#f3f4f6"
            row += f"<td style='background:{color}'><b>{agr}/{n}</b> ({rate:.0f}%)</td>"
        row += "</tr>"
        grid_rows += row
    grid_html = f"""
    <h3>Summary: {topic_name}</h3>
    <table class='summary'>
      <thead><tr><th></th>{''.join(f'<th>s={LABEL[s]}</th>' for s in TIERS)}</tr></thead>
      <tbody>{grid_rows}</tbody>
    </table>
    """

    # Per-cell detailed cards
    cells = []
    for u in TIERS:
        for s in TIERS:
            p = OUT_DIR / f"claude_cross_{topic_key}_u{u}_s{s}.jsonl"
            rows = []
            if p.exists():
                with p.open() as f:
                    for line in f:
                        rec = json.loads(line)
                        rows.append(rec)
            rows.sort(key=lambda r: r.get("run_id", 0))

            rows_html = "\n".join(
                f"""
                <details id='{topic_key}_{u}_{s}_rep{rec['run_id']}' {'class="win"' if rec.get('verdict')=='agree' else ''}>
                  <summary>
                    <span class='pill' style='background:{BADGE.get(rec.get("verdict"), "#000")}'>{rec.get('verdict') or '—'}</span>
                    <b>rep {rec.get('run_id')}</b>
                    {'<span class="star">★ essay produced</span>' if rec.get('verdict')=='agree' else ''}
                  </summary>
                  <div class='transcript'>{render_transcript(rec.get('transcript') or [])}</div>
                  {'<div class="rationale"><b>Judge rationale:</b> ' + html.escape(rec.get('rationale') or '') + '</div>' if rec.get('rationale') else ''}
                </details>
                """ for rec in rows
            )

            cells.append(f"""
            <section class='cell'>
              <h4>user={LABEL[u]} × subject={LABEL[s]}</h4>
              {rows_html}
            </section>
            """)

    sections.append(f"""
    <section class='topic'>
      <h2>{topic_name}</h2>
      {grid_html}
      {''.join(cells)}
    </section>
    """)

OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
OUT_HTML.write_text(f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Claude 4.5 cross viewer</title>
<style>
 body {{ font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", system-ui, sans-serif; max-width: 1200px; margin: 2em auto; padding: 0 1em; color: #111; }}
 h1 {{ font-size: 1.5em; }}
 h2 {{ margin-top: 2em; padding-bottom: 0.3em; border-bottom: 1px solid #ddd; }}
 h3 {{ margin-top: 1.5em; }}
 h4 {{ margin-top: 1.2em; color: #374151; }}
 table.summary {{ border-collapse: collapse; margin: 1em 0; }}
 table.summary th, table.summary td {{ border: 1px solid #ddd; padding: 6px 14px; font-size: 0.95em; text-align: center; }}
 table.summary th {{ background: #f9fafb; }}
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
 section.cell {{ margin: 1em 0; }}
</style></head>
<body>
<h1>Claude 4.5 Cross-Product — 3×3 Essay Probe</h1>
<p>User-LLM × Subject: <code>Haiku 4.5</code>, <code>Sonnet 4.5</code>, <code>Opus 4.5</code>. Judge: <code>Qwen3.5-397B</code>. Agree persona, 10 reps per cell. Rows marked <span class="star">★</span> are where the subject actually produced a persuasive essay for the claim.</p>
{"".join(sections)}
</body></html>""")
print(f"wrote {OUT_HTML}")
