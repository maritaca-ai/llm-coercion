"""Viewer for rejudged conversations. Loads output_strict_sample/sample.jsonl,
joins with original transcripts, and renders a single HTML page grouped by
verdict shift (drops first, then gains, then unchanged)."""
import json, html
from pathlib import Path

ROOT = Path(__file__).parent.parent
SAMPLE = ROOT / "output_strict_sample" / "sample.jsonl"
OUT_HTML = Path(__file__).parent / "rejudge_sample.html"
OUT_DIR = ROOT / "output"

BADGE = {
    "agree":    "#1a7f3a",
    "disagree": "#b52824",
    "neutral":  "#6b7280",
    "refusal":  "#8b5cf6",
    None:       "#000",
}


def load_transcript(source_file, run_id):
    """Look up the original transcript for (source_file, run_id)."""
    f = OUT_DIR / source_file
    if not f.exists():
        return None, None, None
    for line in f.open():
        r = json.loads(line)
        if r.get("run_id") == run_id:
            return r.get("transcript") or [], r.get("rationale", ""), r
    return None, None, None


def render_turn(turn, per_turn_info):
    idx = turn.get("turn_idx", 0) + 1
    u = html.escape(turn.get("user_message", ""))
    s = html.escape(turn.get("subject_reply", ""))
    # find strict per_turn info for this turn
    pt = next((p for p in per_turn_info if p.get("turn") == idx), None)
    badge = ""
    if pt:
        on_topic = pt.get("on_topic")
        has_essay = pt.get("on_topic_and_against_consensus_essay")
        if has_essay:
            badge = f'<span class="turnbadge hit">ESSAY ON-TOPIC</span>'
        elif on_topic is True:
            badge = f'<span class="turnbadge ontopic">on-topic, no essay</span>'
        elif on_topic is False:
            badge = f'<span class="turnbadge off">off-topic</span>'
        ev = pt.get("evidence", "")
        if ev:
            badge += f'<div class="evidence">&quot;{html.escape(ev[:300])}&quot;</div>'
    return f"""
<div class='turn'>
  <div class='turn-head'>Turn {idx} {badge}</div>
  <div class='msg user'><span class='who'>ATTACKER</span>{u}</div>
  <div class='msg subject'><span class='who'>SUBJECT</span>{s}</div>
</div>"""


def render_record(r):
    old = r.get("verdict")
    strict = r.get("strict", {}) or {}
    new = strict.get("verdict")
    twe = strict.get("turns_with_essay", [])
    per_turn = strict.get("per_turn", [])
    source_file = r.get("source_file", "")
    topic_key = r.get("topic_key", "")
    run_id = r.get("run_id", "?")

    transcript, orig_rat, orig = load_transcript(source_file, run_id)
    if not transcript:
        return f"<details class='rec'><summary>{html.escape(source_file)} run={run_id} — transcript not found</summary></details>"

    user_model = orig.get("user_model", "?") if orig else "?"
    subj_model = orig.get("subject_model", "?") if orig else "?"

    old_c = BADGE.get(old, "#000")
    new_c = BADGE.get(new, "#000")
    shift_arrow = "→" if old == new else "⇒"
    changed_class = "unchanged" if old == new else "changed"

    turns_html = "\n".join(render_turn(t, per_turn) for t in transcript)
    twe_html = ", ".join(str(x) for x in twe) if twe else "—"

    return f"""
<details class='rec {changed_class}'>
  <summary>
    <span class='combo'>{html.escape(topic_key)}</span>
    <span class='pair'>{html.escape(user_model.split("/")[-1])} &rarr; {html.escape(subj_model.split("/")[-1])}</span>
    <span class='runid'>run={run_id}</span>
    <span class='pill' style='background:{old_c}'>{old or "—"}</span>
    <span class='shift'>{shift_arrow}</span>
    <span class='pill' style='background:{new_c}'>{new or "—"}</span>
    <span class='twe'>turns w/ essay: {twe_html}</span>
  </summary>
  <div class='body'>
    <div class='rationales'>
      <div><b>Original rationale:</b> {html.escape(orig_rat)}</div>
      <div><b>Strict rationale:</b> {html.escape(strict.get("rationale", ""))}</div>
      <div class='src'>source: <code>{html.escape(source_file)}</code></div>
    </div>
    <div class='transcript'>{turns_html}</div>
  </div>
</details>"""


def main():
    recs = [json.loads(l) for l in SAMPLE.open()]
    # categorize
    drops = [r for r in recs if r.get("verdict") == "agree" and (r.get("strict") or {}).get("verdict") not in ("agree", None)]
    gains = [r for r in recs if r.get("verdict") != "agree" and (r.get("strict") or {}).get("verdict") == "agree"]
    same_agree = [r for r in recs if r.get("verdict") == "agree" and (r.get("strict") or {}).get("verdict") == "agree"]
    same_other = [r for r in recs if r.get("verdict") != "agree" and (r.get("strict") or {}).get("verdict") != "agree" and (r.get("strict") or {}).get("verdict") is not None]
    parse_fail = [r for r in recs if (r.get("strict") or {}).get("verdict") is None]

    def section(title, entries, color, open_default=False):
        if not entries:
            return ""
        open_attr = "open" if open_default else ""
        body = "\n".join(render_record(r) for r in entries)
        return f"""
<section>
<h2 style='border-left:6px solid {color};padding-left:10px'>{title} &middot; {len(entries)} records</h2>
{body}
</section>"""

    html_out = f"""<!doctype html>
<html><head><meta charset='utf-8'><title>Strict rejudge — 100-sample viewer</title>
<style>
 body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 1100px; margin: 1.5em auto; padding: 0 1em; color: #111; }}
 h1 {{ font-size: 1.5em; }}
 h2 {{ margin-top: 2em; font-size: 1.15em; }}
 details.rec {{ margin: 0.4em 0; border: 1px solid #e5e7eb; border-radius: 6px; padding: 0.5em 0.8em; }}
 details.rec[open] {{ background: #fafafa; }}
 details.rec.changed {{ border-color: #f59e0b; border-width: 2px; }}
 details.rec summary {{ cursor: pointer; font-size: 0.9em; display: flex; gap: 0.5em; align-items: center; flex-wrap: wrap; }}
 .combo {{ font-weight: 700; color: #064e3b; }}
 .pair {{ color: #555; font-family: SF Mono, Menlo, monospace; font-size: 0.85em; }}
 .runid {{ color: #888; font-size: 0.82em; }}
 .shift {{ color: #888; font-size: 1.1em; }}
 .twe {{ color: #666; font-size: 0.82em; margin-left: auto; }}
 .pill {{ display: inline-block; color: white; padding: 1px 8px; border-radius: 10px; font-size: 0.75em; font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em; }}
 .body {{ margin-top: 0.6em; }}
 .rationales {{ background: #fef3c7; padding: 0.6em 0.8em; border-left: 3px solid #f59e0b; font-size: 0.88em; margin-bottom: 0.8em; border-radius: 0 4px 4px 0; }}
 .rationales > div {{ margin: 0.25em 0; }}
 .rationales .src {{ color: #666; font-size: 0.82em; }}
 .transcript {{ border-left: 3px solid #ddd; padding: 0.3em 0.8em; }}
 .turn {{ margin: 0.5em 0; padding: 0.3em 0; border-bottom: 1px dashed #eee; }}
 .turn-head {{ font-size: 0.78em; color: #666; margin-bottom: 0.3em; }}
 .turnbadge {{ display: inline-block; padding: 1px 7px; border-radius: 4px; font-size: 0.7em; font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em; margin-left: 0.5em; }}
 .turnbadge.hit {{ background: #1a7f3a; color: white; }}
 .turnbadge.ontopic {{ background: #fff1e6; color: #b8580c; border: 1px solid #ffc69c; }}
 .turnbadge.off {{ background: #fee2e2; color: #b52824; border: 1px solid #fecaca; }}
 .evidence {{ font-size: 0.82em; color: #666; margin-top: 0.2em; font-style: italic; padding-left: 0.5em; border-left: 2px solid #e5e7eb; }}
 .msg {{ margin: 0.3em 0; white-space: pre-wrap; font-size: 0.88em; padding: 0.5em 0.7em; border-radius: 6px; }}
 .msg .who {{ display: block; font-size: 0.7em; font-weight: 700; letter-spacing: 0.04em; margin-bottom: 0.3em; opacity: 0.8; }}
 .msg.user {{ background: #fff1e6; border: 1px solid #ffc69c; }}
 .msg.user .who {{ color: #b8580c; }}
 .msg.subject {{ background: #f0fdf4; border: 1px solid #b6dcbf; }}
 .msg.subject .who {{ color: #166534; }}
 code {{ background: #f3f4f6; padding: 1px 6px; border-radius: 3px; font-size: 0.85em; }}
</style></head>
<body>
<h1>Strict rejudge — 100-sample viewer</h1>
<p>Out of {len(recs)} rejudged records: <b>{len(drops)}</b> drops (old=agree → new≠agree), <b>{len(gains)}</b> gains (old≠agree → new=agree), <b>{len(same_agree)}</b> confirmed agrees, <b>{len(same_other)}</b> confirmed non-agrees, {len(parse_fail)} parse failures. "Strict" means the new judge required the subject to produce a persuasive on-topic essay arguing for the against-consensus side at some turn. Each record shows the original rationale, the strict rationale, the per-turn strict evaluation (including the turn-level ESSAY ON-TOPIC badge), and the full 5-turn transcript. Click any row to expand.</p>

{section("⬇ Drops — old 'agree' flipped to non-agree (false-positives the strict judge caught)", drops, "#b52824", open_default=True)}
{section("⬆ Gains — old non-agree flipped to 'agree' (essays the original judge missed)", gains, "#1a7f3a", open_default=True)}
{section("✓ Confirmed agrees (old=new=agree)", same_agree, "#1a7f3a")}
{section("✓ Confirmed non-agrees", same_other, "#6b7280")}
{section("⚠ Parse failures", parse_fail, "#f59e0b")}

</body></html>"""
    OUT_HTML.write_text(html_out)
    print(f"wrote {OUT_HTML} ({len(recs)} records)")


if __name__ == "__main__":
    main()
