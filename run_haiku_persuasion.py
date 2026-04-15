"""Haiku 4.5 as user-LLM against all 13 subjects on neutral-baseline topics.
For each subject, topics are those where the main-run neutral-persona verdict
was 'neutral' (i.e., subject had no pre-existing opinion). We test agree and
disagree personas = 110 conversations. Compare with Opus-as-user (main run).

Usage:
  python3 run_haiku_persuasion.py --run --parallel 8
  python3 run_haiku_persuasion.py --report
"""
import argparse, json, os, sys, time, glob
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from bias_bench import (
    get_client, load_topics, run_conversation, judge_turn,
    APIDownError, MAX_TURNS,
)

ROOT = Path(__file__).parent
OUTPUT = ROOT / "output"
RESULTS_PATH = OUTPUT / "haiku_persuasion_results.jsonl"

USER_LLM_MODEL = "anthropic/claude-haiku-4.5"
USER_LLM_SHORT = "haiku45"
JUDGE_MODEL = "qwen/qwen3.5-397b-a17b"

SUBJECT_CONFIG = {
    "claudeopus46":           {"model": "anthropic/claude-opus-4.6",            "maritaca": False},
    "gpt54":                  {"model": "openai/gpt-5.4",                       "maritaca": False},
    "grok420":                {"model": "x-ai/grok-4.20",                       "maritaca": False},
    "gemini31propreview":     {"model": "google/gemini-3.1-pro-preview",        "maritaca": False},
    "qwen35397ba17b":         {"model": "qwen/qwen3.5-397b-a17b",               "maritaca": False},
    "kimik2thinking":         {"model": "moonshotai/kimi-k2-thinking",          "maritaca": False},
    "mistrallarge2512":       {"model": "mistralai/mistral-large-2512",         "maritaca": False},
    "llama4maverick":         {"model": "meta-llama/llama-4-maverick",          "maritaca": False},
    "claudehaiku45":          {"model": "anthropic/claude-haiku-4.5",           "maritaca": False},
    "gpt54mini":              {"model": "openai/gpt-5.4-mini",                  "maritaca": False},
    "gemini31flashlitepre":   {"model": "google/gemini-3.1-flash-lite-preview", "maritaca": False},
    "sabia4":                 {"model": "sabia-4",                              "maritaca": True},
    "sabiazinho4":            {"model": "sabiazinho-4",                         "maritaca": True},
}

CATEGORY = "indirect"
PERSONAS = ("agree", "disagree")


def find_neutral_baseline_triples():
    """(subject_key, topic_id) where main-run verdict=neutral under neutral persona, indirect."""
    out = set()
    for fp in glob.glob(str(OUTPUT / "*_indirect.jsonl")):
        name = fp.split("/")[-1].replace("_indirect.jsonl", "")
        if any(tag in name for tag in ("ablation", "userllm", "stress", "persuasion")):
            continue
        if name not in SUBJECT_CONFIG:
            continue
        seen = {}
        with open(fp) as f:
            for line in f:
                r = json.loads(line)
                if r["persona"] != "neutral":
                    continue
                seen[r["topic_id"]] = r
        for tid, r in seen.items():
            vs = r.get("verdicts") or []
            parsed = (vs[-1].get("parsed") if vs else None) or {}
            if parsed.get("verdict") == "neutral":
                out.add((name, tid))
    return out


def already_done():
    done = set()
    if RESULTS_PATH.exists():
        with RESULTS_PATH.open() as f:
            for line in f:
                d = json.loads(line)
                done.add((d["subject_key"], d["topic_id"], d["persona"]))
    return done


def run(parallel=8):
    triples = find_neutral_baseline_triples()
    print(f"Neutral-baseline triples: {len(triples)}", file=sys.stderr)

    topics_map = {t["topic_id"]: t for t in load_topics(ROOT / "data" / "topics.jsonl")}
    or_client = get_client()
    mar_client = None

    done = already_done()
    jobs = []
    for subj_key, tid in sorted(triples):
        for persona in PERSONAS:
            if (subj_key, tid, persona) not in done:
                jobs.append((subj_key, tid, persona))
    print(f"Jobs to run: {len(jobs)} (already done: {len(done)})", file=sys.stderr)

    def get_subj_client(subj_key):
        nonlocal mar_client
        cfg = SUBJECT_CONFIG[subj_key]
        if not cfg["maritaca"]:
            return or_client
        if mar_client is None:
            mar_client = get_client(base_url="https://chat.maritaca.ai/api", env_var="MARITACA_API_KEY")
        return mar_client

    def do_one(subj_key, tid, persona):
        cfg = SUBJECT_CONFIG[subj_key]
        t = topics_map[tid]
        sc = get_subj_client(subj_key)
        transcript = run_conversation(
            t, persona, CATEGORY,
            cfg["model"], USER_LLM_MODEL,
            or_client, sc,
        )
        v = judge_turn(t, persona, CATEGORY, transcript, MAX_TURNS, JUDGE_MODEL, or_client)
        return {
            "userllm_short": USER_LLM_SHORT,
            "userllm_model": USER_LLM_MODEL,
            "subject_key": subj_key,
            "subject_model": cfg["model"],
            "topic_id": tid,
            "persona": persona,
            "category": CATEGORY,
            "verdict": v["parsed"]["verdict"] if v.get("parsed") else None,
            "rationale": (v["parsed"].get("rationale", "") if v.get("parsed") else ""),
            "transcript": transcript,
            "tstamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

    with RESULTS_PATH.open("a", encoding="utf-8") as out_f, \
         ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = {pool.submit(do_one, s, t, p): (s, t, p) for s, t, p in jobs}
        n = 0
        for fut in as_completed(futures):
            s, t, p = futures[fut]
            try:
                rec = fut.result()
            except APIDownError as e:
                print(f"  [FATAL] {e}", file=sys.stderr); raise
            except Exception as e:
                print(f"  [error] {s}|{t}|{p}: {type(e).__name__}: {e}", file=sys.stderr)
                continue
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()
            n += 1
            if n % 10 == 0:
                print(f"  progress: {n}/{len(jobs)}", file=sys.stderr)
    print(f"Done. Results -> {RESULTS_PATH}", file=sys.stderr)


def report():
    if not RESULTS_PATH.exists():
        raise SystemExit("No results yet")
    rows = []
    with RESULTS_PATH.open() as f:
        for l in f: rows.append(json.loads(l))

    # Haiku-as-user persuasion
    flip = tot = 0
    by_subj = defaultdict(lambda: [0, 0])
    for r in rows:
        v = r.get("verdict")
        if v is None: continue
        tot += 1; by_subj[r["subject_key"]][1] += 1
        if v == r["persona"]:
            flip += 1; by_subj[r["subject_key"]][0] += 1

    print(f"\nHaiku-as-user persuasion on neutral-baseline topics (indirect):")
    print(f"  overall: {flip}/{tot} = {100*flip/tot:.1f}%\n")
    print(f"  {'subject':25s}  Haiku  vs  Opus (from main run)")

    # Opus baseline from main run
    neutral = set()
    for fp in glob.glob(str(OUTPUT / "*_indirect.jsonl")):
        name = fp.split("/")[-1].replace("_indirect.jsonl", "")
        if any(t in name for t in ("ablation", "userllm", "stress", "persuasion")): continue
        if name not in SUBJECT_CONFIG: continue
        seen = {}
        with open(fp) as f:
            for line in f:
                r = json.loads(line)
                if r["persona"] != "neutral": continue
                seen[r["topic_id"]] = r
        for tid, r in seen.items():
            vs = r.get("verdicts") or []
            parsed = (vs[-1].get("parsed") if vs else None) or {}
            if parsed.get("verdict") == "neutral":
                neutral.add((name, tid))

    opus_by_subj = defaultdict(lambda: [0, 0])
    for fp in glob.glob(str(OUTPUT / "*_indirect.jsonl")):
        name = fp.split("/")[-1].replace("_indirect.jsonl", "")
        if any(t in name for t in ("ablation","userllm","stress","persuasion")): continue
        if name not in SUBJECT_CONFIG: continue
        seen = {}
        with open(fp) as f:
            for line in f:
                r = json.loads(line)
                if r["persona"] not in ("agree","disagree"): continue
                seen[(r["topic_id"], r["persona"])] = r
        for (tid, p), r in seen.items():
            if (name, tid) not in neutral: continue
            vs = r.get("verdicts") or []
            parsed = (vs[-1].get("parsed") if vs else None) or {}
            v = parsed.get("verdict")
            if v is None: continue
            opus_by_subj[name][1] += 1
            if v == r["persona"]:
                opus_by_subj[name][0] += 1

    opus_total = sum(v[0] for v in opus_by_subj.values()), sum(v[1] for v in opus_by_subj.values())
    for subj_key in sorted(by_subj, key=lambda k: -by_subj[k][1]):
        h = by_subj[subj_key]; o = opus_by_subj.get(subj_key, [0,0])
        hp = 100*h[0]/h[1] if h[1] else 0
        op = 100*o[0]/o[1] if o[1] else 0
        print(f"  {subj_key:25s}  {h[0]:2d}/{h[1]:2d}={hp:5.1f}%   vs   {o[0]:2d}/{o[1]:2d}={op:5.1f}%")
    print(f"  {'TOTAL':25s}  {flip:3d}/{tot:3d}={100*flip/tot:5.1f}%   vs   {opus_total[0]:3d}/{opus_total[1]:3d}={100*opus_total[0]/opus_total[1]:5.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--parallel", type=int, default=8)
    args = ap.parse_args()
    if args.run: run(parallel=args.parallel)
    if args.report: report()


if __name__ == "__main__":
    main()
