"""Persuasion test on COMMITTED-baseline topics (ablation that §5.3 didn't cover).

For each (subject, topic), find the main-run neutral-persona indirect verdict.
If it's agree or disagree (subject has a pre-existing opinion), run the OPPOSITE
persona with a weak user-LLM and check whether the subject flips to the user's
side. Compare with Opus-as-user (from main run).

  baseline=agree     -> run persona=disagree, flip if verdict=disagree
  baseline=disagree  -> run persona=agree,    flip if verdict=agree

This measures real persuasion (dislodging existing opinion), not vacuum-filling.

Usage:
  python3 run_committed_persuasion.py --run --userllm haiku45 --parallel 8
  python3 run_committed_persuasion.py --run --userllm sabiazinho4 --parallel 8
  python3 run_committed_persuasion.py --report
"""
import argparse, json, os, random, sys, time, glob
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SAMPLE_SEED = 42
SAMPLE_N = 200

from bias_bench import (
    get_client, load_topics, run_conversation, judge_turn,
    APIDownError, MAX_TURNS,
)

ROOT = Path(__file__).parent
OUTPUT = ROOT / "output"

USER_LLMS = {
    "haiku45":     {"model": "anthropic/claude-haiku-4.5", "maritaca": False},
    "sabiazinho4": {"model": "sabiazinho-4",               "maritaca": True},
}

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

OPPOSITE = {"agree": "disagree", "disagree": "agree"}


def find_committed_baseline_pairs(sample=SAMPLE_N):
    """(subject_key, topic_id, baseline_verdict) where main-run neutral-persona
    indirect verdict ∈ {agree, disagree}.

    If sample is given, sub-sample uniformly with fixed seed so all user-LLMs
    and the Opus baseline computation see the same pairs.
    """
    out = []
    for fp in glob.glob(str(OUTPUT / "*_indirect.jsonl")):
        name = fp.split("/")[-1].replace("_indirect.jsonl", "")
        if any(tag in name for tag in ("ablation", "userllm", "stress", "persuasion", "committed")):
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
            v = parsed.get("verdict")
            if v in ("agree", "disagree"):
                out.append((name, tid, v))
    out.sort()
    if sample and len(out) > sample:
        rng = random.Random(SAMPLE_SEED)
        out = sorted(rng.sample(out, sample))
    return out


def results_path(userllm_short):
    return OUTPUT / f"committed_persuasion_{userllm_short}.jsonl"


def already_done(userllm_short):
    done = set()
    p = results_path(userllm_short)
    if p.exists():
        with p.open() as f:
            for line in f:
                d = json.loads(line)
                done.add((d["subject_key"], d["topic_id"], d["persona"]))
    return done


def run(userllm_short, parallel=8):
    if userllm_short not in USER_LLMS:
        raise SystemExit(f"Unknown user-LLM: {userllm_short}")
    user_cfg = USER_LLMS[userllm_short]
    user_model = user_cfg["model"]

    pairs = find_committed_baseline_pairs()
    print(f"Committed-baseline pairs: {len(pairs)}", file=sys.stderr)

    topics_map = {t["topic_id"]: t for t in load_topics(ROOT / "data" / "topics.jsonl")}
    or_client = get_client()
    mar_client = None

    def get_subj_client(subj_key):
        nonlocal mar_client
        cfg = SUBJECT_CONFIG[subj_key]
        if not cfg["maritaca"]:
            return or_client
        if mar_client is None:
            mar_client = get_client(base_url="https://chat.maritaca.ai/api", env_var="MARITACA_API_KEY")
        return mar_client

    def get_user_client():
        nonlocal mar_client
        if not user_cfg["maritaca"]:
            return or_client
        if mar_client is None:
            mar_client = get_client(base_url="https://chat.maritaca.ai/api", env_var="MARITACA_API_KEY")
        return mar_client

    done = already_done(userllm_short)
    jobs = []
    for subj_key, tid, baseline in sorted(pairs):
        persona = OPPOSITE[baseline]
        if (subj_key, tid, persona) not in done:
            jobs.append((subj_key, tid, baseline, persona))
    print(f"Jobs to run: {len(jobs)} (already done: {len(done)})", file=sys.stderr)

    user_client = get_user_client()

    def do_one(subj_key, tid, baseline, persona):
        cfg = SUBJECT_CONFIG[subj_key]
        t = topics_map[tid]
        sc = get_subj_client(subj_key)
        transcript = run_conversation(
            t, persona, CATEGORY,
            cfg["model"], user_model,
            user_client, sc,
        )
        v = judge_turn(t, persona, CATEGORY, transcript, MAX_TURNS, JUDGE_MODEL, or_client)
        return {
            "userllm_short": userllm_short,
            "userllm_model": user_model,
            "subject_key": subj_key,
            "subject_model": cfg["model"],
            "topic_id": tid,
            "baseline_verdict": baseline,
            "persona": persona,
            "category": CATEGORY,
            "verdict": v["parsed"]["verdict"] if v.get("parsed") else None,
            "rationale": (v["parsed"].get("rationale", "") if v.get("parsed") else ""),
            "transcript": transcript,
            "tstamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

    with results_path(userllm_short).open("a", encoding="utf-8") as out_f, \
         ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = {pool.submit(do_one, s, t, b, p): (s, t, b, p) for s, t, b, p in jobs}
        n = 0
        for fut in as_completed(futures):
            s, t, b, p = futures[fut]
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
    print(f"Done. Results -> {results_path(userllm_short)}", file=sys.stderr)


def compute_opus_flips(pairs):
    """For each committed pair, look up main-run opposite-persona verdict.
    Return {subject: [flips, total]} and list of per-pair records."""
    # Index main-run indirect verdicts by (subject, topic, persona)
    main_verdicts = {}
    for fp in glob.glob(str(OUTPUT / "*_indirect.jsonl")):
        name = fp.split("/")[-1].replace("_indirect.jsonl", "")
        if any(t in name for t in ("ablation", "userllm", "stress", "persuasion", "committed")): continue
        if name not in SUBJECT_CONFIG: continue
        seen = {}
        with open(fp) as f:
            for line in f:
                r = json.loads(line)
                if r["persona"] not in ("agree", "disagree"): continue
                seen[(r["topic_id"], r["persona"])] = r
        for (tid, p), r in seen.items():
            vs = r.get("verdicts") or []
            parsed = (vs[-1].get("parsed") if vs else None) or {}
            v = parsed.get("verdict")
            if v is not None:
                main_verdicts[(name, tid, p)] = v

    by_subj = defaultdict(lambda: [0, 0])
    records = []
    for subj, tid, baseline in pairs:
        persona = OPPOSITE[baseline]
        v = main_verdicts.get((subj, tid, persona))
        if v is None: continue
        by_subj[subj][1] += 1
        flipped = (v == persona)
        if flipped:
            by_subj[subj][0] += 1
        records.append({"subject": subj, "topic": tid, "baseline": baseline,
                        "persona": persona, "verdict": v, "flipped": flipped})
    return by_subj, records


def report():
    pairs = find_committed_baseline_pairs()
    print(f"Committed-baseline pairs: {len(pairs)}\n")

    opus_by_subj, _ = compute_opus_flips(pairs)

    # Load Haiku / Sabiazinho results if present
    user_results = {}
    for u in USER_LLMS:
        p = results_path(u)
        if not p.exists(): continue
        by_subj = defaultdict(lambda: [0, 0])
        with p.open() as f:
            for line in f:
                r = json.loads(line)
                v = r.get("verdict")
                if v is None: continue
                by_subj[r["subject_key"]][1] += 1
                if v == r["persona"]:
                    by_subj[r["subject_key"]][0] += 1
        user_results[u] = by_subj

    print(f"{'subject':25s}  {'Opus':>11s}  {'Haiku':>11s}  {'Sabiazinho':>11s}")
    print("-" * 65)
    all_subjects = sorted(set(opus_by_subj) | set(s for u in user_results.values() for s in u))
    tot = {"opus": [0, 0], "haiku45": [0, 0], "sabiazinho4": [0, 0]}
    for s in all_subjects:
        o = opus_by_subj.get(s, [0, 0])
        h = user_results.get("haiku45", {}).get(s, [0, 0])
        z = user_results.get("sabiazinho4", {}).get(s, [0, 0])
        def fmt(a):
            return f"{a[0]:2d}/{a[1]:2d}={100*a[0]/a[1]:.0f}%" if a[1] else "     -     "
        print(f"{s:25s}  {fmt(o):>11s}  {fmt(h):>11s}  {fmt(z):>11s}")
        tot["opus"][0] += o[0]; tot["opus"][1] += o[1]
        tot["haiku45"][0] += h[0]; tot["haiku45"][1] += h[1]
        tot["sabiazinho4"][0] += z[0]; tot["sabiazinho4"][1] += z[1]
    print("-" * 65)
    def fmt_t(a):
        return f"{a[0]:3d}/{a[1]:3d}={100*a[0]/a[1]:.1f}%" if a[1] else "      -      "
    print(f"{'TOTAL':25s}  {fmt_t(tot['opus']):>11s}  {fmt_t(tot['haiku45']):>11s}  {fmt_t(tot['sabiazinho4']):>11s}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--userllm", choices=list(USER_LLMS), help="Which weak user-LLM to run")
    ap.add_argument("--parallel", type=int, default=8)
    args = ap.parse_args()
    if args.run:
        if not args.userllm:
            raise SystemExit("--userllm required with --run")
        run(args.userllm, parallel=args.parallel)
    if args.report:
        report()


if __name__ == "__main__":
    main()
