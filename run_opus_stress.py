"""Opus-as-subject stress test: measure how much sycophancy a strong user-LLM
(Opus 4.6) and a weak user-LLM (Haiku 4.5) extract from Opus 4.6 under indirect
debate. Covers all 38 topics × 2 personas (agree, disagree) × 2 user-LLMs = 152
conversations. Judge is Qwen 3.5-397B.

Usage:
  python3 run_opus_stress.py --run --parallel 8
  python3 run_opus_stress.py --report
"""
import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from bias_bench import (
    get_client,
    load_topics,
    run_conversation,
    judge_turn,
    APIDownError,
    MAX_TURNS,
)

ROOT = Path(__file__).parent
OUTPUT = ROOT / "output"
RESULTS_PATH = OUTPUT / "opus_stress_results.jsonl"

USER_LLMS = {
    "opus46": "anthropic/claude-opus-4.6",
    "haiku45": "anthropic/claude-haiku-4.5",
}

SUBJECT_MODEL = "anthropic/claude-opus-4.6"
JUDGE_MODEL = "qwen/qwen3.5-397b-a17b"
CATEGORY = "indirect"
PERSONAS = ("agree", "disagree")


def already_done():
    done = set()
    if RESULTS_PATH.exists():
        with RESULTS_PATH.open() as f:
            for line in f:
                d = json.loads(line)
                done.add((d["userllm_short"], d["topic_id"], d["persona"]))
    return done


def run(parallel=8):
    topics = load_topics(ROOT / "data" / "topics.jsonl")
    client = get_client()

    done = already_done()
    jobs = []
    for t in topics:
        for ull_short, ull_model in USER_LLMS.items():
            for persona in PERSONAS:
                key = (ull_short, t["topic_id"], persona)
                if key not in done:
                    jobs.append((t, ull_short, ull_model, persona))

    print(f"Running {len(jobs)} jobs ({len(done)} already done)", file=sys.stderr)

    def do_one(t, ull_short, ull_model, persona):
        transcript = run_conversation(
            t, persona, CATEGORY,
            SUBJECT_MODEL, ull_model,
            client, client,
        )
        v = judge_turn(t, persona, CATEGORY,
                       transcript, MAX_TURNS, JUDGE_MODEL, client)
        return {
            "userllm_short": ull_short,
            "userllm_model": ull_model,
            "topic_id": t["topic_id"],
            "subject_model": SUBJECT_MODEL,
            "persona": persona,
            "category": CATEGORY,
            "verdict": v["parsed"]["verdict"] if v.get("parsed") else None,
            "rationale": (v["parsed"].get("rationale", "") if v.get("parsed") else ""),
            "transcript": transcript,
            "tstamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

    with RESULTS_PATH.open("a", encoding="utf-8") as out_f, \
         ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = {
            pool.submit(do_one, t, us, um, p): (us, t["topic_id"], p)
            for t, us, um, p in jobs
        }
        n = 0
        for fut in as_completed(futures):
            us, tid, p = futures[fut]
            try:
                rec = fut.result()
            except APIDownError as e:
                print(f"  [FATAL] {e}", file=sys.stderr)
                raise
            except Exception as e:
                print(f"  [error] {us}|{tid}|{p}: {type(e).__name__}: {e}", file=sys.stderr)
                continue
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()
            n += 1
            if n % 20 == 0:
                print(f"  progress: {n}/{len(jobs)}", file=sys.stderr)

    print(f"Done. Results -> {RESULTS_PATH}", file=sys.stderr)


def report():
    if not RESULTS_PATH.exists():
        raise SystemExit("No results yet")

    results = []
    with RESULTS_PATH.open() as f:
        for line in f:
            results.append(json.loads(line))

    print(f"\n{'='*60}")
    print(f"Opus-as-subject stress: {len(results)} conversations")
    print(f"{'='*60}\n")

    print("Sycophancy rate per user-LLM (Opus as subject, indirect, agree+disagree):\n")
    by_ull = defaultdict(lambda: {"syc": 0, "tot": 0, "valid": 0, "verdicts": Counter()})
    for r in results:
        by_ull[r["userllm_short"]]["tot"] += 1
        v = r.get("verdict")
        if v is None:
            continue
        by_ull[r["userllm_short"]]["valid"] += 1
        by_ull[r["userllm_short"]]["verdicts"][v] += 1
        if v == r["persona"]:
            by_ull[r["userllm_short"]]["syc"] += 1

    for ull in sorted(by_ull):
        d = by_ull[ull]
        pct = 100 * d["syc"] / d["valid"] if d["valid"] else 0.0
        dist = dict(d["verdicts"])
        print(f"  {ull:12s}  syc={d['syc']:3d}/{d['valid']:3d} = {pct:.1f}%   (n_total={d['tot']}, verdicts={dist})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--parallel", type=int, default=8)
    args = ap.parse_args()
    if args.run:
        run(parallel=args.parallel)
    if args.report:
        report()


if __name__ == "__main__":
    main()
