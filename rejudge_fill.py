"""Fill-in rejudge: rejudge only records not yet in output_strict/all.jsonl."""
import json, os, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))
from rejudge_strict import TOPIC_META, rejudge_record

ROOT = Path(__file__).parent
OUT_PATH = ROOT / "output_strict" / "all.jsonl"
INPUT_DIR = ROOT / "output"


def main():
    done = set()
    with OUT_PATH.open() as f:
        for line in f:
            r = json.loads(line)
            done.add((r.get("source_file"), r.get("run_id")))

    # enumerate all records now
    patterns = ["3fam_*.jsonl", "qwen_cross_*.jsonl"]
    missing = []
    for pat in patterns:
        for fp in sorted(INPUT_DIR.glob(pat)):
            stem = fp.stem
            parts = stem.split("_")
            topic_key = parts[2] if stem.startswith("qwen_cross_") else parts[1]
            if topic_key not in TOPIC_META:
                continue
            for line in fp.open():
                rec = json.loads(line)
                if (fp.name, rec.get("run_id")) in done:
                    continue
                missing.append((fp.name, topic_key, rec))

    print(f"filling in {len(missing)} missing records")
    or_key = os.environ["OPENROUTER_API_KEY"]
    client = OpenAI(api_key=or_key, base_url="https://openrouter.ai/api/v1", timeout=600.0)

    done_count = 0
    with ThreadPoolExecutor(max_workers=6) as ex, OUT_PATH.open("a") as fout:
        futures = {ex.submit(rejudge_record, client, tk, rec): (fn, tk, rec) for fn, tk, rec in missing}
        for fut in as_completed(futures):
            fn, tk, orig = futures[fut]
            try:
                result = fut.result()
            except Exception as e:
                print(f"  err {fn}#{orig.get('run_id')}: {e}", file=sys.stderr)
                result = dict(orig); result["strict"] = {"verdict": None, "error": str(e)}
            result["source_file"] = fn
            result["topic_key"] = tk
            stripped = {k: v for k, v in result.items() if k != "transcript"}
            fout.write(json.dumps(stripped) + "\n"); fout.flush()
            done_count += 1
            if done_count % 10 == 0:
                print(f"  {done_count}/{len(missing)}")
    print(f"done. appended {done_count}")


if __name__ == "__main__":
    main()
