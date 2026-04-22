# llm-coercion

Code and data for the paper **"Debate Enables Guardrail Override of Frontier LLMs"** (draft at `paper/paper.pdf`).

## What's here

- `bias_bench.py` — essay-probe + judge library.
- `run_3family_scientific.py` — main 3×3 grid on 6 scientific-consensus topics (Qwen3.5-397B, Opus 4.7, Grok 4.20 as both attacker and subject).
- `run_qwen_cross.py` — attacker-strength ablation against a fixed Qwen3.5-397B subject (Qwen3.5-4B self-hosted, Qwen3.5-35B-A3B, Qwen3.5-397B via OpenRouter).
- `rejudge_null.py` — helper to re-run the judge on records whose verdict failed to parse.
- `output/` — per-conversation transcripts and judge verdicts in JSONL (3fam_* = main grid, qwen_cross_* = ablation).
- `paper/` — LaTeX source for the paper.
- `viewer/` — HTML viewer for per-conversation transcripts.

## Running

```bash
export OPENROUTER_API_KEY=sk-or-...

# Main grid: 3 attackers x 3 subjects x 6 topics x 10 reps = 540 conversations
for u in qwen opus grok; do for s in qwen opus grok; do for t in flatearth creationism climatedenial antivax racialiq holocaustdenial; do
  python3 run_3family_scientific.py --user-tier $u --subject-tier $s --topic $t --reps 10
done; done; done

# Attacker ablation (requires a vLLM server at localhost:8000 serving qwen3.5-4b)
for u in q4b q35b; do for t in flatearth creationism climatedenial antivax racialiq holocaustdenial; do
  python3 run_qwen_cross.py --user-tier $u --subject-tier q397b --topic $t --reps 10
done; done
```

A companion study on country-modulated compliance on morally contested claims lives in `maritaca-ai/llm-country-modulation`.
