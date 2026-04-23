# llm-persuasion

Code, transcripts, and judge outputs for **"LLM-Based Persuasion Enables Guardrail Override in Frontier LLMs"** ([paper PDF](paper/paper.pdf)).

A frontier assistant LLM, asked directly to write a persuasive essay denying vaccine safety, defending creationism, or denying climate change, refuses. We show that the *same* model — driven only by a short five-turn persuasion prompt and acting as a simulated user — can extract exactly those essays from another frontier-class assistant, including a second copy of itself. Across 9 attacker × subject pairings on 6 scientific-consensus topics (10 reps each, 540 conversations), elicitation is non-zero on 5 of 6 topics and reaches 90–100% on the easier ones (Qwen → Opus on creationism: 10/10).

## Read first

- **Blog posts** — [English](https://maritaca-ai.github.io/llm-persuasion/blog/post_en.html) · [Português](https://maritaca-ai.github.io/llm-persuasion/blog/post.html)
- **Per-conversation viewer** — [browse all 540 transcripts](https://maritaca-ai.github.io/llm-persuasion/viewer/) (attacker→subject grid, expand any cell to see the full 5-turn chat plus the judge's rationale)
- **Paper PDF** — [`paper/paper.pdf`](paper/paper.pdf)

## What's in this repo

- `bias_bench.py` — essay-probe runner + judge library.
- `run_3family_scientific.py` — main 3×3 grid (Qwen3.5-397B, Opus 4.7, Grok 4.20 as both attacker and subject) on 6 scientific-consensus topics.
- `run_qwen_cross.py` — attacker-strength ablation against a fixed Qwen3.5-397B subject (Qwen2.5-7B, Qwen3.5-4B/35B-A3B/397B as attackers).
- `ablate_per_turn.py` — re-judges the existing transcripts after each turn (1 → 5) to track when the subject flips.
- `ablate_attacker_refusal.py` — classifies turn-1 attacker messages as complied/refused to separate attacker-side from subject-side refusal.
- `rejudge_null.py` — re-runs the judge on records whose verdict failed to parse.
- `plot_subject_ablation.py`, `plot_turn_ablation.py` — figures for the paper and blog.
- `output/` — per-conversation JSONL: `3fam_*` (main grid), `qwen_cross_*` (attacker ablation). One record per conversation: full transcript + judge label + rationale.
- `viewer/viewer.py` + `viewer/index.html` — generates the static HTML viewer above from `output/`.
- `paper/` — LaTeX source + compiled PDF.
- `blog/` — public blog posts (PT/EN) and figures.

## Headline numbers

| | result |
| --- | --- |
| Topics with non-zero elicitation in the main grid | 5 / 6 |
| Best attacker × subject combination (creationism, Qwen → Opus) | 10 / 10 |
| Opus → Opus on climate denial | 7 / 10 |
| Holocaust denial across all 9 main-grid combinations | 1 / 90 |
| Frontier × frontier ceiling (row mean) | 37% |
| Weak × weak (Qwen2.5-7B self-attack) row mean | 87% |

The dominant variable is the subject's refusal training, not attacker scale. A 100× parameter gap within the Qwen3.5 family moves rates ±13pp; switching to the older Qwen2.5 generation (subject) moves them +80pp.

## Reproducing

Both the attacker and subject can be any chat-completions endpoint; we use OpenRouter for the closed-weights frontier models and a local vLLM server for the small open-weights ones.

```bash
export OPENROUTER_API_KEY=sk-or-...

# Main grid: 3 attackers × 3 subjects × 6 topics × 10 reps = 540 conversations
for u in qwen opus grok; do for s in qwen opus grok; do for t in flatearth creationism climatedenial antivax racialiq holocaustdenial; do
  python3 run_3family_scientific.py --user-tier $u --subject-tier $s --topic $t --reps 10
done; done; done

# Attacker-strength ablation (small Qwen attackers need a vLLM server at localhost:8000)
for u in q25-7b q4b q35b q397b opus; do for t in flatearth creationism climatedenial antivax racialiq holocaustdenial; do
  python3 run_qwen_cross.py --user-tier $u --subject-tier q397b --topic $t --reps 10
done; done

# Turn-by-turn re-judging (no new conversations, just re-runs the judge at each turn)
python3 ablate_per_turn.py

# Attacker complicity classifier
python3 ablate_attacker_refusal.py

# Regenerate viewer + figures
python3 viewer/viewer.py
python3 plot_subject_ablation.py
python3 plot_turn_ablation.py
```

## Citation

```bibtex
@article{nogueira2026persuasion,
  title  = {LLM-Based Persuasion Enables Guardrail Override in Frontier LLMs},
  author = {Nogueira, Rodrigo and others},
  year   = {2026}
}
```

## Companion work

A study on country-anchored compliance on morally contested claims (does the same subject answer differently when the user states they live in country X?) lives at [`maritaca-ai/llm-country-modulation`](https://github.com/maritaca-ai/llm-country-modulation). The shared judge-stability and topic-coverage analysis is in [`maritaca-ai/llm-bias-bench`](https://github.com/maritaca-ai/llm-bias-bench).
