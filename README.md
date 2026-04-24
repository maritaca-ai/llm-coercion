# llm-persuasion

Code, transcripts, and judge outputs for **"LLM-Based Persuasion Enables Guardrail Override in Frontier LLMs"** ([paper PDF](paper/paper.pdf)).

A frontier assistant LLM, asked directly to write a persuasive essay denying vaccine safety, defending creationism, or denying climate change, refuses. We show that the *same* model, driven only by a short five-turn persuasion prompt and acting as a simulated user, can extract exactly those essays from another frontier-class assistant, including a second copy of itself. Across 9 attacker × subject pairings on 6 scientific-consensus topics (10 reps each, 540 conversations), elicitation is non-zero on 5 of 6 topics and reaches 100% on multiple cells (Opus → Opus on creationism, flat-earth, and climate denial; Qwen → Opus on creationism and flat-earth).

## Read first

- **Blog posts** — [English](https://maritaca-ai.github.io/llm-persuasion/blog/post_en.html) · [Português](https://maritaca-ai.github.io/llm-persuasion/blog/post.html)
- **Per-conversation viewer** — [browse all 540 transcripts](https://maritaca-ai.github.io/llm-persuasion/viewer/) (attacker → subject grid; expand any cell to see the full 5-turn chat plus the judge's rationale)
- **Paper PDF** — [`paper/paper.pdf`](paper/paper.pdf)

## What's in this repo

- `essay_probe.py` — essay-probe runner + judge library.
- `run_3family_scientific.py` — main 3×3 grid (Qwen3.5-397B, Opus 4.7, Grok 4.20 as both attacker and subject) on 6 scientific-consensus topics.
- `run_qwen_cross.py` — attacker- and subject-scale ablation across Qwen2.5-7B, Qwen3.5-4B / 35B-A3B / 397B-A17B, and Opus 4.7.
- `ablate_attacker_refusal.py` — classifies turn-1 attacker messages as complied/refused to separate attacker-side from subject-side refusal.
- `rejudge_null.py` — re-runs the judge on records whose verdict failed to parse.
- `plot_main_grid.py`, `plot_subject_ablation.py`, `plot_turn_ablation.py` — figures for the paper and blog.
- `output/` — per-conversation JSONL (`3fam_*` main grid, `qwen_cross_*` ablation). One record per conversation: full transcript + verdict + turns_with_essay + per_turn_evaluation + judge rationale.
- `viewer/viewer.py` + `viewer/index.html` — static HTML viewer over `output/`.
- `paper/` — LaTeX source + compiled PDF.
- `blog/` — public blog posts (PT/EN) and figures.

## Headline numbers

| | result |
| --- | --- |
| Topics with non-zero elicitation in the main grid | 5 / 6 |
| Opus → Opus cross-topic average | 65% |
| Opus → Opus on creationism / flat-earth / climate denial | 10 / 10 each |
| Holocaust denial across all 9 main-grid combinations | 1 / 90 |
| Weak × weak (Qwen2.5-7B self-persuasion) cross-topic average | 78% |
| Opus → Qwen2.5-7B | 95% |

Persuasion rate scales monotonically with attacker capability (Qwen2.5-7B → Opus 4.7 raises the row mean against Qwen3.5-397B from 10% to 40%). Swapping the subject to the previous Qwen2.5 generation removes the refusal floor for every attacker tested.

## Reproducing

Both the attacker and subject can be any chat-completions endpoint; we use OpenRouter for the closed-weights frontier models and a local vLLM server for the small open-weights ones.

```bash
export OPENROUTER_API_KEY=sk-or-...

# Main grid: 3 attackers × 3 subjects × 6 topics × 10 reps = 540 conversations
for u in qwen opus grok; do for s in qwen opus grok; do for t in flatearth creationism climatedenial antivax racialiq holocaustdenial; do
  python3 run_3family_scientific.py --user-tier $u --subject-tier $s --topic $t --reps 10
done; done; done

# Attacker × subject ablation (small Qwen attackers need a vLLM server at localhost:8000)
for u in q25-7b q4b q35b q397b opus; do for s in q397b q4b q25-7b; do for t in flatearth creationism climatedenial antivax racialiq holocaustdenial; do
  python3 run_qwen_cross.py --user-tier $u --subject-tier $s --topic $t --reps 10
done; done; done

# Attacker complicity classifier
python3 ablate_attacker_refusal.py

# Regenerate viewer + figures
python3 viewer/viewer.py
python3 plot_main_grid.py
python3 plot_subject_ablation.py pt    # PT version
python3 plot_subject_ablation.py en    # EN version
python3 plot_turn_ablation.py pt
python3 plot_turn_ablation.py en
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
