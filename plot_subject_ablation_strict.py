"""Plot subject-ablation row means using STRICT judge verdicts
(from output_strict/all.jsonl)."""
import json, os
import matplotlib.pyplot as plt
import numpy as np

import sys
LANG = sys.argv[1] if len(sys.argv) > 1 else 'en'

ATTS = [('q25-7b','Qwen2.5-7B','#c03030'),
        ('q4b',   'Qwen3.5-4B','#d98c3e'),
        ('q35b',  'Qwen3.5-35B','#4a8cb8'),
        ('q397b', 'Qwen3.5-397B','#1a5c8a'),
        ('opus',  'Opus 4.7','#2ca02c')]
SUBJS = ['q397b', 'q4b', 'q25-7b']
TOPICS = ['creationism','flatearth','climatedenial','antivax','racialiq','holocaustdenial']

if LANG == 'pt':
    SUBJ_LABEL = ['Qwen3.5-397B\n(fronteira, grande)', 'Qwen3.5-4B\n(fronteira, pequeno)', 'Qwen2.5-7B\n(geração anterior, pequeno)']
    YLABEL = 'produção média da redação (%) — juiz strict'
    XLABEL = 'modelo-alvo'
    LEGEND_TITLE = 'atacante'
    TITLE = 'Atacante × Alvo: média por linha (6 tópicos de consenso científico)'
    PNG_OUT = 'blog/img/subject_ablation.png'
    PDF_OUT = 'paper/subject_ablation.pdf'
else:
    SUBJ_LABEL = ['Qwen3.5-397B\n(frontier, big)', 'Qwen3.5-4B\n(frontier, small)', 'Qwen2.5-7B\n(prev-gen, small)']
    YLABEL = 'essay-production row mean (%) — strict judge'
    XLABEL = 'subject model'
    LEGEND_TITLE = 'attacker'
    TITLE = 'Attacker×Subject row means — strict rejudge (6 scientific-consensus topics)'
    PNG_OUT = 'blog/img/subject_ablation_en.png'
    PDF_OUT = 'paper/subject_ablation.pdf'

# Load strict verdicts keyed by (filename, run_id)
strict = {}
with open('output_strict/all.jsonl') as f:
    for line in f:
        r = json.loads(line)
        strict[(r['source_file'], r['run_id'])] = (r.get('strict') or {}).get('verdict')

def rate(att, subj):
    n = 0; a = 0
    for t in TOPICS:
        if att == 'opus' and subj == 'q397b':
            fn = f'3fam_{t}_uopus_sqwen.jsonl'
        elif att == 'q397b' and subj == 'q397b':
            fn = f'3fam_{t}_uqwen_sqwen.jsonl'
        else:
            fn = f'qwen_cross_{t}_u{att}_s{subj}.jsonl'
        # enumerate records
        fp = f'output/{fn}'
        if not os.path.exists(fp):
            continue
        for line in open(fp):
            r = json.loads(line)
            rid = r.get('run_id')
            v = strict.get((fn, rid))
            if v is None:
                continue
            n += 1
            if v == 'agree':
                a += 1
    return 100*a/n if n else None

fig, ax = plt.subplots(figsize=(8, 5))
xs = np.arange(len(SUBJS))
bar_w = 0.15
for i, (att, lbl, color) in enumerate(ATTS):
    vals = [rate(att, s) for s in SUBJS]
    plot_xs = [xs[j] + (i - 2)*bar_w for j, v in enumerate(vals) if v is not None]
    ys = [v for v in vals if v is not None]
    ax.bar(plot_xs, ys, bar_w, label=lbl, color=color, edgecolor='black', linewidth=0.4)
    for x, y in zip(plot_xs, ys):
        ax.text(x, y + 1.5, f'{int(y)}', ha='center', va='bottom', fontsize=7)

ax.set_xticks(xs)
ax.set_xticklabels(SUBJ_LABEL, fontsize=9)
ax.set_ylabel(YLABEL, fontsize=10)
ax.set_ylim(0, 105)
ax.set_yticks([0, 25, 50, 75, 100])
ax.grid(True, axis='y', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlabel(XLABEL, fontsize=10)
ax.legend(title=LEGEND_TITLE, loc='upper left', fontsize=8, frameon=True)
plt.title(TITLE, fontsize=10)
plt.tight_layout()
if LANG == 'en':
    plt.savefig(PDF_OUT, bbox_inches='tight')
plt.savefig(PNG_OUT, bbox_inches='tight', dpi=220)
print(f'saved {PNG_OUT}')

# Print the actual numbers used
print('\nrow means (strict):')
for att, lbl, _ in ATTS:
    line = f'  {lbl:<14}'
    for s in SUBJS:
        r = rate(att, s)
        line += f'  {s}: {r:.0f}%' if r is not None else f'  {s}: -'
    print(line)
