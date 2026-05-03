"""Same data as plot_subject_ablation.py but as line series.
Y-axis: 5 attackers (categorical, ordered weakest -> strongest).
X-axis: average essay-production rate (%).
3 series: one per subject."""
import json, os, sys
import matplotlib.pyplot as plt
import numpy as np

LANG = sys.argv[1] if len(sys.argv) > 1 else 'en'

ATTS = [('q25-7b','Qwen2.5-7B'),
        ('q4b',   'Qwen3.5-4B'),
        ('q35b',  'Qwen3.5-35B'),
        ('q397b', 'Qwen3.5-397B'),
        ('opus',  'Opus 4.7')]
SUBJS = [('q397b','Qwen3.5-397B','#1a5c8a','o'),
         ('q4b',   'Qwen3.5-4B','#d98c3e','s'),
         ('q25-7b','Qwen2.5-7B','#c03030','^')]
TOPICS = ['creationism','flatearth','climatedenial','antivax','racialiq','holocaustdenial']

def rate(att, subj):
    n = 0; a = 0
    for t in TOPICS:
        if att == 'opus' and subj == 'q397b':
            fn = f'3fam_{t}_uopus_sqwen.jsonl'
        elif att == 'q397b' and subj == 'q397b':
            fn = f'3fam_{t}_uqwen_sqwen.jsonl'
        else:
            fn = f'qwen_cross_{t}_u{att}_s{subj}.jsonl'
        fp = f'output/{fn}'
        if not os.path.exists(fp):
            continue
        for line in open(fp):
            r = json.loads(line)
            v = r.get('verdict')
            if v is None: continue
            n += 1
            if v == 'agree': a += 1
    return 100*a/n if n else None

fig, ax = plt.subplots(figsize=(7.5, 4))
xs = np.arange(len(ATTS))

for subj, lbl, color, marker in SUBJS:
    vals = [rate(att, subj) for att, _ in ATTS]
    valid = [(x, v) for x, v in zip(xs, vals) if v is not None]
    line_x = [x for x, _ in valid]
    line_y = [v for _, v in valid]
    ax.plot(line_x, line_y, color=color, marker=marker, markersize=8,
            linewidth=2, label=lbl, alpha=0.9)
    for x, v in valid:
        ax.annotate(f'{int(v)}', xy=(x, v), xytext=(0, 7),
                    textcoords='offset points', fontsize=8, color=color, ha='center')

ax.set_xticks(xs)
ax.set_xticklabels([lbl for _, lbl in ATTS], fontsize=9)
ax.set_ylabel('average essay-production rate (%)' if LANG == 'en' else 'produção média da redação (%)', fontsize=10)
ax.set_xlabel('attacker' if LANG == 'en' else 'atacante', fontsize=10)
ax.set_ylim(0, 105)
ax.set_yticks([0, 25, 50, 75, 100])
ax.grid(True, axis='y', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(title='subject' if LANG == 'en' else 'modelo-alvo', loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=True)
ax.set_title('Average essay-production rate by attacker, per subject (6 sci. consensus topics)' if LANG == 'en' else 'Taxa média de produção, por atacante e por subject (6 tópicos)', fontsize=10)
plt.tight_layout()

PDF_OUT = 'paper/subject_ablation_lines.pdf'
plt.savefig(PDF_OUT, bbox_inches='tight')
print(f'saved {PDF_OUT}')
