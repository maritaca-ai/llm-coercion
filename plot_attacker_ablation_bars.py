"""Bar chart: x = topics (+ avg), series = 5 attackers, on fixed Q397B subject.
Same data as Table tab:attacker-ablation."""
import json, os
import matplotlib.pyplot as plt
import numpy as np

ATTS = [('q25-7b','Qwen2.5-7B','#c03030'),
        ('q4b',   'Qwen3.5-4B','#d98c3e'),
        ('q35b',  'Qwen3.5-35B','#4a8cb8'),
        ('q397b', 'Qwen3.5-397B','#1a5c8a'),
        ('opus',  'Opus 4.7','#2ca02c')]
TOPICS = [('creationism','Creationism'),
          ('flatearth','Flat earth'),
          ('climatedenial','Climate denial'),
          ('antivax','Antivax'),
          ('racialiq','Racial IQ'),
          ('holocaustdenial','Holocaust')]
SUBJ = 'q397b'

def rate(att, topic):
    if att == 'opus':
        fn = f'3fam_{topic}_uopus_sqwen.jsonl'
    elif att == 'q397b':
        fn = f'3fam_{topic}_uqwen_sqwen.jsonl'
    else:
        fn = f'qwen_cross_{topic}_u{att}_s{SUBJ}.jsonl'
    fp = f'output/{fn}'
    if not os.path.exists(fp):
        return None
    n = 0; a = 0
    for line in open(fp):
        r = json.loads(line)
        v = r.get('verdict')
        if v is None: continue
        n += 1
        if v == 'agree': a += 1
    return 100*a/n if n else None

fig, ax = plt.subplots(figsize=(10, 4))
xs = np.arange(len(TOPICS) + 1)  # 6 topics + 1 avg
bar_w = 0.16

for i, (att, lbl, color) in enumerate(ATTS):
    rates = [rate(att, t) for t, _ in TOPICS]
    avg = np.mean([r for r in rates if r is not None])
    vals = rates + [avg]
    plot_xs = [xs[j] + (i - 2)*bar_w for j, v in enumerate(vals) if v is not None]
    ys = [v for v in vals if v is not None]
    ax.bar(plot_xs, ys, bar_w, label=lbl, color=color, edgecolor='black', linewidth=0.4)
    for x, y in zip(plot_xs, ys):
        ax.text(x, y + 1.5, f'{int(y)}', ha='center', va='bottom', fontsize=7)

xticklabels = [lbl for _, lbl in TOPICS] + ['avg']
ax.set_xticks(xs)
ax.set_xticklabels(xticklabels, fontsize=9)
# Highlight avg with bold and a separator
for tick in ax.get_xticklabels():
    if tick.get_text() == 'avg':
        tick.set_fontweight('bold')
ax.axvline(len(TOPICS) - 0.5, color='gray', linestyle=':', linewidth=0.8, alpha=0.6)
ax.set_ylabel('essay-production rate (%)', fontsize=10)
ax.set_xlabel('topic', fontsize=10)
ax.set_ylim(0, 110)
ax.set_yticks([0, 25, 50, 75, 100])
ax.grid(True, axis='y', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(title='attacker', loc='upper right', fontsize=8, frameon=True, ncol=2)
ax.set_title('Attacker-ablation: essay-production rate per topic, fixed Qwen3.5-397B subject (last column = mean across topics)', fontsize=10)
plt.tight_layout()
PDF_OUT = 'paper/attacker_ablation_bars.pdf'
plt.savefig(PDF_OUT, bbox_inches='tight')
print(f'saved {PDF_OUT}')
