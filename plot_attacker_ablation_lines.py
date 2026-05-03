"""Line chart: x = topics (+avg), series = 5 attackers, fixed Q397B subject.
Same data as Table tab:attacker-ablation."""
import json, os
import matplotlib.pyplot as plt
import numpy as np

ATTS = [('q25-7b','Qwen2.5-7B','#c03030','o'),
        ('q4b',   'Qwen3.5-4B','#d98c3e','s'),
        ('q35b',  'Qwen3.5-35B','#4a8cb8','^'),
        ('q397b', 'Qwen3.5-397B','#1a5c8a','D'),
        ('opus',  'Opus 4.7','#2ca02c','v')]
TOPICS = [('creationism','Creationism'),
          ('flatearth','Flat earth'),
          ('climatedenial','Climate denial'),
          ('antivax','Antivax'),
          ('racialiq','Racial IQ'),
          ('holocaustdenial','Holocaust')]
SUBJ = 'q397b'

def rate(att, topic):
    if att == 'opus': fn = f'3fam_{topic}_uopus_sqwen.jsonl'
    elif att == 'q397b': fn = f'3fam_{topic}_uqwen_sqwen.jsonl'
    else: fn = f'qwen_cross_{topic}_u{att}_s{SUBJ}.jsonl'
    fp = f'output/{fn}'
    if not os.path.exists(fp): return None
    n = 0; a = 0
    for line in open(fp):
        r = json.loads(line); v = r.get('verdict')
        if v is None: continue
        n += 1
        if v == 'agree': a += 1
    return 100*a/n if n else None

fig, ax = plt.subplots(figsize=(9, 4.2))
xs = np.arange(len(TOPICS) + 1)

for att, lbl, color, marker in ATTS:
    rates = [rate(att, t) for t, _ in TOPICS]
    avg = np.mean([r for r in rates if r is not None])
    vals = rates + [avg]
    valid = [(x, v) for x, v in zip(xs, vals) if v is not None]
    line_x = [x for x, _ in valid]; line_y = [v for _, v in valid]
    ax.plot(line_x[:-1], line_y[:-1], color=color, marker=marker, markersize=7,
            linewidth=2, label=lbl, alpha=0.9)
    # avg point: marker without connecting line to last topic
    ax.plot(line_x[-1], line_y[-1], color=color, marker=marker, markersize=10,
            markeredgewidth=1.5, markeredgecolor='black', alpha=0.95)

xticklabels = [lbl for _, lbl in TOPICS] + ['mean']
ax.set_xticks(xs)
ax.set_xticklabels(xticklabels, fontsize=9)
for tick in ax.get_xticklabels():
    if tick.get_text() == 'mean': tick.set_fontweight('bold')
ax.axvline(len(TOPICS) - 0.5, color='gray', linestyle=':', linewidth=0.8, alpha=0.6)
ax.set_ylabel('essay-production rate (%)', fontsize=10)
ax.set_xlabel('topic', fontsize=10)
ax.set_ylim(0, 105)
ax.set_yticks([0, 25, 50, 75, 100])
ax.grid(True, axis='y', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(title='attacker', loc='upper right', fontsize=8, frameon=True, ncol=2)
ax.set_title('Attacker-ablation: essay-production rate per topic, fixed Qwen3.5-397B subject (rightmost = mean across topics)', fontsize=10)
plt.tight_layout()
plt.savefig('paper/attacker_ablation_lines.pdf', bbox_inches='tight')
print('saved paper/attacker_ablation_lines.pdf')
