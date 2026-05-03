"""Line chart from Table tab:attacker-role-break (Attacker complicity rates).
X-axis: 6 topics + avg. Series: 6 attackers."""
import matplotlib.pyplot as plt
import numpy as np

ATTS = [
    ('Grok 4.20',         [100,100,100,100,100,100], '#2ca02c', 'v'),
    ('Claude Opus 4.7',   [100,100,100,100,100, 83], '#1a5c8a', 'D'),
    ('Qwen2.5-7B',        [ 90, 80,100,100,100, 90], '#c03030', 'o'),
    ('Qwen3.5-35B-A3B',   [100,100,100,100, 70,  0], '#4a8cb8', '^'),
    ('Qwen3.5-397B-A17B', [100,100,100, 90, 43,  0], '#5599cc', 's'),
    ('Qwen3.5-4B',        [100, 90,100, 50, 70,  0], '#d98c3e', 'P'),
]
TOPICS = ['Creationism','Flat earth','Climate','Antivax','Racial IQ','Holocaust']

fig, ax = plt.subplots(figsize=(9, 4.2))
xs = np.arange(len(TOPICS) + 1)

for lbl, vals, color, marker in ATTS:
    avg = np.mean(vals)
    full = vals + [avg]
    ax.plot(xs[:-1], full[:-1], color=color, marker=marker, markersize=7,
            linewidth=2, label=lbl, alpha=0.9)
    ax.plot(xs[-1], full[-1], color=color, marker=marker, markersize=10,
            markeredgewidth=1.5, markeredgecolor='black', alpha=0.95)

xticklabels = TOPICS + ['mean']
ax.set_xticks(xs)
ax.set_xticklabels(xticklabels, fontsize=9)
for tick in ax.get_xticklabels():
    if tick.get_text() == 'mean': tick.set_fontweight('bold')
ax.axvline(len(TOPICS) - 0.5, color='gray', linestyle=':', linewidth=0.8, alpha=0.6)
ax.set_ylabel('attacker complicity rate (%)', fontsize=10)
ax.set_xlabel('topic', fontsize=10)
ax.set_ylim(-5, 110)
ax.set_yticks([0, 25, 50, 75, 100])
ax.grid(True, axis='y', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(title='attacker', loc='lower left', fontsize=8, frameon=True, ncol=2)
ax.set_title('Attacker complicity rates per topic (rightmost = mean across topics)', fontsize=10)
plt.tight_layout()
plt.savefig('paper/complicity_lines.pdf', bbox_inches='tight')
print('saved paper/complicity_lines.pdf')
