"""Per-turn essay-production curves using STRICT rejudge's `turns_with_essay`.
3 subplots side by side: per topic, per attacker, per subject (main 3fam grid).

Cumulative rate at turn N = fraction of conversations where min(turns_with_essay) <= N.
"""
import json, os, sys
from collections import defaultdict
import matplotlib.pyplot as plt

LANG = sys.argv[1] if len(sys.argv) > 1 else 'en'

ROOT = os.path.dirname(os.path.abspath(__file__))

ATTACKERS = ['opus','qwen','grok']   # legend order: highest→lowest on y-axis
SUBJECTS  = ['opus','qwen','grok']
TOPICS    = ['creationism','flatearth','climatedenial','racialiq','antivax','holocaustdenial']  # by turn-5 height

if LANG == 'pt':
    TOPIC_LABEL = {'creationism':'Criacionismo','flatearth':'Terra plana','climatedenial':'Neg. climática',
                   'antivax':'Antivacina','racialiq':'QI racial','holocaustdenial':'Neg. Holocausto'}
    MODEL_LABEL = {'qwen':'Qwen3.5-397B','opus':'Opus 4.7','grok':'Grok 4.20'}
    XLABEL = 'turno'
    YLABEL = 'conversas com redação produzida até o turno N (%)'
    TITLES = ['por tópico', 'por atacante', 'por alvo']
    LEGEND_TITLES = ['tópico', 'atacante', 'alvo']
    SUPTITLE = 'Taxa cumulativa de produção da redação por turno'
    PNG_OUT = 'blog/img/turn_ablation_all.png'
    PDF_OUT = 'paper/turn_ablation.pdf'
else:
    TOPIC_LABEL = {'creationism':'Creationism','flatearth':'Flat earth','climatedenial':'Climate denial',
                   'antivax':'Antivax','racialiq':'Racial IQ','holocaustdenial':'Holocaust denial'}
    MODEL_LABEL = {'qwen':'Qwen3.5-397B','opus':'Opus 4.7','grok':'Grok 4.20'}
    XLABEL = 'turn'
    YLABEL = 'conversations with essay produced by turn N (%)'
    TITLES = ['by topic', 'by attacker', 'by subject']
    LEGEND_TITLES = ['topic', 'attacker', 'subject']
    SUPTITLE = 'Cumulative essay-production rate by turn'
    PNG_OUT = 'blog/img/turn_ablation_all_en.png'
    PDF_OUT = 'paper/turn_ablation.pdf'

TOPIC_COLOR = {
    'creationism':'#1f77b4','flatearth':'#ff7f0e','climatedenial':'#2ca02c',
    'antivax':'#d62728','racialiq':'#9467bd','holocaustdenial':'#8c564b',
}
MODEL_COLOR = {'qwen':'#c03030','opus':'#2ca02c','grok':'#4a8cb8'}

# Load strict turns_with_essay, keyed by (source_file, run_id)
twe = {}
with open(os.path.join(ROOT, 'output_strict/all.jsonl')) as f:
    for line in f:
        r = json.loads(line)
        s = r.get('strict') or {}
        twe[(r['source_file'], r['run_id'])] = s.get('turns_with_essay') or []

# Gather all 3fam conversations with their first-essay-turn
# entry: (topic, attacker, subject, first_essay_turn_or_None)
entries = []
for a in ATTACKERS:
    for s in SUBJECTS:
        for t in TOPICS:
            fn = f'3fam_{t}_u{a}_s{s}.jsonl'
            fp = os.path.join(ROOT, 'output', fn)
            if not os.path.exists(fp):
                continue
            for line in open(fp):
                r = json.loads(line)
                rid = r['run_id']
                turns = twe.get((fn, rid))
                if turns is None:
                    continue
                first = min(turns) if turns else None
                entries.append((t, a, s, first))

TURNS = [1, 2, 3, 4, 5]

def cumulative_by_key(group_by):
    """For each unique key value, compute cumulative rate at turn 1..5."""
    groups = defaultdict(list)
    for e in entries:
        k = e[group_by]
        groups[k].append(e[3])
    result = {}
    for k, firsts in groups.items():
        ys = []
        n = len(firsts)
        for N in TURNS:
            hits = sum(1 for f in firsts if f is not None and f <= N)
            ys.append(100 * hits / n if n else 0)
        result[k] = ys
    return result

per_topic   = cumulative_by_key(0)
per_attacker = cumulative_by_key(1)
per_subject = cumulative_by_key(2)

fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
for ax, title in zip(axes, TITLES):
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(XLABEL, fontsize=9)
    ax.set_xticks(TURNS)
    ax.set_ylim(-3, 85)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
axes[0].set_ylabel(YLABEL, fontsize=9)

# subplot 1: per topic
for t in TOPICS:
    if t in per_topic:
        axes[0].plot(TURNS, per_topic[t], marker='o', label=TOPIC_LABEL[t], color=TOPIC_COLOR[t], linewidth=1.6)
axes[0].legend(title=LEGEND_TITLES[0], fontsize=7, loc='upper left', frameon=True)

# subplot 2: per attacker
for a in ATTACKERS:
    if a in per_attacker:
        axes[1].plot(TURNS, per_attacker[a], marker='s', label=MODEL_LABEL[a], color=MODEL_COLOR[a], linewidth=1.8)
axes[1].legend(title=LEGEND_TITLES[1], fontsize=8, loc='upper left', frameon=True)

# subplot 3: per subject
for s in SUBJECTS:
    if s in per_subject:
        axes[2].plot(TURNS, per_subject[s], marker='^', label=MODEL_LABEL[s], color=MODEL_COLOR[s], linewidth=1.8)
axes[2].legend(title=LEGEND_TITLES[2], fontsize=8, loc='upper left', frameon=True)

plt.suptitle(SUPTITLE, fontsize=11, y=1.02)
plt.tight_layout()
if LANG == 'en':
    plt.savefig(PDF_OUT, bbox_inches='tight')
plt.savefig(PNG_OUT, bbox_inches='tight', dpi=220)
print(f'saved {PNG_OUT}')

# Report numbers
print('\nper-topic cumulative (by turn 1..5):')
for t in TOPICS:
    print(f'  {t:<18}', per_topic.get(t, 'n/a'))
print('\nper-attacker:')
for a in ATTACKERS: print(f'  {a:<8}', per_attacker.get(a))
print('\nper-subject:')
for s in SUBJECTS: print(f'  {s:<8}', per_subject.get(s))
