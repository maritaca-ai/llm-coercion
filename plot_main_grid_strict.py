"""Main 3×3 grid chart using STRICT verdicts. Saves PT and EN versions to blog/img/."""
import json, os
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))

ATTACKERS = ['opus','qwen','grok']  # row order: strongest first
SUBJECTS  = ['opus','qwen','grok']  # col order: most compliant first
TOPICS    = ['creationism','flatearth','climatedenial','antivax','racialiq','holocaustdenial']

LANG_PT = {
    'attacker_label': {'qwen':'Atacante Qwen','opus':'Atacante Opus','grok':'Atacante Grok'},
    'subject_label':  {'qwen':'Alvo Qwen','opus':'Alvo Opus','grok':'Alvo Grok'},
    'topic_label':    {'creationism':'Criacion.','flatearth':'Terra plana','climatedenial':'Neg. clima',
                       'antivax':'Antivacina','racialiq':'QI racial','holocaustdenial':'Neg. Holo.'},
    'ylabel': 'produção da redação (%)',
    'out': 'blog/img/main_grid.png',
}
LANG_EN = {
    'attacker_label': {'qwen':'Qwen attacker','opus':'Opus attacker','grok':'Grok attacker'},
    'subject_label':  {'qwen':'Qwen subject','opus':'Opus subject','grok':'Grok subject'},
    'topic_label':    {'creationism':'Creat.','flatearth':'Flat-earth','climatedenial':'Climate',
                       'antivax':'Antivax','racialiq':'Racial IQ','holocaustdenial':'Holocaust'},
    'ylabel': 'essay-production (%)',
    'out': 'blog/img/main_grid_en.png',
}

# strict verdicts
strict = {}
with open(os.path.join(ROOT, 'output_strict/all.jsonl')) as f:
    for line in f:
        r = json.loads(line)
        strict[(r['source_file'], r['run_id'])] = (r.get('strict') or {}).get('verdict')

def rate(a, s, t):
    fn = f'3fam_{t}_u{a}_s{s}.jsonl'
    fp = os.path.join(ROOT, 'output', fn)
    if not os.path.exists(fp):
        return 0
    n = agree = 0
    for line in open(fp):
        r = json.loads(line)
        v = strict.get((fn, r['run_id']))
        if v is None:
            continue
        n += 1
        if v == 'agree':
            agree += 1
    return 100*agree/n if n else 0

data = {(a,s,t): rate(a,s,t) for a in ATTACKERS for s in SUBJECTS for t in TOPICS}

def save(lang):
    fig, axes = plt.subplots(3, 3, figsize=(11, 7.5), sharey=True, sharex=True)
    for row, a in enumerate(ATTACKERS):
        for col, s in enumerate(SUBJECTS):
            ax = axes[row][col]
            vals = [data[(a,s,t)] for t in TOPICS]
            labels = [lang['topic_label'][t] for t in TOPICS]
            colors = ['#1a5c8a' if v >= 80 else '#4a8cb8' if v >= 50 else '#d98c3e' if v >= 20 else '#c03030' for v in vals]
            xs = list(range(len(TOPICS)))
            ax.bar(xs, vals, color=colors, edgecolor='black', linewidth=0.4)
            for x, v in zip(xs, vals):
                ax.text(x, v + 2, f'{int(v)}', ha='center', va='bottom', fontsize=7)
            ax.set_xticks(xs)
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
            ax.set_ylim(0, 115)
            ax.set_yticks([0, 25, 50, 75, 100])
            ax.grid(True, axis='y', alpha=0.3)
            if col == 0:
                ax.set_ylabel(f'{lang["attacker_label"][a]}\n\n{lang["ylabel"]}', fontsize=9)
            if row == 0:
                ax.set_title(lang['subject_label'][s], fontsize=10)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
    plt.tight_layout()
    out = os.path.join(ROOT, lang['out'])
    plt.savefig(out, bbox_inches='tight', dpi=220)
    plt.close(fig)
    print(f'saved {out}')

save(LANG_PT)
save(LANG_EN)
