"""Zero-shot OpenDecider on two Doom checks, against Laya's numbers on the same rows.
1. Movement: regret on the outcome-labelled fight rollouts (validation runs of laya-combat-movement-v1).
2. Weapon: share of attack weapon questions answered with a weapon that cannot hit, on logged questions.

Needs the `opendecider` package, which scripts/setup.sh does not install. Run from ext/doomLaya-dev:
    .venv/bin/python ../../doom/opendecider_check.py [MODEL] [RUN_GLOB]
RUN_GLOB selects the runs whose weapon questions are replayed (default: the cm1 batch in runs/)."""
import collections, glob, json, random, sys, time
from opendecider import load

MODEL = sys.argv[1] if len(sys.argv) > 1 else 'manjunathshiva/opendecider-nano'
RUN_GLOB = sys.argv[2] if len(sys.argv) > 2 else 'runs/2026*-cm1-*'
REPO = str(__import__('pathlib').Path(__file__).resolve().parents[1]) + '/'
VALID = {'20261025_080701_182035_repeat-planner_ext_safe-fight1-map04-seed109',
         '20261025_094736_087794_repeat-planner_ext_safe-fight1-map03-seed4'}
MOVES = ['stationary', 'strafe_left', 'strafe_right', 'backward']
model = load(MODEL)
t0 = time.time()

def choose(state, q):
    return model.system_one(state, {'q': q})['answers']['q']['choice']

rows = [json.loads(l) for l in open(REPO + 'results/doom_heads/fight-rollouts-v1.jsonl')]
val = [r for r in rows if r['run'] in VALID]
reg = collections.defaultdict(float)
picks = collections.Counter()
for r in val:
    best = max(r['scores'].values())
    for name, (state, q) in {'raw': (r['raw_state'], r['raw_question']), 'compact': (r['state'], r['question'])}.items():
        c = choose(state, q)
        picks[name, c] += 1
        reg[name] += best - r['scores'][c]
    reg['uniform'] += best - sum(r['scores'][m] for m in MOVES) / 4
n = len(val)
print(f'MOVEMENT {MODEL} validation rows {n} ({time.time() - t0:.0f}s)')
for k in ('raw', 'compact', 'uniform'):
    print(f'  regret {k}: {reg[k] / n:.3f}')
print('  picks', dict(picks))
print('  Laya (same rows): doom-trained head 2.236, recorded in game 2.153, outcome-trained head 1.920, stationary 5.075')

# Weapon questions from recent games (current wording), with Laya's logged answer.
random.seed(0)
qs = []
for d in sorted(glob.glob(RUN_GLOB)):
    for line in open(d + '/decisions.jsonl'):
        x = json.loads(line)
        q = x['packet']['questions'].get('weapon')
        if not q or (x.get('directive') or {}).get('action') != 'attack':
            continue
        c = q['criteria']
        if not any('Can hit visible enemies from here: yes' in v for v in c.values()):
            continue
        laya = x['answers'].get('weapon', {}).get('choice')
        qs.append((x['packet']['state'], q, laya))
sample = random.sample(qs, min(300, len(qs)))
bad = collections.Counter()
for state, q, laya in sample:
    c = choose(state, q)
    bad['opendecider'] += 'Can hit visible enemies from here: no' in q['criteria'].get(c, '')
    bad['laya'] += 'Can hit visible enemies from here: no' in q['criteria'].get(laya, '')
    bad['keep_od'] += c == 'keep'
print(f'WEAPON {len(sample)} attack weapon questions (of {len(qs)}); choices that cannot hit: '
      f'Laya {bad["laya"]}, OpenDecider {bad["opendecider"]}; OpenDecider "keep" {bad["keep_od"]} ({time.time() - t0:.0f}s)')
