"""Judgement POC, label B: at the 209 fight states of the command rollouts, did Laya's own command lead to a death
within 15 s? Features come from the run's telemetry row at that tick (what the agent knew then), never from the
rollout. Leave-one-run-out over the 7 runs.

    ext/vision-venv/bin/python poc/judgement/label_b.py
Adds `label_B` to results/poc/judgement/result.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import HITSCAN, MELEE, COMMANDS, SIDES, FAR  # noqa: E402
from calib import leave_one_run_out, metrics, bootstrap_auroc, auroc, ece  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / 'ext/doomLaya/runs'
OUT = REPO / 'results/poc/judgement'
SPLIT = json.loads((REPO / 'doom/map_split.json').read_text())


def telemetry_features(row, command):
    f = dict(hp=float(row['hp']), armor=float(row['armor']), ammo=float(row['ammo']), kills=float(row.get('kills', 0)))
    for slot in range(2, 8):
        inv = row.get('inventory', {}).get(str(slot), {})
        f[f'owned_{slot}'] = float(inv.get('owned', 0))
        f[f'ammo_{slot}'] = float(inv.get('ammo', 0))
    f['weapon_slot'] = float(row.get('weapon_slot', 0))
    en = row.get('enemies', [])
    vis = [e for e in en if e.get('visible')]
    mem = [e for e in en if not e.get('visible')]
    f['n_vis'], f['n_mem'] = float(len(vis)), float(len(mem))
    f['nearest_vis_m'] = min([e['distance'] for e in vis], default=FAR)
    f['nearest_any_m'] = min([e['distance'] for e in en], default=FAR)
    for lim in (5, 10, 20):
        f[f'vis_within_{lim}'] = float(sum(1 for e in vis if e['distance'] <= lim))
    f['vis_hitscan'] = float(sum(1 for e in vis if e['name'] in HITSCAN))
    f['vis_melee'] = float(sum(1 for e in vis if e['name'] in MELEE))
    f['vis_projectile'] = float(sum(1 for e in vis if e['name'] not in HITSCAN and e['name'] not in MELEE))
    f['threat_sum'] = sum(1.0 / max(e['distance'], 1.0) for e in vis)
    cf = row.get('combat_facts', {}) or {}
    f['incoming'] = float(cf.get('incoming', 0) or 0)
    f['incoming_seconds'] = float(cf['seconds']) if cf.get('seconds') is not None else -1.0
    f['incoming_known'] = 0.0 if cf.get('seconds') is None else 1.0
    f['incoming_miss_yes'] = 1.0 if cf.get('miss') in (True, 'yes') else 0.0
    f['incoming_miss_no'] = 1.0 if cf.get('miss') in (False, 'no') else 0.0
    for side in SIDES:
        f[f'incoming_{side}'] = 1.0 if cf.get('side') == side else 0.0
    f['melee_nearest'] = float(cf.get('melee', FAR) or FAR)
    f['hitscan_visible'] = float(cf.get('hitscan', 0) or 0)
    walls = row.get('walls', {}) or {}
    for k in ('left', 'ahead', 'right'):
        f[f'wall_{k}'] = float(walls.get(k, FAR) or FAR)
    rd = row.get('recent_damage', {}) or {}
    f['recent_hp_loss'] = float(rd.get('hp_loss', 0) or 0)
    f['recent_damage_age'] = float(rd['latest_age_seconds']) if rd.get('latest_age_seconds') is not None else 99.0
    health = [i['distance'] for i in row.get('items', []) if i.get('category') == 'Health']
    f['nearest_health_m'] = min(health, default=FAR)
    for c in COMMANDS:
        f[f'cmd_{c}'] = 1.0 if command == c else 0.0
    return f


def main():
    rows = [json.loads(l) for l in open(REPO / 'results/doom/command-rollouts-v1.jsonl') if l.strip()]
    model_rows = [r for r in rows if r['command'] == 'model']
    assert all(r['map'] in SPLIT['development'] for r in model_rows)
    # Laya's actual command at that state is the row with the same scenario state whose command is 'model'; the
    # command it stood for is the other row with identical outcome fields is not recoverable, so read it from telemetry.
    feats, y, groups, names = [], [], [], None
    cache = {}
    for r in model_rows:
        tick = int(Path(r['state']).stem)
        if r['run'] not in cache:
            cache[r['run']] = {t['tick']: t for t in (json.loads(l) for l in open(RUNS / r['run'] / 'telemetry.jsonl') if l.strip())}
        row = cache[r['run']][tick]
        command = ((row.get('execution') or {}).get('command')) or 'attack'
        f = telemetry_features(row, command)
        names = names or list(f.keys())
        feats.append([f[n] for n in names])
        y.append(1 if r['died'] else 0)
        groups.append(r['run'])
    X, y, groups = np.array(feats, np.float32), np.array(y), np.array(groups)
    print(f'label B: {len(y)} states, {len(set(groups))} runs, {int(y.sum())} deaths within 15 s, {X.shape[1]} features')
    fi = {n: i for i, n in enumerate(names)}
    L = dict(constant={k: v for k, v in metrics(np.full(len(y), y.mean()), y).items() if k != 'reliability'},
             rule_nearest_visible=dict(auroc=round(auroc(-X[:, fi['nearest_vis_m']], y), 4)),
             rule_hp=dict(auroc=round(auroc(-X[:, fi['hp']], y), 4)),
             rule_threat_sum=dict(auroc=round(auroc(X[:, fi['threat_sum']], y), 4)))
    for name, hidden in (('logistic', 0), ('mlp64', 64)):
        p_cal, p_raw, folds = leave_one_run_out(X, y, groups, hidden=hidden)
        m = metrics(p_cal, y)
        m['auroc_ci95'] = bootstrap_auroc(p_cal, y)
        m['ece_before_temperature'] = round(ece(p_raw, y)[0], 4)
        m['folds'] = folds
        L[name] = m
        print(f'  {name}: AUROC {m["auroc"]} CI {m["auroc_ci95"]} ECE {m["ece"]} Brier {m["brier"]}')
    L['features'] = names
    L['label'] = "Laya's own command at a fight state up to 25 s before a death: died within the 15 s rollout"
    path = OUT / 'result.json'
    result = json.loads(path.read_text()) if path.exists() else {}
    result['labels'] = result.get('labels', {})
    result['labels']['B'] = L
    path.write_text(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
