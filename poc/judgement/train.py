"""Judgement POC, labels A and died: can a small calibrated model, from what Laya saw, say whether the fight is about
to go badly? Leave-one-run-out over the 8 fight1 runs (development maps MAP02-04).

    ext/vision-venv/bin/python poc/judgement/train.py
Writes results/poc/judgement/result.json and reliability_*.svg.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import vectorize, label_a, label_died, features  # noqa: E402
from calib import leave_one_run_out, metrics, bootstrap_auroc, auroc, ece  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / 'results/poc/judgement'
OUT.mkdir(parents=True, exist_ok=True)
SPLIT = json.loads((REPO / 'doom/map_split.json').read_text())
CRITERION = dict(label='A', auroc_min=0.70, ece_max=0.05)


def load():
    recs = [json.loads(l) for l in open(REPO / 'results/doom_heads/fight-rollouts-v1.jsonl') if l.strip()]
    bad = [r['map'] for r in recs if r['map'] not in SPLIT['development']]
    assert not bad, f'non-development maps in the labels: {set(bad)}'
    return recs


def reliability_svg(table, title, path):
    w, h, m = 320, 320, 40
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" font-family="sans-serif" font-size="11">',
             f'<text x="{w/2}" y="16" text-anchor="middle">{title}</text>',
             f'<line x1="{m}" y1="{h-m}" x2="{w-m}" y2="{m}" stroke="#999" stroke-dasharray="4"/>',
             f'<line x1="{m}" y1="{h-m}" x2="{w-m}" y2="{h-m}" stroke="#000"/><line x1="{m}" y1="{m}" x2="{m}" y2="{h-m}" stroke="#000"/>',
             f'<text x="{w/2}" y="{h-8}" text-anchor="middle">predicted probability</text>',
             f'<text x="12" y="{h/2}" text-anchor="middle" transform="rotate(-90 12 {h/2})">fraction positive</text>']
    s = (w - 2 * m)
    for row in table:
        lo = float(row['bin'].split('-')[0])
        x = m + lo * s
        y = h - m - row['frac_pos'] * s
        parts.append(f'<rect x="{x+1}" y="{y}" width="{s/10-2}" height="{h-m-y}" fill="#4a7bd0" opacity="0.8"/>')
        parts.append(f'<text x="{x+s/20}" y="{h-m+12}" text-anchor="middle" font-size="8">{row["n"]}</text>')
    parts.append('</svg>')
    path.write_text('\n'.join(parts))


def evaluate(name, X, y, groups, hidden, log):
    t = time.time()
    log(f'{name}: {X.shape[1]} features, {len(y)} rows, {int(y.sum())} positives, hidden={hidden}')
    p_cal, p_raw, folds = leave_one_run_out(X, y, groups, hidden=hidden, log=log)
    res = metrics(p_cal, y)
    res['auroc_ci95'] = bootstrap_auroc(p_cal, y)
    res['ece_before_temperature'] = round(ece(p_raw, y)[0], 4)
    res['folds'] = folds
    res['per_map'] = {}
    for mp in sorted(set(g.split('-map')[1].split('-')[0].upper() for g in groups)):
        m = np.array([g.split('-map')[1].split('-')[0].upper() == mp for g in groups])
        res['per_map'][f'MAP{mp}'] = dict(n=int(m.sum()), positives=int(y[m].sum()), auroc=round(auroc(p_cal[m], y[m]), 4), ece=round(ece(p_cal[m], y[m])[0], 4))
    res['seconds'] = round(time.time() - t, 1)
    log(f'  -> AUROC {res["auroc"]} CI {res["auroc_ci95"]}  ECE {res["ece"]} (raw {res["ece_before_temperature"]})  Brier {res["brier"]}  {res["seconds"]} s')
    return res, p_cal


def main():
    log = print
    recs = load()
    groups = np.array([r['run'] for r in recs])
    X, names = vectorize(recs, include_chosen=True)
    X_nc, names_nc = vectorize(recs, include_chosen=False)
    assert not np.isnan(X).any()
    yA = np.array([label_a(r) for r in recs])
    yD = np.array([label_died(r) for r in recs])
    log(f'records {len(recs)}, runs {len(set(groups))}, label A positives {yA.sum()} ({yA.mean():.3%}), died {yD.sum()} ({yD.mean():.3%})')
    result = dict(criterion=CRITERION, data=dict(file='results/doom_heads/fight-rollouts-v1.jsonl', records=len(recs), runs=sorted(set(groups)),
                  maps=sorted(set(r['map'] for r in recs)), label_A='chosen movement: died or lost >= 20 health in the rollout',
                  label_died='chosen movement: died in the rollout', split='leave-one-run-out; inner leave-one-run-out picks epochs and temperature'),
                  features=names, labels={})
    fi = {n: i for i, n in enumerate(names)}
    for lname, y in (('A', yA), ('died', yD)):
        L = {}
        prev = y.mean()
        L['constant'] = metrics(np.full(len(y), prev), y)
        for k in ('reliability',):
            L['constant'].pop(k)
        L['rule_nearest_threat'] = dict(auroc=round(auroc(-X[:, fi['nearest_threat_m']], y), 4))
        L['rule_nearest_visible'] = dict(auroc=round(auroc(-X[:, fi['nearest_vis_m']], y), 4))
        L['rule_hp'] = dict(auroc=round(auroc(-X[:, fi['hp']], y), 4))
        L['rule_threat_sum'] = dict(auroc=round(auroc(X[:, fi['threat_sum']], y), 4))
        L['logistic'], p_lr = evaluate(f'label {lname} logistic', X, y, groups, 0, log)
        L['mlp64'], p_mlp = evaluate(f'label {lname} mlp64', X, y, groups, 64, log)
        L['mlp64_without_chosen'], _ = evaluate(f'label {lname} mlp64 no chosen', X_nc, y, groups, 64, log)
        best = max(('logistic', 'mlp64'), key=lambda k: L[k]['auroc'])
        L['best_model'] = best
        reliability_svg(L[best]['reliability'], f'label {lname}, {best}: reliability (n per bin below)', OUT / f'reliability_{lname}.svg')
        np.save(OUT / f'oof_{lname}_{best}.npy', (p_lr if best == 'logistic' else p_mlp).astype(np.float32))
        result['labels'][lname] = L
    A = result['labels']['A'][result['labels']['A']['best_model']]
    result['verdict'] = dict(passed=bool(A['auroc'] >= CRITERION['auroc_min'] and A['ece'] <= CRITERION['ece_max']),
                             label_A_best_model=result['labels']['A']['best_model'], auroc=A['auroc'], auroc_ci95=A['auroc_ci95'], ece=A['ece'])
    (OUT / 'result.json').write_text(json.dumps(result, indent=1))
    log('VERDICT', json.dumps(result['verdict']))


if __name__ == '__main__':
    main()
