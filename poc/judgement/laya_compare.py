"""Side by side: Laya's own yes/no (noul) head against the POC model on the same held-out fight states.

Needs a doomLaya model server (doom/ane_doomlaya.py) on --endpoint. Each sampled fight state's full packet text
(`raw_state`) is sent with two noul questions; the yes-probabilities are scored against label A and the died label,
next to the POC model's out-of-fold probabilities on the same rows.

    ext/vision-venv/bin/python poc/judgement/laya_compare.py --endpoint http://127.0.0.1:8017/predict --negatives 1592
Adds `laya_comparison` to results/poc/judgement/result.json.
"""
import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import label_a, label_died  # noqa: E402
from calib import auroc, ece, bootstrap_auroc  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / 'results/poc/judgement'
QUESTIONS = {
    'danger': dict(type='noul', instructions='The player will lose a lot of health or die in the next few seconds if the fight continues as it is.'),
    'retreat': dict(type='noul', instructions='Pull back now to heal or take cover instead of continuing the fight.'),
}


def ask(endpoint, state):
    body = json.dumps({'state': state, 'questions': QUESTIONS, 'model': 'doom-adapted'}).encode()
    req = urllib.request.Request(endpoint, body, {'content-type': 'application/json'})
    with urllib.request.urlopen(req, timeout=120) as r:
        answers = json.load(r)['answers']
    return {k: float(answers[k]['noul']) for k in QUESTIONS}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--endpoint', default='http://127.0.0.1:8017/predict')
    p.add_argument('--negatives', type=int, default=1592)
    a = p.parse_args()
    recs = [json.loads(l) for l in open(REPO / 'results/doom_heads/fight-rollouts-v1.jsonl') if l.strip()]
    yA = np.array([label_a(r) for r in recs])
    yD = np.array([label_died(r) for r in recs])
    rng = np.random.default_rng(0)
    idx = np.concatenate([np.where(yA == 1)[0], rng.choice(np.where(yA == 0)[0], a.negatives, replace=False)])
    idx.sort()
    oof_A = np.load(OUT / 'oof_A_mlp64.npy')
    oof_D = np.load(OUT / 'oof_died_logistic.npy')
    t = time.time()
    laya = {k: np.zeros(len(idx)) for k in QUESTIONS}
    for n, i in enumerate(idx):
        ans = ask(a.endpoint, recs[i]['raw_state'])
        for k in QUESTIONS:
            laya[k][n] = ans[k]
        if n % 200 == 0:
            print(f'  {n}/{len(idx)} asked, {time.time() - t:.0f} s', flush=True)
    res = dict(states=int(len(idx)), positives_A=int(yA[idx].sum()), positives_died=int(yD[idx].sum()),
               seconds=round(time.time() - t, 1), questions={k: v['instructions'] for k, v in QUESTIONS.items()}, state_text='raw_state (the full packet state)')
    for lab, y in (('A', yA[idx]), ('died', yD[idx])):
        res[lab] = {}
        for k in QUESTIONS:
            res[lab][f'laya_{k}'] = dict(auroc=round(auroc(laya[k], y), 4), auroc_ci95=bootstrap_auroc(laya[k], y), ece=round(ece(laya[k], y)[0], 4),
                                        mean_yes=round(float(laya[k].mean()), 4), mean_yes_when_yes=round(float(laya[k][y == 1].mean()), 4), mean_yes_when_no=round(float(laya[k][y == 0].mean()), 4))
        poc = oof_A[idx] if lab == 'A' else oof_D[idx]
        res[lab]['poc_model_same_rows'] = dict(auroc=round(auroc(poc, y), 4), auroc_ci95=bootstrap_auroc(poc, y), ece=round(ece(poc, y)[0], 4))
        print(lab, json.dumps(res[lab]))
    np.savez_compressed(OUT / 'laya_noul_sample.npz', idx=idx, **laya)
    path = OUT / 'result.json'
    result = json.loads(path.read_text())
    result['laya_comparison'] = res
    path.write_text(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
