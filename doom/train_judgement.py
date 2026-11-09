"""Train the calibrated judgement head judgement-v1 (phase 3, docs/LAB_NOTEBOOK.md 2026-11-08).

Question answered in play: "pull back now?", at every attack decision. Label: the chosen movement's rollout in
results/doom_heads/fight-rollouts-v1.jsonl lost at least 20 health or died (doom/fight_rollouts.py, 1.2 s,
deterministic from the saved engine state). Features come from doomLaya's doomlib/judgement.py, the same code
the agent and the authority audit run, so training and play cannot drift apart.

Model: MLP 2x64, ReLU, dropout 0.1, Adam 3e-3, weight decay 1e-3, full batch, standardised inputs, seed 0.
Leave-one-run-out over the 8 runs gives out-of-fold logits at every epoch; the epoch with the lowest mean
out-of-fold BCE is kept and the temperature is fitted on those logits; the final head is trained on all runs
for that many epochs. Weights go to --out/weights.json (numpy-readable JSON), the report to --out/report.json.

Run from ext/doomLaya-dev (its venv has torch; doomlib is imported from the working directory):
  .venv/bin/python ../../doom/train_judgement.py --out ../../results/doom_heads/judgement-v1
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path.cwd()))
from doomlib.judgement import features, QUESTION, FALL_BACK_MAX_TICKS  # noqa: E402
from doomlib.combat_facts import parse as parse_facts  # noqa: E402

SPLIT = json.loads((REPO / 'doom/map_split.json').read_text())
torch.set_num_threads(4)


def auroc(scores, y):
    scores, y = np.asarray(scores, float), np.asarray(y, int)
    order = np.argsort(scores, kind='mergesort')
    ranks = np.empty(len(scores), float)
    s = scores[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    pos, neg = y == 1, y == 0
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))


def ece(p, y, bins=10):
    p, y = np.asarray(p, float), np.asarray(y, float)
    edges = np.linspace(0, 1, bins + 1)
    total, table = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if m.sum():
            total += m.sum() / len(p) * abs(y[m].mean() - p[m].mean())
            table.append(dict(bin=f'{lo:.1f}-{hi:.1f}', n=int(m.sum()), mean_p=round(float(p[m].mean()), 4), frac_pos=round(float(y[m].mean()), 4)))
    return float(total), table


class Net(torch.nn.Module):
    def __init__(self, d, hidden=64):
        super().__init__()
        self.f = torch.nn.Sequential(torch.nn.Linear(d, hidden), torch.nn.ReLU(), torch.nn.Dropout(0.1),
                                     torch.nn.Linear(hidden, hidden), torch.nn.ReLU(), torch.nn.Dropout(0.1),
                                     torch.nn.Linear(hidden, 1))

    def forward(self, x):
        return self.f(x).squeeze(-1)


def fit(X, y, epochs, seed=0, lr=3e-3, wd=1e-3, X_val=None):
    torch.manual_seed(seed)
    net = Net(X.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    Xt, yt = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    Xv = torch.tensor(X_val, dtype=torch.float32) if X_val is not None else None
    val_logits = []
    for _ in range(epochs):
        net.train()
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(net(Xt), yt)
        loss.backward()
        opt.step()
        if Xv is not None:
            net.eval()
            with torch.no_grad():
                val_logits.append(net(Xv).numpy().copy())
    net.eval()
    return net, (np.stack(val_logits) if Xv is not None else None)


def fit_temperature(logits, y):
    log_t = torch.zeros(1, requires_grad=True)
    z, yt = torch.tensor(logits, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z / torch.exp(log_t), yt)
        loss.backward()
        return loss
    opt.step(closure)
    return float(torch.exp(log_t).item())


def standardise(X):
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return mu, sd


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--labels', type=Path, default=REPO / 'results/doom_heads/fight-rollouts-v1.jsonl')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=300, help='epochs searched by the leave-one-run-out')
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    started = time.time()
    records = [json.loads(line) for line in a.labels.read_text().splitlines() if line.strip()]
    bad = {r['map'] for r in records if r['map'] not in SPLIT['development']}
    if bad:
        raise SystemExit(f'non-development maps in the labels: {sorted(bad)}')
    rows = [features(r['raw_state'], r['raw_question'], parse_facts(r['raw_question']['instructions'])) for r in records]
    names = list(rows[0])
    X = np.array([[row[n] for n in names] for row in rows], dtype=np.float64)
    y = np.array([1 if (r['outcomes'][r['chosen']]['died'] or r['outcomes'][r['chosen']]['lost'] >= 20) else 0 for r in records])
    groups = np.array([r['run'] for r in records])
    runs = sorted(set(groups))
    print(f'{len(records)} records, {len(runs)} runs, {len(names)} features, {int(y.sum())} positives ({y.mean():.3%})', flush=True)
    # leave-one-run-out: out-of-fold logits at every epoch, standardising on the training runs of each fold
    oof = np.zeros((a.epochs, len(y)))
    for run in runs:
        test = groups == run
        mu, sd = standardise(X[~test])
        _, val = fit((X[~test] - mu) / sd, y[~test], a.epochs, a.seed, X_val=(X[test] - mu) / sd)
        oof[:, test] = val
        print(f'  fold {run[-24:]}: n={int(test.sum())} positives={int(y[test].sum())}', flush=True)
    bce = [float(np.mean(np.logaddexp(0, -oof[e]) * y + np.logaddexp(0, oof[e]) * (1 - y))) for e in range(a.epochs)]
    epochs = int(np.argmin(bce)) + 1
    temperature = fit_temperature(oof[epochs - 1], y)
    p_raw = 1 / (1 + np.exp(-oof[epochs - 1]))
    p_cal = 1 / (1 + np.exp(-oof[epochs - 1] / temperature))
    e_cal, table = ece(p_cal, y)
    # the threshold: the smallest grid value at which the head says yes, out of fold, no more often than bad outcomes happen
    threshold = next(t for t in np.round(np.arange(0.01, 1.0, 0.01), 2) if (p_cal >= t).mean() <= y.mean())
    yes = p_cal >= threshold
    at_threshold = dict(threshold=float(threshold), share_yes=round(float(yes.mean()), 4), precision=round(float(y[yes].mean()), 4) if yes.any() else None,
                        recall=round(float(yes[y == 1].mean()), 4), base_rate=round(float(y.mean()), 4))
    per_fold = {run: dict(n=int((groups == run).sum()), positives=int(y[groups == run].sum()),
                          auroc=round(auroc(p_cal[groups == run], y[groups == run]), 4)) for run in runs}
    # the final head on all runs
    mu, sd = standardise(X)
    net, _ = fit((X - mu) / sd, y, epochs, a.seed)
    layers = [dict(w=m.weight.detach().numpy().astype(np.float64).tolist(), b=m.bias.detach().numpy().astype(np.float64).tolist())
              for m in net.f if isinstance(m, torch.nn.Linear)]
    weights = dict(question=QUESTION, label='chosen movement: died or lost >= 20 health in the 1.2 s rollout',
                   features=names, mean=mu.tolist(), std=sd.tolist(), layers=layers, activation='relu', temperature=temperature,
                   threshold=float(threshold), threshold_rule='smallest 0.01 step with out-of-fold yes share <= base rate', fall_back_max_ticks=FALL_BACK_MAX_TICKS,
                   training=dict(labels=str(a.labels.relative_to(REPO)), labels_sha256=hashlib.sha256(a.labels.read_bytes()).hexdigest(),
                                 records=len(records), runs=runs, epochs=epochs, seed=a.seed, lr=3e-3, weight_decay=1e-3, hidden=[64, 64], dropout=0.1))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'weights.json').write_text(json.dumps(weights))
    sha = hashlib.sha256((a.out / 'weights.json').read_bytes()).hexdigest()
    # the exported head must reproduce torch on the training rows
    from doomlib.judgement import Judgement
    head = Judgement(a.out)
    with torch.no_grad():
        torch_p = torch.sigmoid(net(torch.tensor((X - mu) / sd, dtype=torch.float32)) / temperature).numpy()
    numpy_p = np.array([head.probability(row) for row in rows])
    drift = float(np.abs(torch_p - numpy_p).max())
    report = dict(weights_sha256=sha, epochs=epochs, temperature=round(temperature, 4), features=len(names), records=len(records), positives=int(y.sum()),
                  out_of_fold=dict(auroc=round(auroc(p_cal, y), 4), ece=round(e_cal, 4), ece_before_temperature=round(ece(p_raw, y)[0], 4),
                                   brier=round(float(np.mean((p_cal - y) ** 2)), 4), share_at_least_half=round(float((p_cal >= 0.5).mean()), 4),
                                   at_threshold=at_threshold, reliability=table, per_fold=per_fold),
                  training_fit=dict(auroc=round(auroc(numpy_p, y), 4), share_yes=round(float((numpy_p >= threshold).mean()), 4)),
                  torch_numpy_max_abs_diff=drift, seconds=round(time.time() - started, 1))
    (a.out / 'report.json').write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != 'out_of_fold'}, indent=1))
    print('out-of-fold', json.dumps({k: v for k, v in report['out_of_fold'].items() if k not in ('reliability', 'per_fold')}))
    print('per fold', json.dumps(per_fold))


if __name__ == '__main__':
    main()
