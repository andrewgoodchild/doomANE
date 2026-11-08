"""Train and test the track-belief scorer for remembered pixel enemies (doomLaya doomlib/track_belief.py).

Pixel games recorded with agent.py --shadow-engine-packet are replayed tic by tic through TrackBelief, exactly as
in play, and every remembered entry at a decision tic becomes an example. Its label: most of the track's
sightings so far had a monster from the labels buffer within MATCH_UNITS plus 20 % of their distance. The
labels are used only here, never in play.

  train: an MLP (FEATURES -> 32 -> 1) on the training runs; the threshold keeps KEEP_REAL of their real entries.
  test:  share of real entries kept and of phantom entries removed on other runs, against the registered rule.

Run from ext/doomLaya-dev:
  .venv/bin/python ../../doom/vision/train_track_belief.py train runs/<run>... --out ../../results/vision/track_belief_v1.json
  .venv/bin/python ../../doom/vision/train_track_belief.py test runs/<run>... --scorer ../../results/vision/track_belief_v1.json
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.cwd()))
from doomlib.track_belief import FEATURES, Geometry, Scorer, TrackBelief  # noqa: E402

MATCH_UNITS = 96
KEEP_REAL = 0.92
RULE_KEEP, RULE_REMOVE = 0.90, 0.50


def examples(run, geometry_for):
    config = json.loads((run / 'config.json').read_text())
    game_map = config['args']['map'].upper()
    belief = TrackBelief(geometry_for(game_map))
    decisions = {json.loads(l)['tick'] for l in (run / 'decisions.jsonl').open()}
    real = {}  # id -> [real sightings, sightings]
    out = []
    for line in (run / 'telemetry.jsonl').open():
        r = json.loads(line)
        if 'engine_enemies' not in r:
            raise SystemExit(f'{run}: telemetry has no engine_enemies (record with --shadow-engine-packet)')
        px, py, tick = r['x'], r['y'], r['tick']
        for e in r['enemies']:
            if e.get('visible'):
                d = math.hypot(e['x'] - px, e['y'] - py)
                hit = any(math.hypot(m['x'] - e['x'], m['y'] - e['y']) <= MATCH_UNITS + 0.2 * d for m in r['engine_enemies'])
                count = real.setdefault(e['id'], [0, 0])
                count[0] += hit
                count[1] += 1
        belief.observe(r['enemies'], px, py, r['angle'], tick)
        if tick not in decisions:
            continue
        for e in r['enemies']:
            if e.get('visible'):
                continue
            f = belief.features(e, tick)
            if f is None:
                continue
            out.append(dict(f, run=run.name, map=game_map, tick=tick, id=e['id'], real=real[e['id']][0] * 2 > real[e['id']][1]))
    return out


def load(runs):
    import vizdoom
    wad = Path(vizdoom.__file__).parent / 'freedoom2.wad'
    cache = {}
    rows = [r for run in runs for r in examples(run, lambda m: cache.setdefault(m, Geometry(wad, m)))]
    return rows, np.array([[r[f] for f in FEATURES] for r in rows], float), np.array([r['real'] for r in rows])


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def train(a):
    import torch
    rows, X, y = load(a.runs)
    mean, std = X.mean(0), X.std(0) + 1e-6
    z = torch.tensor((X - mean) / std, dtype=torch.float32)
    t = torch.tensor(y, dtype=torch.float32)
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(len(FEATURES), 32), torch.nn.ReLU(), torch.nn.Linear(32, 1))
    opt = torch.optim.Adam(net.parameters(), 1e-2, weight_decay=1e-3)
    for _ in range(500):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(net(z).squeeze(1), t)
        opt.zero_grad()
        loss.backward()
        opt.step()
    p = torch.sigmoid(net(z).squeeze(1)).detach().numpy()
    threshold = float(np.quantile(p[y], 1 - KEEP_REAL))
    spec = dict(features=list(FEATURES), mean=mean.tolist(), std=std.tolist(),
                w1=net[0].weight.detach().tolist(), b1=net[0].bias.detach().tolist(),
                w2=net[2].weight.detach().tolist(), b2=net[2].bias.detach().tolist(),
                threshold=threshold, keep_real=KEEP_REAL, examples=len(rows), real_share=round(float(y.mean()), 3),
                train_runs={r.name: digest(r / 'telemetry.jsonl') for r in a.runs}, trainer_sha256=digest(__file__))
    a.out.write_text(json.dumps(spec))
    # The numpy scorer used in play must reproduce the trained network.
    scorer = Scorer(a.out)
    numpy_p = np.array([scorer.probability(dict(zip(FEATURES, x))) for x in X])
    if np.abs(numpy_p - p).max() > 1e-4:
        raise SystemExit('numpy scorer differs from the trained network')
    print(json.dumps(dict(examples=len(rows), real_share=spec['real_share'], threshold=round(threshold, 4),
                          train_phantoms_removed=round(float((p[~y] < threshold).mean()), 3))))


def test(a):
    rows, X, y = load(a.runs)
    scorer = Scorer(a.scorer)
    p = np.array([scorer.probability(dict(zip(FEATURES, x))) for x in X])
    kept = p >= scorer.threshold
    keep, remove = float(kept[y].mean()), float((~kept[~y]).mean())
    by_map = {}
    for m in sorted({r['map'] for r in rows}):
        s = np.array([r['map'] == m for r in rows])
        by_map[m] = dict(real=int((s & y).sum()), phantom=int((s & ~y).sum()),
                         real_kept=round(float(kept[s & y].mean()), 3), phantoms_removed=round(float((~kept[s & ~y]).mean()), 3))
    report = dict(scorer=str(a.scorer), runs=[r.name for r in a.runs], examples=len(rows), real=int(y.sum()), phantom=int((~y).sum()),
                  real_kept=round(keep, 3), phantoms_removed=round(remove, 3),
                  real_share_before=round(float(y.mean()), 3), real_share_after=round(float(y[kept].mean()), 3),
                  rule=dict(keep=RULE_KEEP, remove=RULE_REMOVE, passed=keep >= RULE_KEEP and remove >= RULE_REMOVE), by_map=by_map)
    print(json.dumps(report, indent=1))
    if a.out:
        a.out.write_text(json.dumps(report, indent=1))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='step', required=True)
    t = sub.add_parser('train')
    t.add_argument('runs', type=Path, nargs='+')
    t.add_argument('--out', type=Path, required=True)
    s = sub.add_parser('test')
    s.add_argument('runs', type=Path, nargs='+')
    s.add_argument('--scorer', type=Path, required=True)
    s.add_argument('--out', type=Path)
    a = p.parse_args()
    train(a) if a.step == 'train' else test(a)


if __name__ == '__main__':
    main()
