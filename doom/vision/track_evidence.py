"""Which evidence tells a real remembered pixel enemy from a phantom? (offline check, no games)

Pixel games recorded with agent.py --shadow-engine-packet carry in telemetry, on every tic, the pixel enemy list
(tracker ids, positions, visible or remembered) and the monsters the labels buffer shows (`engine_enemies`).
A sighting of a pixel track is real when an engine monster is within MATCH_UNITS plus 20 % of its distance on
that tic; a track is real when most of its sightings so far are. Each remembered entry (listed, not visible) at
a decision tic is one example, described by features computed only from what was known by then:

  time and counts (what the memory uses now): tics since the last sighting, tics seen, entries listed,
    distance;
  geometry from the WAD (lines, sides and sectors; never its things):
    wall_distance    from the track's mean position to the nearest one-sided line;
    void             the mean position lies in no sector, or in one whose floor meets its ceiling;
    behind_wall      share of sightings whose position lies beyond a one-sided line seen from the player;
    scatter          spread of the sighting positions over the mean distance at which they were seen;
    clear_misses     tics since the last sighting on which the remembered spot was in the view cone,
                     within MISS_RANGE and with no one-sided line between the player and it.

Each feature is scored by AUC against real/phantom on all examples, and a small MLP on all of them is trained
on the first seed and tested on the second. Run from ext/doomLaya-dev:

  .venv/bin/python ../../doom/vision/track_evidence.py runs/*-pix3_truth-map0*-seed2? --out ../../results/vision/track_evidence.json
"""
import argparse
import json
import math
import struct
from collections import defaultdict
from pathlib import Path

import numpy as np

MATCH_UNITS = 96
VIEW_BEARING = 40
MISS_RANGE = 30 * 32
FEATURES = ('age', 'seen_tics', 'listed', 'distance', 'wall_distance', 'void', 'behind_wall', 'scatter', 'clear_misses')


class Geometry:
    def __init__(self, wad, name):
        data = Path(wad).read_bytes()
        count, offset = struct.unpack_from('<ii', data, 4)
        directory = [struct.unpack_from('<ii8s', data, offset + i * 16) for i in range(count)]
        index = next(i for i, (_, _, n) in enumerate(directory) if n.rstrip(b'\0').decode() == name.upper())
        lumps = {n.rstrip(b'\0').decode(): data[p:p + size] for p, size, n in directory[index + 1:index + 11]}
        vertices = np.array(list(struct.iter_unpack('<hh', lumps['VERTEXES'])), float)
        lines = list(struct.iter_unpack('<hhhhhhh', lumps['LINEDEFS']))
        sides = [s[5] for s in struct.iter_unpack('<hh8s8s8sh', lumps['SIDEDEFS'])]
        sectors = [(s[0], s[1]) for s in struct.iter_unpack('<hh8s8shhh', lumps['SECTORS'])]
        self.a = vertices[[l[0] for l in lines]]
        self.b = vertices[[l[1] for l in lines]]
        self.front = np.array([sides[l[5]] for l in lines])
        self.back = np.array([sides[l[6]] if l[6] != -1 else -1 for l in lines])
        self.solid = self.back == -1
        self.sectors = sectors

    def crosses(self, p, q, mask=None):
        """Whether segment p-q crosses any (masked) line."""
        a, b = (self.a, self.b) if mask is None else (self.a[mask], self.b[mask])
        def orient(u, v, w):
            return (v[..., 0] - u[..., 0]) * (w[..., 1] - u[..., 1]) - (v[..., 1] - u[..., 1]) * (w[..., 0] - u[..., 0])
        p, q = np.asarray(p, float), np.asarray(q, float)
        d1, d2 = orient(a, b, p), orient(a, b, q)
        d3, d4 = orient(p, q, a), orient(p, q, b)
        return bool(np.any((d1 * d2 < 0) & (d3 * d4 < 0)))

    def wall_distance(self, p):
        a, b = self.a[self.solid], self.b[self.solid]
        ab = b - a
        t = np.clip(((p - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-9), 0, 1)
        return float(np.min(np.hypot(*(a + t[:, None] * ab - p).T)))

    def sector(self, p):
        """The sector holding p: the first line hit by a ray towards +x, and the side of it p is on."""
        x, y = p
        a, b = self.a, self.b
        dy = b[:, 1] - a[:, 1]
        with np.errstate(divide='ignore', invalid='ignore'):
            t = (y - a[:, 1]) / dy
            hx = a[:, 0] + t * (b[:, 0] - a[:, 0])
        ok = (dy != 0) & (t >= 0) & (t < 1) & (hx > x)
        if not ok.any():
            return None
        i = int(np.argmin(np.where(ok, hx, np.inf)))
        right = (b[i, 0] - a[i, 0]) * (y - a[i, 1]) - (b[i, 1] - a[i, 1]) * (x - a[i, 0]) < 0  # front side is on the right
        s = self.front[i] if right else self.back[i]
        return None if s == -1 else self.sectors[s]


def bearing(x, y, px, py, angle):
    return (angle - math.degrees(math.atan2(y - py, x - px)) + 180) % 360 - 180


def examples(run, geometry_for):
    config = json.loads((run / 'config.json').read_text())
    game_map = config['args']['map'].upper()
    geo = geometry_for(game_map)
    decisions = {json.loads(l)['tick'] for l in (run / 'decisions.jsonl').open()}
    history = defaultdict(list)  # id -> [(x, y, distance units, real)]
    misses = defaultdict(int)
    out = []
    for line in (run / 'telemetry.jsonl').open():
        r = json.loads(line)
        if 'engine_enemies' not in r:
            raise SystemExit(f'{run}: telemetry has no engine_enemies (record with --shadow-engine-packet)')
        px, py, angle, tick = r['x'], r['y'], r['angle'], r['tick']
        truth = r['engine_enemies']
        for e in r['enemies']:
            if e.get('visible'):
                d = math.hypot(e['x'] - px, e['y'] - py)
                real = any(math.hypot(m['x'] - e['x'], m['y'] - e['y']) <= MATCH_UNITS + 0.2 * d for m in truth)
                behind = geo.crosses((px, py), (e['x'], e['y']), geo.solid)
                history[e['id']].append((e['x'], e['y'], d, real, behind, tick))
                misses[e['id']] = 0
            else:
                d = math.hypot(e['x'] - px, e['y'] - py)
                if abs(bearing(e['x'], e['y'], px, py, angle)) < VIEW_BEARING and d < MISS_RANGE \
                        and not geo.crosses((px, py), (e['x'], e['y']), geo.solid):
                    misses[e['id']] += 1
        if tick not in decisions:
            continue
        listed = len(r['enemies'])
        for e in r['enemies']:
            h = history.get(e['id'])
            if e.get('visible') or not h:
                continue
            xy = np.array([(x, y) for x, y, *_ in h])
            mean = xy.mean(0)
            sector = geo.sector(mean)
            out.append(dict(run=run.name, map=game_map, tick=tick, id=e['id'],
                            real=sum(s[3] for s in h) * 2 > len(h),
                            age=tick - h[-1][5], seen_tics=len(h), listed=listed, distance=e['distance'],
                            wall_distance=geo.wall_distance(mean),
                            void=float(sector is None or sector[1] - sector[0] <= 0),
                            behind_wall=sum(s[4] for s in h) / len(h),
                            scatter=float(np.hypot(*(xy - mean).T).mean() / max(1.0, np.mean([s[2] for s in h]))),
                            clear_misses=misses[e['id']]))
    return out


def auc(score, label):
    order = np.argsort(score, kind='mergesort')
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    for v in np.unique(score):  # ties share their mean rank
        tie = score == v
        ranks[tie] = ranks[tie].mean()
    pos = label.sum()
    return float((ranks[label].sum() - pos * (pos + 1) / 2) / (pos * (len(label) - pos)))


def main():
    import vizdoom
    p = argparse.ArgumentParser()
    p.add_argument('runs', type=Path, nargs='+')
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    wad = Path(vizdoom.__file__).parent / 'freedoom2.wad'
    cache = {}
    rows = []
    for run in a.runs:
        rows += examples(run, lambda m: cache.setdefault(m, Geometry(wad, m)))
    y = np.array([r['real'] for r in rows])
    X = np.array([[r[f] for f in FEATURES] for r in rows], float)
    report = dict(examples=len(rows), real_share=round(float(y.mean()), 3), runs=[r.name for r in a.runs], auc={})
    for i, f in enumerate(FEATURES):
        s = auc(X[:, i], y)
        report['auc'][f] = dict(auc=round(max(s, 1 - s), 3), higher_means=('real' if s >= .5 else 'phantom'),
                                real_mean=round(float(X[y, i].mean()), 3), phantom_mean=round(float(X[~y, i].mean()), 3))
    import torch
    seeds = sorted({r['run'].rsplit('seed', 1)[1] for r in rows})
    test = np.array([r['run'].endswith('seed' + seeds[-1]) for r in rows])

    def mlp(cols):
        Z = X[:, cols]
        mu, sd = Z[~test].mean(0), Z[~test].std(0) + 1e-6
        z = torch.tensor((Z - mu) / sd, dtype=torch.float32)
        t = torch.tensor(y, dtype=torch.float32)
        torch.manual_seed(0)
        net = torch.nn.Sequential(torch.nn.Linear(len(cols), 32), torch.nn.ReLU(), torch.nn.Linear(32, 1))
        opt = torch.optim.Adam(net.parameters(), 1e-2, weight_decay=1e-3)
        tr = torch.tensor(~test)
        for _ in range(500):
            loss = torch.nn.functional.binary_cross_entropy_with_logits(net(z[tr]).squeeze(1), t[tr])
            opt.zero_grad(); loss.backward(); opt.step()
        return round(auc(net(z[~tr]).squeeze(1).detach().numpy(), y[test]), 3)
    report['mlp_test_seed'] = seeds[-1]
    report['mlp_test_examples'] = int(test.sum())
    report['mlp_auc_time_and_counts'] = mlp([0, 1, 2, 3])
    report['mlp_auc_geometry'] = mlp([4, 5, 6, 7, 8])
    report['mlp_auc_all'] = mlp(list(range(len(FEATURES))))
    by_map = defaultdict(list)
    for r in rows:
        by_map[r['map']].append(r['real'])
    report['real_share_by_map'] = {m: round(float(np.mean(v)), 3) for m, v in sorted(by_map.items())}
    print(json.dumps(report, indent=1))
    if a.out:
        a.out.write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
