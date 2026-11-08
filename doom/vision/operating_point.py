"""Compare pixel perception models at a matched operating point, on single views and on view sequences
(backlog items 1 and 8, docs/PIXEL_BACKLOG.md).

A cell is a monster when its summed monster-class probability is at least a threshold. Each model's
threshold is set so that its monster object recall on MAP04 (a validation map that is not a game map)
equals --recall (default: the first model's recall at the plain argmax). Then, at that threshold:
- walked views (collect_frames.py --from-runs): share of views with a false monster;
- sequences (collect_sequences.py): phantom tracks, false monster groups linked across consecutive frames
  when they overlap (within 2 cells), and how long they last.
A false monster is a group of monster cells touching no monster of any size in the labels buffer.

    ext/vision-venv/bin/python doom/vision/operating_point.py results/vision/mobileclip results/vision/mobileclip_pix3 ...
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import train_seg as T  # noqa: E402

DATA = Path('ext/vision-data')
STRIDE = 2  # tics between sequence frames
LENGTHS = (2, 9, 18)  # frames: the runtime's confirmation, about half a second, about a second


def load_model(run):
    ckpt = torch.load(run / 'model.pt', map_location='cpu', weights_only=False)
    model = T.Seg(ckpt['backbone'], len(ckpt['classes']), ckpt.get('depth', False))
    model.load_state_dict(ckpt['state']); model.eval().to('mps')
    monster = torch.tensor([c != 'none' and ckpt['category'][c] == 'Monster' for c in ckpt['classes']])
    return model, monster, ckpt.get('depth', False)


def monster_probability(model, monster, depth, d):
    images, depths = d['images'], d['depth']
    out = []
    with torch.no_grad():
        for i in range(0, len(images), 32):
            x = np.concatenate([images[i:i + 32], depths[i:i + 32, ..., None]], -1) if depth else images[i:i + 32]
            x = torch.from_numpy(x).permute(0, 3, 1, 2).float()
            x = (F.interpolate(x, size=(T.H, T.W), mode='bilinear', align_corners=False).round().clamp(0, 255) / 255).contiguous()
            p = model(x.to('mps')).softmax(1).cpu()
            out.append(p[:, monster].sum(1).half())
    return torch.cat(out).numpy()


def truth(d, tables):
    """Per frame: cells of each monster instance of at least MIN_PIXELS, and cells of all monsters."""
    out = []
    for i, table in enumerate(tables):
        inst = F.interpolate(torch.from_numpy(d['instances'][i]).float()[None, None], size=(T.H, T.W), mode='nearest')[0, 0].long().numpy()
        cell = inst.reshape(T.H // T.CELL, T.CELL, T.W // T.CELL, T.CELL)
        objects, anything = [], np.zeros((T.H // T.CELL, T.W // T.CELL), bool)
        for k, r in enumerate(table, 1):
            if r['category'] != 'Monster' or r['name'].startswith('Dead'):
                continue
            cells = (cell == k).any(axis=(1, 3))
            anything |= cells
            if r['pixels'] >= T.MIN_PIXELS:
                objects.append(cells)
        out.append((objects, anything))
    return out


def recall(prob, gt, threshold):
    found = total = 0
    for p, (objects, _) in zip(prob, gt):
        m = p >= threshold
        total += len(objects)
        found += sum(bool((m & o).any()) for o in objects)
    return found / total


def false_groups(p, anything, threshold):
    lab, n = T.blobs((p >= threshold).astype(np.int32))
    return [lab == j for j in range(1, n + 1) if not (anything & (lab == j)).any()]


def dilate(m, r=2):
    t = torch.from_numpy(m.astype(np.float32))[None, None]
    return F.max_pool2d(t, 2 * r + 1, 1, r)[0, 0].numpy() > 0


def tracks(prob, gt, window, threshold):
    """Lengths (in frames) of phantom tracks within each window."""
    lengths, live = [], []  # live: (mask, length)
    previous = None
    for p, (_, anything), w in zip(prob, gt, window):
        if w != previous:
            lengths += [n for _, n in live]; live = []; previous = w
        groups = false_groups(p, anything, threshold)
        nxt, used = [], set()
        for g in groups:
            match = next((i for i, (m, _) in enumerate(live) if i not in used and (dilate(m) & g).any()), None)
            if match is None:
                nxt.append((g, 1))
            else:
                used.add(match); nxt.append((g, live[match][1] + 1))
        lengths += [n for i, (_, n) in enumerate(live) if i not in used]
        live = nxt
    return lengths + [n for _, n in live]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('runs', nargs='+', type=Path)
    p.add_argument('--recall', type=float, default=None)
    p.add_argument('--out', type=Path, default=Path('results/vision/operating_point.json'))
    a = p.parse_args()
    cal = np.load(DATA / 'frames_d/MAP04.npz'); cal_gt = truth(cal, json.loads((DATA / 'frames_d/MAP04.json').read_text()))
    walk = {m: np.load(DATA / f'walk_d/{m}.npz') for m in ('MAP01', 'MAP02', 'MAP03')}
    walk_gt = {m: truth(walk[m], json.loads((DATA / f'walk_d/{m}.json').read_text())) for m in walk}
    seqs = {f.stem: np.load(f) for f in sorted((DATA / 'seq').glob('*.npz'))}
    seq_gt = {k: truth(seqs[k], json.loads((DATA / f'seq/{k}.json').read_text())) for k in seqs}
    report, target = {'recall_target': a.recall}, a.recall
    for run in a.runs:
        model, monster, depth = load_model(run)
        cal_p = monster_probability(model, monster, depth, cal)
        if target is None:  # the first model at its argmax: approximated by p >= 0.5
            target = report['recall_target'] = recall(cal_p, cal_gt, 0.5)
        lo, hi = 0.0, 1.0
        for _ in range(14):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if recall(cal_p, cal_gt, mid) >= target else (lo, mid)
        th = lo
        row = {'threshold': round(th, 4), 'map04_recall': round(recall(cal_p, cal_gt, th), 3)}
        hits = total = 0
        for m in walk:
            wp = monster_probability(model, monster, depth, walk[m])
            hits += sum(bool(false_groups(q, g[1], th)) for q, g in zip(wp, walk_gt[m])); total += len(wp)
            row.setdefault('walk_recall', {})[m] = round(recall(wp, walk_gt[m], th), 3)
        row['walked_views_with_false_monster'] = round(hits / total, 3)
        for k in seqs:
            sp = monster_probability(model, monster, depth, seqs[k])
            n = tracks(sp, seq_gt[k], seqs[k]['window'], th)
            frames = len(sp)
            row[k] = {'frames': frames, **{f'phantom_tracks_{L}+_frames': sum(x >= L for x in n) for L in LENGTHS},
                      'longest_frames': max(n, default=0),
                      'per_1000_frames_18+': round(1000 * sum(x >= 18 for x in n) / frames, 2)}
        report[str(run)] = row
        print(run, json.dumps(row), flush=True)
    a.out.write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
