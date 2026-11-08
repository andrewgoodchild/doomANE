"""Item operating point for a pixel model: a cell keeps an item class only when its summed item-class
probability reaches a threshold (PixelPerception item_threshold). Cells are classified as in play (Core ML,
the model's monster operating point), then each threshold on a grid is scored by:
  item recall: item instances (any cell where the instance covers most of the cell) with at least one such
               cell predicted as an item;
  false item objects: 4-connected groups of item cells touching no item instance cell.
  set:   the highest threshold whose MAP04 recall (frames_d) is at least RECALL_SHARE of the argmax's.
  check: on walk_d MAP02 and MAP03, false item objects per frame at most FALSE_SHARE of the argmax's and recall
         at least KEEP_SHARE of it, on both maps (pre-registered 2026-11-07).
    cd ext/doomLaya-dev && .venv/bin/python ../../doom/vision/item_operating_point.py ../../results/vision/mobileclip_pix3 \
        --out ../../results/vision/item_operating_point_pix3.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'ext/doomLaya-dev'))
from doomlib.pixel_perception import CELLS_H, CELLS_W, VIEW_ROWS, PixelPerception  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from runtime_replay import full_resolution  # noqa: E402

ITEMS = ('Health', 'Armor', 'Ammo', 'Weapon', 'Key', 'Powerup')
GRID = np.round(np.arange(0.50, 0.995, 0.01), 2)
RECALL_SHARE, FALSE_SHARE, KEEP_SHARE = 0.90, 0.50, 0.85
DATA = ROOT / 'ext/vision-data'


def frames(pixel, sets, game_map):
    """Per frame: item mask at the argmax (after the monster step), item probability, item instance id per cell."""
    pixel.item_threshold = None
    out = []
    for name in sets:
        f = DATA / name / f'{game_map}.npz'
        with np.load(f) as z:
            d = {k: z[k] for k in z.files}
        tables = json.loads((f.parent / f'{game_map}.json').read_text())
        for i in range(len(d['images'])):
            screen, depth = full_resolution(d['images'][i], d['depth'][i])
            view = np.concatenate([screen[:VIEW_ROWS:2, ::2], depth[:VIEW_ROWS:2, ::2, None]], -1) if pixel.depth else screen[:VIEW_ROWS:2, ::2]
            logits = pixel.model.predict({'view': view[None].astype(np.float16)})['logits'][0].astype(np.float32)
            p = np.exp(logits - logits.max(0))
            labels = pixel.cells(screen, depth)
            inst, h, w = d['instances'][i], *d['instances'][i].shape
            truth = np.zeros((CELLS_H, CELLS_W), int)
            for r in range(CELLS_H):
                for c in range(CELLS_W):
                    k = np.bincount(inst[int(r * h / CELLS_H):int((r + 1) * h / CELLS_H), int(c * w / CELLS_W):int((c + 1) * w / CELLS_W)].ravel()).argmax()
                    truth[r, c] = k if k and tables[i][k - 1]['category'] in ITEMS else 0
            out.append((pixel.item[labels], (p[pixel.item].sum(0) / p.sum(0)).astype(np.float16), truth))
    return out


def components(mask):
    """4-connected groups of a boolean cell mask: a label per cell (0 outside) and the count."""
    groups, n = np.zeros(mask.shape, int), 0
    for r, c in zip(*np.nonzero(mask)):
        if groups[r, c]:
            continue
        n += 1
        groups[r, c] = n
        stack = [(r, c)]
        while stack:
            i, j = stack.pop()
            for u, v in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                if 0 <= u < mask.shape[0] and 0 <= v < mask.shape[1] and mask[u, v] and not groups[u, v]:
                    groups[u, v] = n
                    stack.append((u, v))
    return groups, n


def score(data, t):
    found = total = false = 0
    for item, p_item, truth in data:
        cells = item & (p_item >= t) if t is not None else item
        ids = set(np.unique(truth)) - {0}
        total += len(ids)
        found += len(ids & set(np.unique(truth[cells])))
        groups, n = components(cells)
        false += n - len(set(np.unique(groups[truth > 0])) - {0})
    return dict(recall=round(found / max(1, total), 4), false_per_frame=round(false / len(data), 4), instances=total, frames=len(data))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('model', type=Path)
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    pixel = PixelPerception(a.model)
    tune = frames(pixel, ('frames_d',), 'MAP04')
    base = score(tune, None)
    curve = {float(t): score(tune, t) for t in GRID}
    chosen = max(t for t, s in curve.items() if s['recall'] >= RECALL_SHARE * base['recall'])
    report = dict(model=str(a.model), map04_argmax=base, map04_curve=curve, item_threshold=chosen, check={})
    passed = True
    for m in ('MAP02', 'MAP03'):
        data = frames(pixel, ('walk_d',), m)
        b, s = score(data, None), score(data, chosen)
        ok = s['false_per_frame'] <= FALSE_SHARE * b['false_per_frame'] and s['recall'] >= KEEP_SHARE * b['recall']
        passed &= ok
        report['check'][m] = dict(argmax=b, threshold=s, false_share=round(s['false_per_frame'] / max(1e-9, b['false_per_frame']), 3),
                                  recall_share=round(s['recall'] / max(1e-9, b['recall']), 3), passed=ok)
    report['rule'] = dict(recall_share=RECALL_SHARE, false_share=FALSE_SHARE, keep_share=KEEP_SHARE, passed=passed)
    print(json.dumps({k: v for k, v in report.items() if k != 'map04_curve'}, indent=1))
    if a.out:
        a.out.write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
