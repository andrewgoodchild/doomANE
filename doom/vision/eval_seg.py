"""Object recall by distance and size on the validation maps (MAP01-04), from the engine's instances.

An engine instance (a monster or item in the labels buffer) is found when any 4x4 cell it covers is predicted
with its category. False objects: predicted groups of cells touching no instance of their category.

    ext/vision-venv/bin/python doom/vision/eval_seg.py results/vision/mobileclip results/vision/tiny

pix2: --data ext/vision-data/walk scores the views along the paths of the engine games on MAP01-03
(collect_frames.py --from-runs). There, frames_with_false_monster is the share of views showing at least
one false monster object: in play one is enough to hold the agent in a fight with a wall.
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

BINS = [(0, 5), (5, 10), (10, 20), (20, 1e9)]  # metres (map units / 32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('runs', nargs='+', type=Path)
    p.add_argument('--data', type=Path, default=Path('ext/vision-data/frames'))
    p.add_argument('--out', type=Path, default=Path('results/vision/recall_by_distance.json'))
    a = p.parse_args()
    data = a.data
    report = {}
    for run in a.runs:
        ckpt = torch.load(run / 'model.pt', map_location='cpu', weights_only=False)
        model = T.Seg(ckpt['backbone'], len(ckpt['classes']), ckpt.get('depth', False))
        model.load_state_dict(ckpt['state']); model.eval().to('mps')
        cat_of = np.array([T.category_number(ckpt['category'], c) for c in ckpt['classes']])
        found, total, false_obj, frames, false_monster_frames = {}, {}, {}, 0, 0
        for m in [m for m in ckpt['val_maps'] if (data / f'{m}.npz').exists()]:
            d = np.load(data / f'{m}.npz'); tables = json.loads((data / f'{m}.json').read_text())
            x = torch.from_numpy(np.concatenate([d['images'], d['depth'][..., None]], -1) if ckpt.get('depth') else d['images']).permute(0, 3, 1, 2).float()
            x = (F.interpolate(x, size=(T.H, T.W), mode='bilinear', align_corners=False).round().clamp(0, 255) / 255).contiguous()
            with torch.no_grad():
                pred = torch.cat([model(x[i:i + 32].to('mps')).argmax(1).cpu() for i in range(0, len(x), 32)]).numpy()
            pc = cat_of[pred]
            for i, table in enumerate(tables):
                frames += 1
                inst = torch.from_numpy(d['instances'][i]).float()[None, None]
                inst = F.interpolate(inst, size=(T.H, T.W), mode='nearest')[0, 0].long().numpy()
                covered = {}  # category -> cells of truth objects of that category
                for k, r in enumerate(table, 1):
                    if T.target_name(r) is None or r['pixels'] < T.MIN_PIXELS:
                        continue
                    cat = T.TARGET_CATEGORIES.index(r['category']) + 1
                    cells = np.zeros((T.H // T.CELL, T.W // T.CELL), bool)
                    rr, cc = np.nonzero(inst == k)
                    cells[rr // T.CELL, cc // T.CELL] = True
                    covered[cat] = covered.get(cat, np.zeros_like(cells)) | cells
                    hit = bool((pc[i][cells] == cat).any())
                    b = next(f'{lo}-{hi:g}m' if hi < 1e9 else f'>{lo}m' for lo, hi in BINS if lo <= r['distance'] < hi)
                    key = (r['category'], b)
                    total[key] = total.get(key, 0) + 1
                    found[key] = found.get(key, 0) + hit
                false_monster = False
                for cat in range(1, len(T.TARGET_CATEGORIES) + 1):
                    lab, n = T.blobs((pc[i] == cat).astype(np.int32))
                    for j in range(1, n + 1):
                        if not (covered.get(cat, np.zeros_like(lab, bool)) & (lab == j)).any():
                            name = T.TARGET_CATEGORIES[cat - 1]
                            false_obj[name] = false_obj.get(name, 0) + 1
                            false_monster |= name == 'Monster'
                false_monster_frames += false_monster
        rows = {}
        for cat in T.TARGET_CATEGORIES:
            r = {b: f'{found[(cat, b)]}/{total[(cat, b)]} = {found[(cat, b)] / total[(cat, b)]:.2f}'
                 for b in [f'{lo}-{hi:g}m' if hi < 1e9 else f'>{lo}m' for lo, hi in BINS] if total.get((cat, b))}
            if r:
                r['false_objects_per_frame'] = round(false_obj.get(cat, 0) / frames, 3)
                rows[cat] = r
        rows['frames'] = frames
        rows['frames_with_false_monster'] = round(false_monster_frames / frames, 3)
        report[str(run)] = rows
        print(run, json.dumps(rows, indent=1), flush=True)
    a.out.write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
