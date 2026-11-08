"""What does the pixel model call items, and what is really there? (offline, validation maps MAP01-04)

Every 4x4 cell of each saved frame (collect_frames.py: frames_d, scenery_d) is classified by the model as
PixelPerception runs it in play (Core ML, the monster operating point). Its truth is the engine instance covering
most of the cell in the instance map (the labels buffer), else "none" (wall, floor, sky). For each predicted
item class: the share of its cells over true classes; for each true item class: the share of its cells over
predicted classes. Cells, not objects: an object takes the class most of its cells have.

    cd ext/doomLaya-dev && .venv/bin/python ../../doom/vision/item_confusion.py ../../results/vision/mobileclip_pix3 \
        --out ../../results/vision/item_confusion_pix3.json
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'ext/doomLaya-dev'))
from doomlib.pixel_perception import CELLS_H, CELLS_W, PixelPerception  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from runtime_replay import full_resolution  # noqa: E402

ITEMS = ('Health', 'Armor', 'Ammo', 'Weapon', 'Key', 'Powerup')


def truth_cells(instances, table):
    """Name of the instance covering most of each cell (half-resolution 200x320 -> 48x80), else 'none'."""
    h, w = instances.shape
    out = np.empty((CELLS_H, CELLS_W), object)
    for r in range(CELLS_H):
        for c in range(CELLS_W):
            block = instances[int(r * h / CELLS_H):int((r + 1) * h / CELLS_H), int(c * w / CELLS_W):int((c + 1) * w / CELLS_W)]
            k = np.bincount(block.ravel()).argmax()
            out[r, c] = 'none' if k == 0 else table[k - 1]['name']
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('model', type=Path)
    p.add_argument('--data', type=Path, nargs='+', default=[ROOT / 'ext/vision-data/frames_d', ROOT / 'ext/vision-data/scenery_d'])
    p.add_argument('--maps', nargs='+', default=['MAP01', 'MAP02', 'MAP03', 'MAP04'])
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    pixel = PixelPerception(a.model)
    category = pixel.category
    predicted_as = defaultdict(Counter)  # predicted item class -> true names
    true_as = defaultdict(Counter)       # true item name -> predicted classes
    frames = Counter()
    for data in a.data:
        for m in a.maps:
            f = data / f'{m}.npz'
            if not f.exists():
                continue
            with np.load(f) as z:
                d = {k: z[k] for k in z.files}
            tables = json.loads((data / f'{m}.json').read_text())
            for i in range(len(d['images'])):
                labels = pixel.cells(*full_resolution(d['images'][i], d['depth'][i]))
                truth = truth_cells(d['instances'][i], tables[i])
                for r, c in zip(*np.nonzero(labels)):
                    name = pixel.classes[labels[r, c]]
                    if category[name] in ITEMS:
                        predicted_as[name][truth[r, c]] += 1
                for r, c in zip(*np.nonzero(truth != 'none')):
                    t = truth[r, c]
                    if any(row['name'] == t and row['category'] in ITEMS for row in tables[i]):
                        true_as[t][pixel.classes[labels[r, c]]] += 1
                frames[m] += 1
            print(data.name, m, frames[m], flush=True)

    def shares(counter, top=6):
        n = sum(counter.values())
        return dict(cells=n, **{k: round(v / n, 3) for k, v in counter.most_common(top)})
    report = dict(model=str(a.model), frames=dict(frames),
                  predicted_as={k: shares(v) for k, v in sorted(predicted_as.items(), key=lambda kv: -sum(kv[1].values()))},
                  true_as={k: shares(v) for k, v in sorted(true_as.items(), key=lambda kv: -sum(kv[1].values()))})
    print(json.dumps(report, indent=1))
    if a.out:
        a.out.write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
