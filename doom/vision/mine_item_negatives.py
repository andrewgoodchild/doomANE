"""Mine training-map frames on which a pixel model, as run in play, shows a false item object (pix6, item
hard negatives). A false item object is a 4-connected group of item cells touching no item instance
(item_operating_point.py). Training maps only: validation and held-out maps are refused.

    cd ext/doomLaya-dev && .venv/bin/python ../../doom/vision/mine_item_negatives.py ../../results/vision/mobileclip_pix3 \
        --data ../vision-data/frames_d ../vision-data/scenery_d ../vision-data/onpolicy_d --out ../vision-data/itemneg_d

Writes <out>/<source>_<stem>.npz and .json in collect_frames.py's format, with only the mined frames.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import item_operating_point as I  # noqa: E402

VAL_MAPS = ('MAP01', 'MAP02', 'MAP03', 'MAP04')
SPLIT = json.loads((I.ROOT / 'doom/map_split.json').read_text())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('model', type=Path)
    p.add_argument('--data', type=Path, nargs='+', required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    pixel = I.PixelPerception(a.model)
    a.out.mkdir(parents=True, exist_ok=False)
    summary = {}
    for data in a.data:
        for f in sorted(data.glob('*.npz')):
            game_map = f.stem.split('_')[-1]
            if game_map in VAL_MAPS or game_map in SPLIT['holdout']:
                continue
            I.DATA = data.parent
            frames = I.frames(pixel, (data.name,), f.stem)
            keep = []
            for k, (item, p_item, truth) in enumerate(frames):
                groups, n = I.components(item)
                if n - len(set(np.unique(groups[truth > 0])) - {0}) > 0:
                    keep.append(k)
            with np.load(f) as z:
                d = {key: z[key] for key in z.files}
            tables = json.loads((data / f'{f.stem}.json').read_text())
            name = f'{data.name}_{f.stem}'
            np.savez_compressed(a.out / f'{name}.npz', **{key: v[keep] for key, v in d.items() if len(v) == len(tables)})
            (a.out / f'{name}.json').write_text(json.dumps([tables[k] for k in keep]))
            summary[name] = [len(keep), len(tables)]
            print(name, len(keep), '/', len(tables), flush=True)
    (a.out / 'mined.json').write_text(json.dumps(dict(model=str(a.model), frames=summary), indent=1))


if __name__ == '__main__':
    main()
