"""What the stack is told about enemies in pixel mode, against the engine's monsters (backlog item 1b).

Each sequence (collect_sequences.py) is fed tic by tic through doomLaya's PixelPerception and the agent's
enemy memory (Sensors.remember_enemies), as in a game, with tracks and memory reset at every window. Each
listed enemy (seen now or remembered) is matched to the nearest monster in the object list within
MATCH_UNITS plus 20 % of its distance; unmatched entries are phantoms, and entries matched to a monster
already matched are duplicates. Items have no truth in the sequences; what is counted is how many the
executor would remember (its item memory keeps every item seen and, in pixel mode, drops only what the
sensors' negative evidence reports). The runtime confirms and forgets by tic, so the sequences must be collected
at every tic (collect_sequences.py --stride 1); other spacings are refused.

    cd ext/doomLaya-dev && .venv/bin/python ../../doom/vision/runtime_replay.py ../../results/vision/mobileclip_pix3
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'ext/doomLaya-dev'))
import agent  # noqa: E402
import doomlib.pixel_perception as pixel_perception  # noqa: E402
from doomlib.pixel_perception import PixelPerception  # noqa: E402

MATCH_UNITS = 96


def full_resolution(image, depth):
    """The saved half-resolution view as the 640x480 buffers the runtime reads."""
    screen = np.zeros((480, 640, 3), np.uint8)
    screen[:400] = image.repeat(2, 0).repeat(2, 1)
    full_depth = np.zeros((480, 640), depth.dtype)
    full_depth[:400] = depth.repeat(2, 0).repeat(2, 1)
    return screen, full_depth


def labels(listed, monsters, px, py):
    """Each listed entry as 'phantom', 'duplicate' or 'real' (the first entry matched to a monster)."""
    matched, out = set(), []
    for e in listed:
        d = math.hypot(e['x'] - px, e['y'] - py)
        best = min(monsters, key=lambda m: math.hypot(m[2] - e['x'], m[3] - e['y']), default=None)
        if best is None or math.hypot(best[2] - e['x'], best[3] - e['y']) > MATCH_UNITS + 0.2 * d:
            out.append('phantom')
        elif best[0] in matched:
            out.append('duplicate')
        else:
            matched.add(best[0])
            out.append('real')
    return out


def one_to_one(listed, monsters, px, py):
    """Phantoms (no monster in reach), extras (a monster in reach, but each taken by another entry) and real
    monsters matched, with a maximum one-to-one matching: close monsters do not make false duplicates."""
    reach = [[math.hypot(m[2] - e['x'], m[3] - e['y']) <= MATCH_UNITS + 0.2 * math.hypot(e['x'] - px, e['y'] - py)
              for m in monsters] for e in listed]
    owner = {}

    def take(a, seen):
        for b, ok in enumerate(reach[a]):
            if ok and b not in seen:
                seen.add(b)
                if b not in owner or take(owner[b], seen):
                    owner[b] = a
                    return True
        return False
    for a in range(len(listed)):
        take(a, set())
    phantoms = sum(not any(r) for r in reach)
    return phantoms, len(listed) - phantoms - len(owner), len(owner)


def score(listed, monsters, px, py):
    found = labels(listed, monsters, px, py)
    return found.count('phantom'), found.count('duplicate'), found.count('real')


def detections(pixel, d, cache):
    """PixelPerception's output for every frame of a sequence. The model's cell classes do not depend on the
    grouping or the memory, so they are computed once per model and sequence and cached; grouping and tracking
    run again each time."""
    if cache.exists():
        cells = np.load(cache)
    else:
        cells = np.stack([pixel.cells(*full_resolution(d['images'][i], d['depth'][i])) for i in range(len(d['images']))]).astype(np.uint8)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, cells)
    model_cells = pixel.cells
    out, previous = [], None
    try:
        for i in range(len(cells)):
            if d['window'][i] != previous:
                pixel.tracks = {}; previous = d['window'][i]
            px, py, angle = d['players'][i]
            pixel.cells = lambda *a, i=i: cells[i]
            out.append(pixel.read(None, full_resolution(d['images'][i], d['depth'][i])[1], px, py, angle, int(d['tick'][i])))
    finally:
        pixel.cells = model_cells
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('model', type=Path)
    p.add_argument('--data', type=Path, default=ROOT / 'ext/vision-data/seq_tic')
    p.add_argument('--out', type=Path, default=None)
    p.add_argument('--negative-evidence', action='store_true', help="pass the depth buffer to the memory (backlog 1c)")
    p.add_argument('--remember-seen', type=int, default=None, help='agent.REMEMBER_SEEN_TICKS for this replay')
    p.add_argument('--cache', type=Path, default=ROOT / 'ext/vision-data/replay_cache', help='cell classes per model and sequence')
    p.add_argument('--memory-ticks', type=int, default=70, help='the enemy memory window for this replay')
    p.add_argument('--forget-range', type=float, default=None, help='agent.NEGATIVE_EVIDENCE_M for this replay')
    p.add_argument('--merge-depth', type=float, default=None, help='pixel_perception.MERGE_DEPTH_M for this replay')
    a = p.parse_args()
    if a.merge_depth is not None:
        pixel_perception.MERGE_DEPTH_M = a.merge_depth
    if a.forget_range is not None:
        agent.NEGATIVE_EVIDENCE_M = a.forget_range
    if a.remember_seen is not None:
        agent.REMEMBER_SEEN_TICKS = a.remember_seen
    pixel = PixelPerception(a.model)
    report = {'model': str(a.model)}
    for f in sorted(a.data.glob('*.npz')):
        with np.load(f) as z:
            d = {k: z[k] for k in z.files}  # indexing the archive itself decompresses the whole array each time
        truth = json.loads((f.parent / f'{f.stem}_monsters.json').read_text())
        steps = np.diff(d['tick'])[np.diff(d['window']) == 0]
        if (steps != 1).any():
            raise SystemExit(f'{f.name}: frames are not one tic apart (collect with --stride 1)')
        seen = detections(pixel, d, a.cache / a.model.name / f'{f.stem}_cells.npy')
        sensors, previous = None, None
        rows, short = [], []  # short: seen_tics of remembered entries matched to a real monster
        for i in range(len(d['images'])):
            if d['window'][i] != previous:
                sensors = agent.Sensors(pixel=pixel, enemy_memory_ticks=a.memory_ticks); previous = d['window'][i]; known = {}
            px, py, angle = d['players'][i]
            tick = int(d['tick'][i])
            depth = full_resolution(d['images'][i], d['depth'][i])[1] if a.negative_evidence else None
            enemies, items = [dict(e) for e in seen[i][0]], [dict(it) for it in seen[i][1]]
            known.update({it['id']: it for it in items})
            if a.negative_evidence:
                for oid in sensors.forget_items(items, px, py, angle, depth):
                    known.pop(oid, None)
            listed = sensors.remember_enemies(enemies, {}, px, py, angle, tick, depth)
            visible = [e for e in listed if e.get('visible')]
            found = labels(listed, truth[i], px, py)
            ph, du, real = found.count('phantom'), found.count('duplicate'), found.count('real')
            short += [e.get('seen_tics', 0) for e, kind in zip(listed, found) if kind == 'real' and not e.get('visible')]
            vph, vdu, vreal = score(visible, truth[i], px, py)
            rows.append((len(listed), ph, du, real, len(visible), vph, vdu, len(known), vreal, *one_to_one(visible, truth[i], px, py)))
        r = np.array(rows, float)
        report[f.stem] = {
            'frames': len(r),
            'listed_per_frame': round(r[:, 0].mean(), 2),
            'phantom_entries_per_frame': round(r[:, 1].mean(), 2),
            'duplicate_entries_per_frame': round(r[:, 2].mean(), 2),
            'real_monsters_listed_per_frame': round(r[:, 3].mean(), 2),
            'real_monsters_seen_per_frame': round(r[:, 8].mean(), 2),
            'frames_listing_a_phantom': round(float((r[:, 1] > 0).mean()), 3),
            'visible_phantom_entries_per_frame': round(r[:, 5].mean(), 2),
            'visible_duplicate_entries_per_frame': round(r[:, 6].mean(), 2),
            'remembered_phantom_entries_per_frame': round((r[:, 1] - r[:, 5]).mean(), 2),
            'remembered_phantom_or_duplicate_per_frame': round((r[:, 1] + r[:, 2] - r[:, 5] - r[:, 6]).mean(), 2),
            'most_listed': int(r[:, 0].max()),
            # visible entries with a maximum one-to-one matching
            'visible_phantoms_one_to_one': round(r[:, 9].mean(), 4),
            'visible_extras_one_to_one': round(r[:, 10].mean(), 4),
            'real_seen_one_to_one': round(r[:, 11].mean(), 4),
            'items_remembered_per_frame': round(r[:, 7].mean(), 2),
            'most_items_remembered': int(r[:, 7].max()),
            # real monsters remembered out of view, and the share from tracks seen on fewer than K tics (what K drops)
            'real_remembered_per_frame': round(len(short) / len(r), 2),
            'real_remembered_share_shorter_than': {k: round(sum(s < k for s in short) / max(1, len(short)), 3) for k in (4, 8, 12, 18)},
        }
        print(f.stem, json.dumps(report[f.stem]), flush=True)
    report['negative_evidence'] = a.negative_evidence
    report['remember_seen_ticks'] = agent.REMEMBER_SEEN_TICKS
    report['merge_depth_m'] = pixel_perception.MERGE_DEPTH_M
    report['negative_evidence_m'] = agent.NEGATIVE_EVIDENCE_M
    report['enemy_memory_ticks'] = a.memory_ticks
    out = a.out or ROOT / f'results/vision/runtime_replay_{a.model.name}{"_ne" if a.negative_evidence else ""}.json'
    out.write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
