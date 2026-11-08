"""Collect contiguous stretches of views along game paths, for the persistent-phantom test (backlog item 1).

For each game run, windows of --window seconds every --every seconds of its telemetry are replayed: the
player is warped to each telemetry position and turned to its facing, every --stride tics, in one game
(god on, as in collect_frames.py), and the view, depth buffer and labels-buffer instances are saved as in
collect_frames.py. Frames carry their window number, so persistence can be measured across consecutive
frames. Development maps only; held-out maps are refused. For evaluation, never for training.

    cd ext/doomLaya-dev && .venv/bin/python ../../doom/vision/collect_sequences.py --map MAP03 \
        --out ../vision-data/seq --name engine --runs runs/2026*pix1_engine-map03-seed*

Writes <out>/<name>_<MAP>.npz (images, depth, instances, players, window, tick) and <out>/<name>_<MAP>.json
(instance tables), plus <out>/<name>_<MAP>_monsters.json: every monster in the object list per frame
(id, name, x, y), the truth for the runtime test (runtime_replay.py).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import vizdoom as vzd

import collect_frames as C


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--map', required=True)
    p.add_argument('--runs', nargs='+', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--window', type=float, default=6.0, help='seconds per window')
    p.add_argument('--every', type=float, default=180.0, help='seconds between window starts')
    p.add_argument('--stride', type=int, default=2, help='tics between frames')
    a = p.parse_args()
    if a.map in C.SPLIT['holdout']:
        raise SystemExit('held-out maps are never used')
    a.out.mkdir(parents=True, exist_ok=True)
    windows = []
    for run in a.runs:
        rows = [json.loads(line) for line in (run / 'telemetry.jsonl').open()]
        rows = [r for r in rows if r.get('map') == a.map]
        start = 0.0
        while rows and start < rows[-1]['seconds']:
            w = [r for r in rows if start <= r['seconds'] < start + a.window][::a.stride]
            if w:
                windows.append(w)
            start += a.every
    game = C.make_game(a.map, 1)
    images, depths, maps, players, tables, window_ids, ticks, monsters = [], [], [], [], [], [], [], []
    for k, w in enumerate(windows):
        for r in w:
            game.send_game_command(f"warp {r['x']:.0f} {r['y']:.0f}")
            game.make_action([0, 0, 0, 0], 1)
            C.turn_to(game, r['angle'] % 360)
            state = game.get_state()
            if state is None:
                continue
            px = game.get_game_variable(vzd.GameVariable.POSITION_X)
            py = game.get_game_variable(vzd.GameVariable.POSITION_Y)
            image, depth, instances, table = C.capture(state, px, py)
            images.append(image); depths.append(depth); maps.append(instances); tables.append(table)
            players.append((px, py, game.get_game_variable(vzd.GameVariable.ANGLE))); window_ids.append(k); ticks.append(r['tick'])
            monsters.append([(o.id, o.name, round(o.position_x, 1), round(o.position_y, 1)) for o in state.objects if o.name in C.MONSTER_NAMES])
    game.close()
    np.savez_compressed(a.out / f'{a.name}_{a.map}.npz', images=np.stack(images), depth=np.stack(depths),
                        instances=np.stack(maps), players=np.array(players), window=np.array(window_ids), tick=np.array(ticks))
    (a.out / f'{a.name}_{a.map}.json').write_text(json.dumps(tables))
    (a.out / f'{a.name}_{a.map}_monsters.json').write_text(json.dumps(monsters))
    print(a.name, a.map, len(windows), 'windows', len(images), 'frames', flush=True)


if __name__ == '__main__':
    main()
