"""Replay a recorded game's path in ViZDoom and save what the player saw: depth buffer, sprite mask and exact pose per frame.
Development maps only (doom/map_split.json). The level's geometry is never read here; the pose is the engine's position,
height and facing, as a game's telemetry gives it. Monsters and doors do not follow the recorded game (doors the player
walked through are closed here), which evaluate.py accounts for.
    cd ext/doomLaya-dev && .venv/bin/python ../../doom/seenmap/replay.py --run runs/<run> --map MAP01 --stride 6 \
        --out ../poc-data/seenmap/<name>.npz
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import vizdoom as vzd

ROOT = Path(__file__).resolve().parents[2]
SPLIT = json.loads((ROOT / 'doom/map_split.json').read_text())
VIEW_ROWS = 400  # the status bar is the bottom 80 rows of 480


def make_game(doom_map):
    game = vzd.DoomGame()
    game.set_doom_game_path(str(Path(vzd.__file__).parent / 'freedoom2.wad'))
    game.set_doom_map(doom_map)
    game.set_doom_skill(3)
    game.set_seed(1)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_labels_buffer_enabled(True)
    game.set_depth_buffer_enabled(True)
    game.set_render_hud(True)
    game.set_window_visible(False)
    game.set_sound_enabled(False)
    game.set_available_buttons([vzd.Button.TURN_LEFT_RIGHT_DELTA, vzd.Button.TURN_LEFT, vzd.Button.TURN_RIGHT])
    game.set_available_game_variables([vzd.GameVariable.ANGLE, vzd.GameVariable.POSITION_X, vzd.GameVariable.POSITION_Y,
                                       vzd.GameVariable.POSITION_Z])
    game.set_episode_timeout(0)
    game.add_game_args('+freelook 0 +sv_cheats 1 +neverswitchonpickup 1')
    game.init()
    game.make_action([0, 0, 0], 12)  # the first tics ignore input
    game.send_game_command('god')
    game.make_action([0, 0, 0], 2)
    return game


class Turner:
    """Turn to a facing with the delta button; its sign is measured once."""

    def __init__(self, game):
        self.game = game
        before = game.get_game_variable(vzd.GameVariable.ANGLE)
        game.make_action([10.0, 0, 0], 1)
        after = game.get_game_variable(vzd.GameVariable.ANGLE)
        self.sign = 1.0 if ((after - before + 180) % 360 - 180) > 0 else -1.0

    def turn_to(self, facing, tol=1.0):
        for _ in range(4):
            error = ((facing - self.game.get_game_variable(vzd.GameVariable.ANGLE) + 180) % 360) - 180
            if abs(error) <= tol:
                return True
            self.game.make_action([self.sign * error, 0, 0], 1)
        return abs(((facing - self.game.get_game_variable(vzd.GameVariable.ANGLE) + 180) % 360) - 180) <= 3


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--map', required=True)
    p.add_argument('--stride', type=int, default=6)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--limit', type=int, default=0)
    a = p.parse_args()
    if a.map in SPLIT['holdout'] or a.map not in SPLIT['development']:
        raise SystemExit('development maps only')
    rows = [json.loads(l) for l in (a.run / 'telemetry.jsonl').open()]
    rows = [r for r in rows if r.get('map', a.map) == a.map]
    frames = rows[::a.stride]
    if a.limit:
        frames = frames[:a.limit]
    game = make_game(a.map)
    turner = Turner(game)
    depth, mask, poses, ticks, episodes = [], [], [], [], []
    t0 = time.perf_counter()
    for i, r in enumerate(frames):
        game.send_game_command(f"warp {r['x']:.0f} {r['y']:.0f}")
        game.make_action([0, 0, 0], 1)
        turner.turn_to(r['angle'] % 360)
        state = game.get_state()
        if state is None:
            continue
        gv = state.game_variables
        depth.append(state.depth_buffer[:VIEW_ROWS:2, ::2].copy())
        mask.append(np.packbits(state.labels_buffer[:VIEW_ROWS:2, ::2] != 0, axis=1))
        poses.append((float(gv[1]), float(gv[2]), float(gv[3]), float(gv[0])))
        ticks.append(int(r['tick']))
        episodes.append(int(r.get('episode', 0)))
        if (i + 1) % 500 == 0:
            el = time.perf_counter() - t0
            print(f'{i + 1}/{len(frames)} frames, {el:.0f} s, {(i + 1) / el:.1f} frames/s', flush=True)
    game.close()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, depth=np.stack(depth), mask=np.stack(mask), poses=np.array(poses, np.float32),
                        tick=np.array(ticks, np.int32), episode=np.array(episodes, np.int16),
                        stride=a.stride, run=str(a.run), map=a.map, turn_sign=turner.sign)
    print(a.out, len(depth), 'frames of', len(rows), 'tics;', f'{time.perf_counter() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    main()
