"""Record action-labelled Freedoom frames for the world-model POC (development maps only, doom/map_split.json).

Each episode: warp the player (god mode, skill 3, so monsters move) to a random standable point, then walk with
a random policy with momentum (an action pattern held for 3-20 tics, biased to moving forward and turning) for
--episode-tics tics, recording every tic: the 256x192 view with no HUD, downscaled 2x to 128x96 RGB uint8, the
7 buttons (MOVE_FORWARD, MOVE_BACKWARD, TURN_LEFT, TURN_RIGHT, MOVE_LEFT, MOVE_RIGHT, ATTACK), x, y, angle and
health, and the episode id (training never crosses an episode boundary). Frames go to a .npy memmap per map.

    cd ext/doomLaya-dev && .venv/bin/python ../../poc/dream/record.py --maps MAP01 MAP02 MAP03 --frames 60000 40000 40000
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import vizdoom as vzd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'doom/vision'))
from collect_frames import inside, standable, random_point, turn_to  # noqa: E402  (reused unchanged)

SPLIT = json.loads((ROOT / 'doom/map_split.json').read_text())
H, W = 96, 128
BUTTONS = [vzd.Button.MOVE_FORWARD, vzd.Button.MOVE_BACKWARD, vzd.Button.TURN_LEFT, vzd.Button.TURN_RIGHT,
           vzd.Button.MOVE_LEFT, vzd.Button.MOVE_RIGHT, vzd.Button.ATTACK]
# action patterns over the 7 buttons and their sampling weights: forward and turning dominate
PATTERNS = [([1, 0, 0, 0, 0, 0, 0], 22), ([1, 0, 1, 0, 0, 0, 0], 10), ([1, 0, 0, 1, 0, 0, 0], 10),
            ([0, 0, 1, 0, 0, 0, 0], 10), ([0, 0, 0, 1, 0, 0, 0], 10), ([0, 0, 0, 0, 1, 0, 0], 5), ([0, 0, 0, 0, 0, 1, 0], 5),
            ([0, 1, 0, 0, 0, 0, 0], 5), ([1, 0, 0, 0, 0, 0, 1], 6), ([0, 0, 0, 0, 0, 0, 1], 6), ([0, 0, 1, 0, 0, 0, 1], 3),
            ([0, 0, 0, 1, 0, 0, 1], 3), ([0, 0, 0, 0, 0, 0, 0], 5)]


def make_game(doom_map, seed):
    game = vzd.DoomGame()
    game.set_doom_game_path(str(Path(vzd.__file__).parent / 'freedoom2.wad'))
    game.set_doom_map(doom_map)
    game.set_doom_skill(3)
    game.set_seed(seed)
    game.set_screen_resolution(vzd.ScreenResolution.RES_256X192)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_sectors_info_enabled(True)
    game.set_render_hud(False)
    game.set_render_crosshair(False)
    game.set_render_messages(False)
    game.set_window_visible(False)
    game.set_sound_enabled(False)
    game.set_available_buttons(BUTTONS)
    game.set_available_game_variables([vzd.GameVariable.POSITION_X, vzd.GameVariable.POSITION_Y, vzd.GameVariable.ANGLE,
                                       vzd.GameVariable.HEALTH])
    game.set_episode_timeout(0)
    game.add_game_args('+freelook 0 +sv_cheats 1 +neverswitchonpickup 1')
    game.init()
    game.make_action([0] * 7, 12)
    game.send_game_command('god')
    game.send_game_command('give weapons')
    game.send_game_command('give ammo')
    game.make_action([0] * 7, 2)
    return game


def downscale(frame):  # 192x256x3 -> 96x128x3, 2x2 mean
    f = frame.astype(np.uint16)
    return ((f[0::2, 0::2] + f[1::2, 0::2] + f[0::2, 1::2] + f[1::2, 1::2]) // 4).astype(np.uint8)


def turn_to7(game, facing):
    """collect_frames.turn_to uses 4-button actions; the same with our 7 buttons."""
    for _ in range(150):
        error = ((facing - game.get_game_variable(vzd.GameVariable.ANGLE) + 180) % 360) - 180
        if abs(error) < 2:
            return True
        game.make_action([0, 0, 1, 0, 0, 0, 0] if error > 0 else [0, 0, 0, 1, 0, 0, 0], 1)
    return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--maps', nargs='+', default=['MAP01'])
    p.add_argument('--frames', nargs='+', type=int, default=[60000])
    p.add_argument('--episode-tics', type=int, default=700)
    p.add_argument('--out', type=Path, default=ROOT / 'ext/poc-data/dream')
    p.add_argument('--seed', type=int, default=7)
    a = p.parse_args()
    for m in a.maps:
        assert m in SPLIT['development'], f'{m} is not a development map'
    a.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    weights = [w for _, w in PATTERNS]
    summary = {}
    for doom_map, n_frames in zip(a.maps, a.frames):
        t0 = time.time()
        frames = np.lib.format.open_memmap(a.out / f'frames_{doom_map}.npy', mode='w+', dtype=np.uint8, shape=(n_frames, H, W, 3))
        actions = np.zeros((n_frames, 7), np.uint8)
        variables = np.zeros((n_frames, 4), np.float32)
        episode = np.zeros(n_frames, np.int32)
        game = make_game(doom_map, a.seed)
        i, ep = 0, 0
        while i < n_frames:
            state = game.get_state()
            if state is None or game.is_episode_finished():
                game.close(); game = make_game(doom_map, a.seed + ep + 1); continue
            point = random_point(state.sectors, rng)
            if point is None:
                continue
            game.send_game_command(f'warp {point[0]:.0f} {point[1]:.0f}')
            if rng.random() < 0.4:
                game.send_game_command(f'slot {rng.randint(1, 7)}')
            game.make_action([0] * 7, 2)
            turn_to7(game, rng.uniform(0, 360))
            game.make_action([0] * 7, 3)
            tics = 0
            while tics < a.episode_tics and i < n_frames:
                pattern = rng.choices(PATTERNS, weights)[0][0]
                hold = rng.randint(3, 20)
                for _ in range(hold):
                    if tics >= a.episode_tics or i >= n_frames:
                        break
                    state = game.get_state()
                    if state is None or game.is_episode_finished():
                        break
                    frames[i] = downscale(state.screen_buffer)
                    actions[i] = pattern
                    variables[i] = state.game_variables
                    episode[i] = ep
                    game.make_action(pattern, 1)  # the action taken after this frame, leading to frame i+1
                    i += 1; tics += 1
                if game.is_episode_finished():
                    break
            ep += 1
            if ep % 20 == 0:
                print(f'{doom_map}: {i}/{n_frames} frames, {ep} episodes, {time.time() - t0:.0f} s', flush=True)
        game.close()
        frames.flush()
        np.savez(a.out / f'meta_{doom_map}.npz', actions=actions, variables=variables, episode=episode)
        summary[doom_map] = {'frames': int(n_frames), 'episodes': int(ep), 'episode_tics': a.episode_tics,
                             'seconds': round(time.time() - t0, 1), 'fps': round(n_frames / (time.time() - t0), 1)}
        print(doom_map, summary[doom_map], flush=True)
    (a.out / 'record_summary.json').write_text(json.dumps({'maps': summary, 'shape': [H, W, 3], 'buttons': [b.name for b in BUTTONS],
                                                           'variables': ['x', 'y', 'angle', 'health'], 'seed': a.seed}, indent=1))


if __name__ == '__main__':
    main()
