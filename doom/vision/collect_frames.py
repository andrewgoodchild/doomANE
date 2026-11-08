"""Collect screen frames with per-pixel ground truth for training a pixel perception model.

Development maps only (doom/map_split.json). The player is warped to random standable points, often
near a monster or an item and facing roughly towards it, with `god` on so the monsters wake, attack and
fire projectiles as they do in play. A few random tics of turning, walking and firing come before each
capture, so weapon animations and muzzle flashes appear, and the weapon is switched at random.

Each frame keeps the view above the status bar (the top 400 of 480 rows) at half resolution (200x320),
with an instance map of the same size taken from the labels buffer (0 = nothing; k = row k-1 of that
frame's instance table: name, category, distance). Held-out maps are refused.

    cd ext/doomLaya-dev && .venv/bin/python ../../doom/vision/collect_frames.py --out ../vision-data/frames --per-map 200

Writes <out>/<MAP>.npz (images, depth, instances, players) and <out>/<MAP>.json (instance tables).

--scenery: random points and facings only, every frame kept whether or not it shows anything (pix2: the
pix1 data was three quarters object views, and wall screens were read as monsters in play).
Each frame also keeps the depth buffer at the same half resolution (pix3 gives it to the model as a fourth
channel: a sprite stands out from what is behind it, a texture lies on its wall).
--from-runs RUN...: the positions and facings of game runs' telemetry instead of random points, for an
evaluation set along the paths the agent actually walks (never for training: those maps are the games').
"""
import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import vizdoom as vzd

ROOT = Path(__file__).resolve().parents[2]
SPLIT = json.loads((ROOT / 'doom/map_split.json').read_text())
VIEW_ROWS = 400  # the status bar is the bottom 80 rows of 480
TARGET_CATEGORIES = ('Monster', 'Health', 'Armor', 'Ammo', 'Weapon', 'Key', 'Powerup')
NEAR_DEPTH = 12  # depth buffer units of about 0.25 m


def make_game(doom_map, seed):
    game = vzd.DoomGame()
    game.set_doom_game_path(str(Path(vzd.__file__).parent / 'freedoom2.wad'))
    game.set_doom_map(doom_map)
    game.set_doom_skill(3)
    game.set_seed(seed)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_labels_buffer_enabled(True)
    game.set_depth_buffer_enabled(True)
    game.set_sectors_info_enabled(True)
    game.set_objects_info_enabled(True)
    game.set_render_hud(True)
    game.set_window_visible(False)
    game.set_sound_enabled(False)
    game.set_available_buttons([vzd.Button.TURN_LEFT, vzd.Button.TURN_RIGHT, vzd.Button.MOVE_FORWARD, vzd.Button.ATTACK])
    game.set_available_game_variables([vzd.GameVariable.ANGLE, vzd.GameVariable.POSITION_X, vzd.GameVariable.POSITION_Y])
    game.set_episode_timeout(0)
    game.add_game_args('+freelook 0 +sv_cheats 1 +neverswitchonpickup 1')
    game.init()
    game.make_action([0, 0, 0, 0], 12)  # the first tics ignore input
    game.send_game_command('god')
    game.send_game_command('give weapons')
    game.send_game_command('give ammo')
    game.make_action([0, 0, 0, 0], 2)
    return game


def turn_to(game, facing):
    """Turn with the discrete buttons until within 2 degrees of facing."""
    for _ in range(150):
        error = ((facing - game.get_game_variable(vzd.GameVariable.ANGLE) + 180) % 360) - 180
        if abs(error) < 2:
            return True
        game.make_action([1, 0, 0, 0] if error > 0 else [0, 1, 0, 0], 1)
    return False


def inside(x, y, lines):
    crossings = 0
    for ln in lines:
        (x1, y1), (x2, y2) = (ln.x1, ln.y1), (ln.x2, ln.y2)
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            crossings += 1
    return crossings % 2 == 1


def standable(sectors, x, y):
    for sector in sectors:
        if sector.lines and inside(x, y, sector.lines):
            return sector.ceiling_height - sector.floor_height >= 56
    return False


def random_point(sectors, rng):
    for _ in range(200):
        sector = rng.choice(sectors)
        if not sector.lines:
            continue
        xs = [v for ln in sector.lines for v in (ln.x1, ln.x2)]
        ys = [v for ln in sector.lines for v in (ln.y1, ln.y2)]
        x, y = rng.uniform(min(xs), max(xs)), rng.uniform(min(ys), max(ys))
        if standable(sectors, x, y):
            return x, y
    return None


def capture(state, px, py):
    """Half-resolution view and instance map, and the instance table."""
    labels = state.labels_buffer[:VIEW_ROWS:2, ::2]
    instances = np.zeros(labels.shape, np.uint8)
    table = []
    for label in state.labels:
        if str(label.object_category) == 'Self':
            continue
        mask = labels == label.value
        if not mask.any():
            continue
        table.append({'name': label.object_name, 'category': str(label.object_category), 'id': int(label.object_id),
                      'distance': round(math.hypot(label.object_position_x - px, label.object_position_y - py) / 32, 2),
                      'pixels': int(mask.sum())})
        instances[mask] = len(table)
        if len(table) == 255:
            break
    return state.screen_buffer[:VIEW_ROWS:2, ::2].copy(), state.depth_buffer[:VIEW_ROWS:2, ::2].copy(), instances, table


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--maps', nargs='*', default=SPLIT['development'])
    p.add_argument('--per-map', type=int, default=200)
    p.add_argument('--restart-every', type=int, default=40, help='frames between map restarts (items picked up, monsters gathered)')
    p.add_argument('--seed', type=int, default=11)
    p.add_argument('--scenery', action='store_true', help='random points and facings, every frame kept')
    p.add_argument('--from-runs', nargs='*', type=Path, default=[], help='game run directories to take positions from (one map)')
    a = p.parse_args()
    walk = []
    if a.from_runs:
        for run in a.from_runs:
            config = json.loads((run / 'config.json').read_text())
            rows = [json.loads(line) for line in (run / 'telemetry.jsonl').open()]
            walk += [(r['x'], r['y'], r['angle']) for r in rows if r.get('map', config.get('map')) in a.maps]
        if len(a.maps) != 1 or not walk:
            raise SystemExit('--from-runs takes one map and runs that played it')
        walk = random.Random(a.seed).sample(walk, min(a.per_map, len(walk)))
    held_out = set(SPLIT['holdout'])
    if any(m in held_out for m in a.maps):
        raise SystemExit('held-out maps are never used for training data')
    a.out.mkdir(parents=True, exist_ok=True)
    for doom_map in a.maps:
        if (a.out / f'{doom_map}.npz').exists():
            print(doom_map, 'exists'); continue
        rng = random.Random(f'{a.seed}-{doom_map}')
        images, depths, maps, players, tables = [], [], [], [], []
        game, tries = None, 0
        while len(images) < a.per_map and tries < 40 * a.per_map:
            if game is None or (len(images) and len(images) % a.restart_every == 0 and tries_since_restart > 0):
                if game: game.close()
                game = make_game(doom_map, rng.randrange(1 << 30))
                tries_since_restart = 0
            tries += 1
            tries_since_restart += 1
            state = game.get_state()
            if state is None or game.is_episode_finished():
                game.close(); game = None; continue
            sectors = state.sectors
            targets = [o for o in state.objects if o.name != 'DoomPlayer']
            mode = 1.0 if a.scenery else rng.random()
            if walk:
                if tries > len(walk):
                    break
                x, y, facing = walk[tries - 1]  # each walked point once
            elif mode < 0.75 and targets:
                pool = [o for o in targets if o.name in MONSTER_NAMES] if mode < 0.45 else targets
                pool = pool or targets
                t = rng.choice(pool)
                d, b = rng.uniform(80, 700), rng.uniform(0, 2 * math.pi)
                x, y = t.position_x + d * math.cos(b), t.position_y + d * math.sin(b)
                if not standable(sectors, x, y):
                    continue
                facing = math.degrees(math.atan2(t.position_y - y, t.position_x - x)) + rng.uniform(-50, 50)
            else:
                point = random_point(sectors, rng)
                if point is None:
                    continue
                x, y = point
                facing = rng.uniform(0, 360)
            game.send_game_command(f'warp {x:.0f} {y:.0f}')
            if rng.random() < 0.3:
                game.send_game_command(f'slot {rng.randint(1, 7)}')
            game.make_action([0, 0, 0, 0], 2)
            if not turn_to(game, facing % 360):
                continue
            for _ in range(0 if walk else rng.randint(0, 3)):  # a little motion and firing before the capture
                game.make_action([int(rng.random() < .2), int(rng.random() < .2), int(rng.random() < .4), int(rng.random() < .35)], rng.randint(1, 6))
            state = game.get_state()
            if state is None:
                continue
            if not walk and (state.depth_buffer[:VIEW_ROWS] > NEAR_DEPTH).mean() < 0.4:
                continue  # facing a wall close up
            px = game.get_game_variable(vzd.GameVariable.POSITION_X)
            py = game.get_game_variable(vzd.GameVariable.POSITION_Y)
            image, depth, instances, table = capture(state, px, py)
            has_target = any(r['category'] in TARGET_CATEGORIES for r in table)
            if not has_target and not (a.scenery or walk) and rng.random() < 0.6:
                continue  # keep some empty frames, but mostly frames with something to find
            images.append(image); depths.append(depth); maps.append(instances); tables.append(table)
            players.append((px, py, game.get_game_variable(vzd.GameVariable.ANGLE)))
        if game: game.close()
        np.savez_compressed(a.out / f'{doom_map}.npz', images=np.stack(images), depth=np.stack(depths), instances=np.stack(maps), players=np.array(players))
        (a.out / f'{doom_map}.json').write_text(json.dumps(tables))
        counts = {}
        for t in tables:
            for r in t:
                if r['category'] in TARGET_CATEGORIES:
                    counts[r['category']] = counts.get(r['category'], 0) + 1
        print(doom_map, len(images), 'frames', tries, 'tries', counts, flush=True)


MONSTER_NAMES = {'Zombieman', 'ShotgunGuy', 'ChaingunGuy', 'WolfensteinSS', 'DoomImp', 'Demon', 'Spectre', 'LostSoul',
                 'Cacodemon', 'HellKnight', 'BaronOfHell', 'Arachnotron', 'PainElemental', 'Revenant', 'Fatso',
                 'Archvile', 'Cyberdemon', 'SpiderMastermind', 'CommanderKeen'}

if __name__ == '__main__':
    main()
