"""Label fight decisions by outcome: replay each combat movement from the saved engine state.

A collection run (agent.py --fight-states) saves the engine state at every movement question. Here each
state is loaded once per movement (stationary, strafe left, strafe right, backward), and the attack is
executed as the executor would execute it for the model's chosen enemy and weapon (turn at most 9 degrees
a tic, fire when the aim is on the body and the weapon reaches), holding that movement for --tics. The
outcome of each movement is the health and armor lost, the damage dealt and whether the player died;
the label is the best score. Rollouts from a saved state are deterministic (ZDoom saves the RNG).
With --run-combat-movement (or --run yes) the movements are run with SPEED, about 1.7-2x the distance.

Labels come only from the game itself, never from a hosted model. Held-out maps are refused.

Run from ext/doomLaya-dev:
  .venv/bin/python ../../doom/fight_rollouts.py runs/<run>... --out ../../results/doom_heads/fight-rollouts-v1.jsonl
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HELD_OUT = set(json.loads((REPO / 'doom/map_split.json').read_text())['holdout'])
MOVEMENTS = ('stationary', 'strafe_left', 'strafe_right', 'backward')
MOVE_BUTTON = {'strafe_left': 2, 'strafe_right': 3, 'backward': 1}
DEATH_PENALTY = 100
DEALT_WEIGHT = 0.3


def score(outcome):
    return -outcome['lost'] + DEALT_WEIGHT * outcome['dealt'] - DEATH_PENALTY * outcome['died']


class Roller:
    def __init__(self, cfg, run=False):
        import vizdoom as vzd
        from agent import BUTTONS
        self.vzd = vzd
        args = cfg['args']
        game = vzd.DoomGame()
        game.set_doom_game_path(str(Path(vzd.__file__).parent / 'freedoom2.wad'))
        if args.get('weapon_sensor'):
            game.set_doom_scenario_path(str(Path.cwd() / 'assets/weapon_sensor.pk3'))
        if args.get('wad'):
            game.add_game_args('-file ' + str(Path(args['wad']).resolve()))
        game.set_doom_map(args['map'])
        game.set_doom_skill(args['skill'])
        game.set_mode(vzd.Mode.PLAYER)
        game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
        game.set_labels_buffer_enabled(True)
        game.set_objects_info_enabled(True)
        game.set_window_visible(False)
        game.set_sound_enabled(False)
        # SPEED last, as agent.py --run-combat-movement adds it; the 14 executor buttons keep their indices.
        self.run = run
        game.set_available_buttons(BUTTONS + ([vzd.Button.SPEED] if run else []))
        game.add_game_args('+freelook 0 +sv_cheats 1 +neverswitchonpickup 1')
        game.init()
        game.new_episode()
        self.game = game

    def var(self, name):
        return float(self.game.get_game_variable(getattr(self.vzd.GameVariable, name)))

    def roll(self, save, target_id, weapon, movement, tics):
        from doomlib.combat import AMMO_COST, WEAPON_SLOTS, fire_reaches, fire_tolerance
        game = self.game
        if game.is_player_dead() or game.is_episode_finished():
            # A load does not revive a dead player: without a new episode every later rollout 'died' at once.
            game.new_episode()
        game.load(save)
        if game.is_player_dead():
            return None  # saved on the death tic itself
        start_hp, start_armor = self.var('HEALTH'), self.var('ARMOR')
        start_dealt, start_kills = self.var('DAMAGECOUNT'), self.var('KILLCOUNT')
        selected = False
        for _ in range(tics):
            if game.is_player_dead() or game.is_episode_finished():
                break
            state = game.get_state()
            px, py, angle = self.var('POSITION_X'), self.var('POSITION_Y'), self.var('ANGLE')
            action = [0.0] * (15 if self.run else 14)
            if movement in MOVE_BUTTON:
                action[MOVE_BUTTON[movement]] = 1
                if self.run:
                    action[14] = 1
            target = next((o for o in state.objects if o.id == target_id), None)
            if target is not None:
                b = math.degrees(math.atan2(target.position_y - py, target.position_x - px))
                b = -((b - angle + 180) % 360 - 180)
                action[4] = max(-9, min(9, b))
                visible = any(label.object_id == target_id for label in state.labels)
                distance = math.hypot(target.position_x - px, target.position_y - py) / 32
                slot = int(self.var('SELECTED_WEAPON'))
                ammo = self.var('SELECTED_WEAPON_AMMO')
                action[5] = float(visible and abs(b) < fire_tolerance(distance) and ammo >= AMMO_COST.get(slot, 1)
                                  and fire_reaches(slot, distance))
            if weapon and not selected and int(self.var('SELECTED_WEAPON')) != weapon:
                action[6 + WEAPON_SLOTS[weapon]] = 1
                action[5] = 0
                selected = True
            game.make_action(action)
        died = game.is_player_dead()
        return dict(lost=max(0.0, start_hp - (0 if died else self.var('HEALTH'))) + max(0.0, start_armor - self.var('ARMOR')),
                    dealt=self.var('DAMAGECOUNT') - start_dealt, kills=self.var('KILLCOUNT') - start_kills, died=died)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('runs', nargs='+')
    p.add_argument('--tics', type=int, default=42, help='how long each movement is held (1.2 s)')
    p.add_argument('--out', required=True)
    p.add_argument('--limit', type=int, default=0, help='states per run (0: all)')
    p.add_argument('--before-death', type=float, default=0, help='only states at most this many game seconds before a death (0: all)')
    p.add_argument('--run', choices=('config', 'yes', 'no'), default='config',
                   help='run during strafes and backward movement: as the collection run did (config), or always/never')
    a = p.parse_args()
    sys.path.insert(0, str(Path.cwd()))
    from doomlib.compact_movement import compact_movement_input
    out = Path(a.out)
    done = set()
    if out.exists():
        done = {(r['run'], r['tick']) for r in map(json.loads, out.read_text().splitlines())}
    with out.open('a') as handle:
        for run in map(Path, a.runs):
            cfg = json.loads((run / 'config.json').read_text())
            name = cfg['args']['map'].upper()
            if name in HELD_OUT and not cfg['args'].get('wad'):
                raise SystemExit(f'{run}: {name} is a held-out map')
            rows = [json.loads(line) for line in (run / 'decisions.jsonl').read_text().splitlines()]
            rows = [r for r in rows if r.get('fight_state') and r['directive']['action'] == 'attack' and r['directive'].get('target')]
            if a.before_death:
                deaths = [e['game_seconds'] for e in map(json.loads, (run / 'events.jsonl').read_text().splitlines()) if e.get('event') == 'death']
                rows = [r for r in rows if any(0 <= d - r['game_seconds'] <= a.before_death for d in deaths)]
            if a.limit:
                rows = rows[::max(1, len(rows) // a.limit)][:a.limit]
            run_moves = cfg['args'].get('run_combat_movement', False) if a.run == 'config' else a.run == 'yes'
            roller, started, count = Roller(cfg, run_moves), time.perf_counter(), 0
            for r in rows:
                if (run.name, r['tick']) in done:
                    continue
                save = os.path.relpath(run / r['fight_state'])
                directive = r['directive']
                outcomes = {m: roller.roll(save, int(directive['target']['id']), directive.get('weapon'), m, a.tics) for m in MOVEMENTS}
                if None in outcomes.values():
                    continue
                scores = {m: score(o) for m, o in outcomes.items()}
                best = max(MOVEMENTS, key=lambda m: (scores[m], m == 'stationary'))
                question = r['packet']['questions']['movement']
                state, projected = compact_movement_input(r['packet']['state'], question)
                handle.write(json.dumps(dict(run=run.name, tick=r['tick'], map=name, wad=cfg['args'].get('wad'), running=run_moves,
                                             chosen=directive.get('movement') or 'stationary', outcomes=outcomes, scores=scores,
                                             label=best, state=state, question=projected, raw_state=r['packet']['state'],
                                             raw_question=question)) + '\n')
                count += 1
            roller.game.close()
            print(f'{run.name}: {count} states in {time.perf_counter() - started:.0f} s', flush=True)


if __name__ == '__main__':
    main()
