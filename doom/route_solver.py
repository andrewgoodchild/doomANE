"""Static route check: can the map's progression be followed from the start with the generic mechanism rules?

From the player start, repeat until nothing changes: collect every key that is reachable (by touch, as the
executor does), use every reachable unused mechanism and apply its move from doomlib/specials.py, ride
reachable lifts both ways, then test the exit. Only the WAD and the engine's own rules are used, the same
for every map, so the score says how much of a map's route knowledge the stack has, without a walkthrough.

Optimistic: reach is a union over everywhere the player has been, so a one-way drop that would strand
the player is not noticed, and monsters, timing and crushers are ignored.

Held-out maps (doom/map_split.json) are refused. Run from ext/doomLaya-dev:
  .venv/bin/python ../../doom/route_solver.py MAP01 MAP02 --extended --out ../../results/doom/route_solver.json
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
HELD_OUT = set(json.loads((REPO / 'doom/map_split.json').read_text())['holdout'])


def start_state(wad, name, scenario=None):
    import vizdoom as vzd
    game = vzd.DoomGame()
    game.set_doom_game_path(str(wad))
    if scenario:
        game.set_doom_scenario_path(str(scenario))
    game.set_doom_map(name)
    game.set_window_visible(False)
    game.set_sectors_info_enabled(True)
    game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
    game.add_game_args('-skill 3')
    game.init()
    game.new_episode()
    sectors = game.get_state().sectors
    start = (game.get_game_variable(vzd.GameVariable.POSITION_X), game.get_game_variable(vzd.GameVariable.POSITION_Y))
    game.close()
    return sectors, start


class Solver:
    def __init__(self, sectors, data):
        self.lines = [s.lines for s in sectors]
        self.heights = [(float(s.floor_height), float(s.ceiling_height)) for s in sectors]
        self.data = data
        from doomlib.mission import navigator_doors
        self.doors = navigator_doors(data)
        self.cache = {}

    def navigator(self, heights):
        from doomlib.navigation import Navigator
        key = tuple(heights)
        if key not in self.cache:
            trial = [SimpleNamespace(floor_height=f, ceiling_height=c, lines=lines) for (f, c), lines in zip(heights, self.lines)]
            self.cache[key] = Navigator(trial, self.doors, self.data['teleports'])
        return self.cache[key]

    def reach(self, heights, seeds, keys):
        nav = self.navigator(heights)
        closed = sorted({i for m in self.data['switches'] if m['kind'] == 'door' for i in m['sectors']
                         if heights[i][1] - heights[i][0] < 56})
        nav.keys = None
        nav.observe(dict(x=seeds[0][0], y=seeds[0][1], keys=keys, closed_remote_doors=closed), 0)
        cells = set()
        for seed in seeds:
            cells |= nav.reachable(seed)
        return nav, cells

    @staticmethod
    def at(nav, point, cells, contact=False):
        if nav.nearest(point) in cells:
            return True
        if contact:
            touch = nav.pickup_point(dict(x=point[0], y=point[1]), cells)
            return tuple(touch) != tuple(point) and nav.nearest(touch) in cells
        return False

    def solve(self, start, limit=200):
        heights, keys, used, seeds, steps = list(self.heights), set(), set(), [start], []
        exit_line = next((e for e in self.data['exits'] if not e['secret']), None)
        for _ in range(limit):
            nav, cells = self.reach(heights, seeds, keys)
            if exit_line and (self.at(nav, exit_line['approach'], cells) or self.at(nav, exit_line['center'], cells)):
                steps.append('exit')
                return dict(solved=True, steps=steps)
            progress = False
            for key in self.data['key_markers']:
                if key['color'] not in keys and self.at(nav, (key['x'], key['y']), cells, contact=True):
                    keys.add(key['color'])
                    steps.append(f"{key['color']} key")
                    progress = True
            if progress:
                continue
            for m in self.data['switches']:
                if m['id'] in used or (m['key'] and m['key'] not in keys):
                    continue
                points = [m['approach']] + ([m['board']] if m.get('trigger') == 'walk' or m['kind'] == 'lift' else [])
                if not any(self.at(nav, p, cells) for p in points):
                    continue
                used.add(m['id'])
                if m['kind'] == 'lift':
                    # Riding both ways: the lift's cells are reachable at its low and its high end.
                    low = list(heights)
                    for i, (part, h) in m.get('moves', {}).items():
                        low[i] = (h, low[i][1])
                    lift_cells = [c for i in m['sectors'] for c in nav.sector_cells.get(i, ())]
                    if lift_cells:
                        seeds.append(nav.point(lift_cells[0]))
                        _, below = self.reach(low, [nav.point(lift_cells[0])], keys)
                        nav_low = self.navigator(low)
                        seeds += [nav_low.point(c) for c in list(below)[::max(1, len(below) // 50)]]
                    steps.append(f"lift #{m['id']}")
                else:
                    moved = False
                    for i, (part, h) in m.get('moves', {}).items():
                        f, c = heights[i]
                        new = (h, c) if part == 'floor' else (f, h)
                        if new != heights[i]:
                            heights[i] = new
                            moved = True
                    steps.append(f"{m['kind']} #{m['id']} ({m.get('special')})" + ('' if moved else ' no move'))
                progress = True
                break
            if not progress:
                break
        nav, cells = self.reach(heights, seeds, keys)
        self.last = (nav, cells, seeds[0])
        missing = [k['color'] for k in self.data['key_markers'] if k['color'] not in keys]
        unused = [f"{m['kind']} #{m['id']} ({m.get('special')}){' needs ' + m['key'] if m['key'] and m['key'] not in keys else ''}"
                  for m in self.data['switches'] if m['id'] not in used]
        return dict(solved=False, steps=steps, missing_keys=sorted(set(missing)), unused=unused[:30], reached_cells=len(cells))


def cut(nav, cells, start, point):
    """Why the rules stop the player short of point: the first unreached cell on a plain 2D path to it."""
    from collections import deque
    goal, origin = nav.nearest(point), nav.nearest(start)
    if goal is None:
        return 'no walkable cell near the target'
    previous, queue = {origin: None}, deque([origin])
    while queue and goal not in previous:
        c = queue.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (c[0] + dx, c[1] + dy)
            if n in nav.free and n not in previous:
                previous[n] = c
                queue.append(n)
    if goal not in previous:
        return 'no 2D path through walkable cells (geometry, e.g. a closed sector or a narrow gap)'
    path, c = [], goal
    while c is not None:
        path.append(c)
        c = previous[c]
    path.reverse()
    i = next(i for i, c in enumerate(path) if c not in cells)
    c, p = path[i], path[i - 1]
    x, y = nav.point(c)
    reason = 'locked (closed door or missing key)' if c in nav.locked else (
        f'step up {nav.floors[c] - nav.floors[p]:.0f}' if nav.floors[c] - nav.floors[p] > 24 else 'rejected edge')
    return f'at ({x:.0f}, {y:.0f}) sectors {nav.body_sectors.get(c)}: {reason}'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('maps', nargs='+')
    p.add_argument('--extended', action='store_true', help='use every mechanism of the specials table')
    p.add_argument('--scenario', help='PWAD with the map (e.g. an Obsidian map); the IWAD is freedoom2')
    p.add_argument('--diagnose', action='store_true', help='for a stuck map, show where the rules stop each missing target')
    p.add_argument('--out')
    a = p.parse_args()
    sys.path.insert(0, str(Path.cwd()))
    import vizdoom
    from doomlib.mission import map_data
    wad = Path(vizdoom.__file__).parent / 'freedoom2.wad'
    results = {}
    for name in a.maps:
        if name.upper() in HELD_OUT and not a.scenario:
            raise SystemExit(f'{name} is a held-out map')
        started = time.perf_counter()
        sectors, start = start_state(wad, name, a.scenario)
        data = map_data(a.scenario or wad, name, extended=a.extended)
        solver = Solver(sectors, data)
        result = solver.solve(start)
        if not result['solved'] and a.diagnose:
            nav, cells, origin = solver.last
            targets = {f"{k['color']} key": (k['x'], k['y']) for k in data['key_markers']}
            targets.update({f"exit #{e['line']}": e['approach'] for e in data['exits'] if not e['secret']})
            result['cuts'] = {name: cut(nav, cells, origin, point) for name, point in targets.items()
                              if not Solver.at(nav, point, cells, contact=name.endswith('key'))}
            for name, why in result['cuts'].items():
                print(f'  {name}: {why}')
        result['seconds'] = round(time.perf_counter() - started, 1)
        results[name] = result
        print(f"{name}: {'SOLVED' if result['solved'] else 'stuck'} in {len(result['steps'])} steps ({result['seconds']} s): "
              + ', '.join(result['steps'][:14]) + ('' if result['solved'] else f"; missing keys {result['missing_keys']}"))
    solved = sum(r['solved'] for r in results.values())
    print(f"solved {solved}/{len(results)}")
    if a.out:
        Path(a.out).write_text(json.dumps(dict(extended=a.extended, scenario=a.scenario, results=results), indent=1))


if __name__ == '__main__':
    main()
