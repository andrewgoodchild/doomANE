"""Milestone 1: the current agent's objective choices against a map built from what the player had seen.

For each recorded run (telemetry, planner log, and the replayed frames from replay.py), the seen map is built tic by tic,
and at every objective choice in the planner log the script records whether the objective had been in view before, whether
its cell was known, whether a path through seen free space existed (strict), and the same with unknown cells passable
(optimistic) or the WAD's door sectors passable (a diagnostic that sizes the door problem). The WAD is read only to score
and to draw; the map never sees it.
    cd ext/doomLaya-dev && .venv/bin/python ../../doom/seenmap/evaluate.py --frames ../poc-data/seenmap/map01_s54_done.npz ... \
        --out ../../results/seenmap
"""
import argparse
import json
import math
import struct
import sys
import time
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mapper import CALIB, FREE, OCCUPIED, UNKNOWN, OccupancyMap, depth_units, frame_is_inside_geometry, frame_points, unpack_mask  # noqa: E402
from planner import INF, plan  # noqa: E402
from wad import MapGeometry, freedoom_wad  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / 'ext/doomLaya/runs'
CELL = 16.0
GOAL_RADIUS_CELLS = 3.0  # 48 units
SEEN_MARGIN = 32.0
COVERAGE_EVERY = 350  # tics, 10 s


def planner_log(run_dir):
    name = run_dir.name.split('_', 3)[3]
    return RUNS / f'{name}-planner.jsonl'


def load_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


WEAPON_THINGS = {2001, 82, 2002, 2003, 2004, 2005, 2006}


def objective_point(choice, row, rows_by_tick, ticks_sorted, keys_wad, things):
    """Target point (where the player goes) and seen point (what the player would see) of a chosen objective."""
    kind = choice.split('_')[0]
    if kind == 'item' and choice.startswith('item_weapon_'):
        # a WAD weapon marker: the planner's id is the THINGS index (doomlib/mission.py weapon_markers)
        idx = int(choice[len('item_weapon_'):])
        if 0 <= idx < len(things) and int(things[idx][3]) in WEAPON_THINGS:
            pt = (float(things[idx][0]), float(things[idx][1]))
            return 'weapon_marker', pt, pt
        return 'weapon_marker', None, None
    if kind == 'switch':
        sid = int(choice[len('switch_'):])
        m = next((m for m in row.get('switches', ()) if int(m['id']) == sid), None)
        if m is None:
            for t in ticks_sorted:
                m = next((m for m in rows_by_tick[t].get('switches', ()) if int(m['id']) == sid), None)
                if m:
                    break
        if m is None:
            return kind, None, None
        target = tuple(m['board']) if m.get('kind') == 'lift' else tuple(m['approach'])
        return kind, target, (float(m['x']), float(m['y']))
    if kind == 'item':
        iid = choice[len('item_'):]
        t0 = row['tick']
        for t in reversed([t for t in ticks_sorted if t <= t0]):
            it = next((i for i in rows_by_tick[t].get('items', ()) if str(i.get('id')) == iid), None)
            if it:
                return kind, (float(it['x']), float(it['y'])), (float(it['x']), float(it['y']))
        for t in ticks_sorted:
            it = next((i for i in rows_by_tick[t].get('items', ()) if str(i.get('id')) == iid), None)
            if it:
                return kind, (float(it['x']), float(it['y'])), (float(it['x']), float(it['y']))
        return kind, None, None
    if kind == 'key':
        colour = choice[len('key_'):]
        k = next(((x, y) for c, x, y in keys_wad if c == colour), None)
        return kind, k, k
    if kind == 'exit':
        ex = (row.get('mission') or {}).get('exit') or {}
        if not ex:
            return kind, None, None
        return kind, tuple(ex.get('approach') or ex['center']), tuple(ex['center'])
    return kind, None, None


def build_objectives(log_rows, tel_rows, keys_wad, level_completed, things):
    recs = [r for r in log_rows if 'choice' in r and 'question' not in r]
    last_tick = tel_rows[-1]['tick']
    recs = [r for r in recs if r['tick'] < last_tick or not level_completed]
    rows_by_tick = {r['tick']: r for r in tel_rows}
    ticks_sorted = sorted(rows_by_tick)
    tick_arr = np.array(ticks_sorted)
    out = []
    for k, r in enumerate(recs):
        t = int(r['tick'])
        row = rows_by_tick.get(t) or rows_by_tick[ticks_sorted[min(int(np.searchsorted(tick_arr, t)), len(ticks_sorted) - 1)]]
        nxt = recs[k + 1] if k + 1 < len(recs) else None
        kind, target, seen_pt = objective_point(r['choice'], row, rows_by_tick, ticks_sorted, keys_wad, things)
        if nxt is not None:
            done = nxt.get('reason') == 'objective done'
            end_tick = int(nxt['tick'])
        else:
            done = bool(level_completed and kind == 'exit')
            end_tick = ticks_sorted[-1]
        pos = (float(row['x']), float(row['y']))
        walked, died = 0.0, False
        prev = None
        for tt in ticks_sorted[int(np.searchsorted(tick_arr, t)):int(np.searchsorted(tick_arr, end_tick))]:
            rr = rows_by_tick[tt]
            if rr.get('episode', 0) != row.get('episode', 0):
                died = True
                break
            if prev is not None:
                walked += math.hypot(rr['x'] - prev['x'], rr['y'] - prev['y'])
            prev = rr
        nav = None
        for tt in ticks_sorted[int(np.searchsorted(tick_arr, t)):int(np.searchsorted(tick_arr, t + 6))]:
            n = (rows_by_tick[tt].get('navigation') or {}).get('path_length')
            if n:
                nav = n
                break
        out.append(dict(tick=t, choice=r['choice'], kind=kind, reason=r.get('reason'), probability=(r.get('probabilities') or {}).get(r['choice']),
                        player=pos, episode=row.get('episode', 0), target=target, seen_point=seen_pt, done=done, end_tick=end_tick,
                        died_before_end=died, walked_units=None if died else walked,
                        straight_units=(math.dist(pos, target) if target else None), nav_path_length_raw=nav,
                        seen_view=False, seen_view_min_distance=None, seen_view_first_tick=None))
    return out


def write_png(path, rgb):
    h, w, _ = rgb.shape
    raw = b''.join(b'\0' + rgb[y].tobytes() for y in range(h))

    def chunk(tag, body):
        return struct.pack('>I', len(body)) + tag + body + struct.pack('>I', zlib.crc32(tag + body) & 0xffffffff)
    png = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0)) + chunk(b'IDAT', zlib.compress(raw, 6)) + chunk(b'IEND', b'')
    Path(path).write_bytes(png)


def render(path, omap, state, geom, path_xy, objectives, scale=3):
    lo, hi = geom.vertices.min(0) - 64, geom.vertices.max(0) + 64
    i0, j0 = omap.cell_of(lo[0], lo[1])
    i1, j1 = omap.cell_of(hi[0], hi[1])
    i0, j0 = max(int(i0), 0), max(int(j0), 0)
    i1, j1 = min(int(i1), omap.size - 1), min(int(j1), omap.size - 1)
    W, H = (i1 - i0 + 1) * scale, (j1 - j0 + 1) * scale
    img = np.zeros((H, W, 3), np.uint8)
    sub = state[i0:i1 + 1, j0:j1 + 1]
    colours = {FREE: (40, 70, 140), OCCUPIED: (230, 60, 60)}
    for val, col in colours.items():
        ii, jj = np.nonzero(sub == val)
        for di in range(scale):
            for dj in range(scale):
                img[(j1 - j0 - jj) * scale + dj, ii * scale + di] = col
    fr = OccupancyMap.frontier(state)[i0:i1 + 1, j0:j1 + 1]
    ii, jj = np.nonzero(fr)
    img[(j1 - j0 - jj) * scale + 1, ii * scale + 1] = (80, 220, 230)

    def to_px(x, y):
        px = (np.asarray(x) - omap.x0) / omap.cell - i0
        py = (np.asarray(y) - omap.y0) / omap.cell - j0
        return (px * scale).astype(int), ((j1 - j0 + 1 - py) * scale).astype(int)

    for a, b in zip(*geom.segments()):
        n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1])) / (omap.cell / scale)) + 1
        xs, ys = to_px(np.linspace(a[0], b[0], n), np.linspace(a[1], b[1], n))
        ok = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
        img[ys[ok], xs[ok]] = (120, 120, 120)
    xs, ys = to_px(path_xy[:, 0], path_xy[:, 1])
    ok = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    img[ys[ok], xs[ok]] = (60, 220, 90)
    for o in objectives:
        if not o['target']:
            continue
        x, y = to_px(o['target'][0], o['target'][1])
        col = (255, 60, 255) if not o['seen_view'] else ((90, 255, 90) if o['done'] else (250, 210, 40))
        img[max(0, y - 3):y + 4, max(0, x - 3):x + 4] = col
        img[max(0, y - 1):y + 2, max(0, x - 1):x + 2] = (0, 0, 0)
    write_png(path, img)


def evaluate_run(frames_path, out_dir, geom_cache):
    d = np.load(frames_path)
    run_dir = RUNS / Path(str(d['run'])).name
    doom_map = str(d['map'])
    tel_rows = [r for r in load_jsonl(run_dir / 'telemetry.jsonl') if r.get('map', doom_map) == doom_map]
    summary = json.loads((run_dir / 'summary.json').read_text())
    level_completed = summary.get('levels_completed', 0) >= 1
    if doom_map not in geom_cache:
        g = MapGeometry(freedoom_wad(), doom_map)
        geom_cache[doom_map] = (g, g.door_sectors(), g.key_markers())
    geom, door_sectors, keys_wad = geom_cache[doom_map]
    objectives = build_objectives(load_jsonl(planner_log(run_dir)), tel_rows, keys_wad, level_completed, geom.things)
    depth, mask, poses, ticks = d['depth'], d['mask'], d['poses'], d['tick']
    omap = OccupancyMap((tel_rows[0]['x'], tel_rows[0]['y']), cell=CELL)
    # WAD-derived grids for scoring only: door cells and walkable cells inside the geometry's bounding box
    lo, hi = geom.vertices.min(0) - 64, geom.vertices.max(0) + 64
    i0, j0 = omap.cell_of(lo[0], lo[1]); i1, j1 = omap.cell_of(hi[0], hi[1])
    ii, jj = np.meshgrid(np.arange(int(i0), int(i1) + 1), np.arange(int(j0), int(j1) + 1), indexing='ij')
    cx_, cy_ = omap.point_of(ii.ravel(), jj.ravel())
    sec = geom.sector_at(np.c_[cx_, cy_])
    door_cells = np.zeros((omap.size, omap.size), bool)
    walkable = np.zeros((omap.size, omap.size), bool)
    is_door = np.array([s in door_sectors for s in sec.tolist()])
    ok_sec = sec >= 0
    headroom = np.zeros(len(sec), bool)
    headroom[ok_sec] = (geom.ceiling[sec[ok_sec]] - geom.floor[sec[ok_sec]]) >= 56
    headroom |= is_door & ok_sec  # a closed door's sector has no headroom in the static WAD, but the player walks through it
    door_cells[ii.ravel(), jj.ravel()] = is_door & ok_sec
    walkable[ii.ravel(), jj.ravel()] = ok_sec & headroom
    n_walkable = int(walkable.sum())
    # objective points as arrays for the view test
    pending = [k for k, o in enumerate(objectives) if o['seen_point'] is not None]
    seen_pts = np.array([objectives[k]['seen_point'] for k in pending], float) if pending else np.zeros((0, 2))
    seen_tick = np.array([objectives[k]['tick'] for k in pending]) if pending else np.zeros(0, int)
    fx, cx = CALIB['fx'], CALIB['cx']
    # frame loop
    tel_i, ob_i = 0, 0
    coverage, map_ms, plan_ms, skipped = [], [], [], 0
    next_cov = 0
    results = []

    def evaluate_objective(o, state):
        if o['target'] is None:
            o.update(unresolved=True)
            return
        si, sj = omap.cell_of(*o['player']); ti, tj = omap.cell_of(*o['target'])
        si, sj, ti, tj = int(si), int(sj), int(ti), int(tj)
        nb = state[max(0, ti - 1):ti + 2, max(0, tj - 1):tj + 2]
        o['seen_map'] = bool((nb != UNKNOWN).any())
        o['target_cell_state'] = int(state[ti, tj])
        t0 = time.perf_counter()
        strict = plan(state, (si, sj), (ti, tj), GOAL_RADIUS_CELLS)
        optimistic = plan(state, (si, sj), (ti, tj), GOAL_RADIUS_CELLS, unknown_cost=2.0)
        doors = plan(state, (si, sj), (ti, tj), GOAL_RADIUS_CELLS, passable=door_cells & (state != FREE))
        plan_ms.append((time.perf_counter() - t0) * 1000 / 3)
        valid_state = np.where(walkable, state, OCCUPIED).astype(np.uint8)  # WAD diagnostic: phantom free cells removed
        strict_valid = plan(valid_state, (si, sj), (ti, tj), GOAL_RADIUS_CELLS)
        o['path_strict_wadvalid_units'] = None if strict_valid is None else strict_valid * CELL
        o['path_strict_units'] = None if strict is None else strict * CELL
        o['path_optimistic_units'] = None if optimistic is None else optimistic * CELL
        o['path_doors_units'] = None if doors is None else doors * CELL
        o['free_cells'] = int((state == FREE).sum()); o['known_cells'] = int((state != UNKNOWN).sum())
        o['frontier_cells'] = int(OccupancyMap.frontier(state).sum())

    def walk_rows_until(T):
        nonlocal tel_i
        while tel_i < len(tel_rows) and tel_rows[tel_i]['tick'] <= T:
            r = tel_rows[tel_i]
            prev = tel_rows[tel_i - 1] if tel_i else None
            if prev is not None and prev.get('episode', 0) == r.get('episode', 0) and math.hypot(r['x'] - prev['x'], r['y'] - prev['y']) < 128:
                omap.walk_segment(prev['x'], prev['y'], r['x'], r['y'])
            else:
                omap.walk(r['x'], r['y'])
            tel_i += 1

    for f in range(len(ticks)):
        T = int(ticks[f])
        walk_rows_until(T)
        while ob_i < len(objectives) and objectives[ob_i]['tick'] <= T:
            evaluate_objective(objectives[ob_i], omap.state())
            ob_i += 1
        if T >= next_cov:
            st = omap.state()
            coverage.append(dict(tick=T, free=int((st == FREE).sum()), known=int((st != UNKNOWN).sum()), walkable=n_walkable))
            next_cov += COVERAGE_EVERY
        pose = poses[f]
        dep = depth[f]
        if frame_is_inside_geometry(dep, pose):
            skipped += 1
            continue
        t0 = time.perf_counter()
        pts, cls = frame_points(dep, unpack_mask(mask[f]), pose, stride=2)
        omap.update(pts, cls, float(pose[2]), camera_xy=(float(pose[0]), float(pose[1])))
        map_ms.append((time.perf_counter() - t0) * 1000)
        # view test for objectives not yet chosen
        if len(pending):
            ahead = seen_tick > T
            if ahead.any():
                dx = seen_pts[ahead, 0] - pose[0]; dy = seen_pts[ahead, 1] - pose[1]
                dist = np.hypot(dx, dy)
                bearing = (np.degrees(np.arctan2(dy, dx)) - pose[3] + 180) % 360 - 180
                in_fov = np.abs(bearing) <= 45
                u = np.clip((cx - fx * np.tan(np.radians(bearing))).astype(int), 1, 318)
                horizon = depth_units(np.array([np.median(dep[98:103, c - 1:c + 2]) for c in u]))
                vis = in_fov & (horizon > 0) & (dist <= horizon + SEEN_MARGIN) & (dist <= depth_units(CALIB['depth_cap']))
                for idx, ok in zip(np.flatnonzero(ahead)[vis], dist[vis]):
                    o = objectives[pending[idx]]
                    if not o['seen_view']:
                        o['seen_view'], o['seen_view_first_tick'] = True, T
                    o['seen_view_min_distance'] = ok if o['seen_view_min_distance'] is None else min(o['seen_view_min_distance'], ok)
    walk_rows_until(tel_rows[-1]['tick'])
    state = omap.state()
    while ob_i < len(objectives):
        evaluate_objective(objectives[ob_i], state); ob_i += 1
    coverage.append(dict(tick=int(ticks[-1]), free=int((state == FREE).sum()), known=int((state != UNKNOWN).sum()), walkable=n_walkable))
    path_xy = np.array([(r['x'], r['y']) for r in tel_rows])
    name = Path(frames_path).stem
    render(out_dir / f'{name}.png', omap, state, geom, path_xy, objectives)
    for o in objectives:
        for k in ('player', 'target', 'seen_point'):
            if o.get(k) is not None:
                o[k] = [float(v) for v in o[k]]
        if o.get('seen_view_min_distance') is not None:
            o['seen_view_min_distance'] = float(o['seen_view_min_distance'])
    return dict(name=name, run=run_dir.name, map=doom_map, level_completed=level_completed, frames=int(len(ticks)), frames_skipped=skipped,
                stride=int(d['stride']), tics=len(tel_rows), objectives=objectives, coverage=coverage,
                mapping_ms=dict(median=float(np.median(map_ms)), p90=float(np.percentile(map_ms, 90))),
                planning_ms_per_call=dict(median=float(np.median(plan_ms)) if plan_ms else None, p90=float(np.percentile(plan_ms, 90)) if plan_ms else None),
                final=dict(free=int((state == FREE).sum()), known=int((state != UNKNOWN).sum()), walkable=n_walkable))


def share(xs):
    xs = [x for x in xs if x is not None]
    return (sum(1 for x in xs if x) / len(xs)) if xs else None


def summarise(runs):
    def table(objs):
        objs = [o for o in objs if o['kind'] != 'explore']
        resolved = [o for o in objs if not o.get('unresolved')]
        row = dict(objectives=len(objs), unresolved=len(objs) - len(resolved),
                   seen_view=share([o['seen_view'] for o in resolved]), seen_map=share([o.get('seen_map') for o in resolved]),
                   reachable_strict=share([o.get('path_strict_units') is not None for o in resolved]),
                   reachable_optimistic=share([o.get('path_optimistic_units') is not None for o in resolved]),
                   reachable_doors=share([o.get('path_doors_units') is not None for o in resolved]),
                   reachable_strict_wadvalid=share([o.get('path_strict_wadvalid_units') is not None for o in resolved]),
                   done=share([o['done'] for o in resolved]))
        ratios = [o['path_strict_units'] / o['straight_units'] for o in resolved if o.get('path_strict_units') is not None and o['straight_units']]
        row['path_over_straight_median'] = float(np.median(ratios)) if ratios else None
        walked = [o['walked_units'] / o['path_strict_units'] for o in resolved
                  if o['done'] and o.get('walked_units') and o.get('path_strict_units')]
        row['walked_over_path_median_done'] = float(np.median(walked)) if walked else None
        row['walked_over_path_n'] = len(walked)
        by_kind = {}
        for k in ('switch', 'item', 'weapon_marker', 'key', 'exit'):
            ks = [o for o in resolved if o['kind'] == k]
            if ks:
                by_kind[k] = dict(n=len(ks), seen_view=share([o['seen_view'] for o in ks]), seen_map=share([o.get('seen_map') for o in ks]),
                                  reachable_strict=share([o['path_strict_units'] is not None for o in ks]),
                                  reachable_strict_wadvalid=share([o.get('path_strict_wadvalid_units') is not None for o in ks]),
                                  reachable_optimistic=share([o['path_optimistic_units'] is not None for o in ks]), reachable_doors=share([o['path_doors_units'] is not None for o in ks]),
                                  done=share([o['done'] for o in ks]))
        row['by_kind'] = by_kind
        row['explore_choices'] = None
        return row

    out = dict(runs={}, by_map={}, overall=None)
    all_objs = []
    for r in runs:
        objs = r['objectives']
        t = table(objs)
        t['explore_choices'] = sum(1 for o in objs if o['kind'] == 'explore')
        t.update(frames=r['frames'], frames_skipped=r['frames_skipped'], mapping_ms=r['mapping_ms'], planning_ms_per_call=r['planning_ms_per_call'],
                 final_free_share_of_walkable=r['final']['free'] / max(r['final']['walkable'], 1), level_completed=r['level_completed'], map=r['map'])
        out['runs'][r['name']] = t
        all_objs += objs
    for m in sorted({r['map'] for r in runs}):
        objs = [o for r in runs if r['map'] == m for o in r['objectives']]
        t = table(objs); t['explore_choices'] = sum(1 for o in objs if o['kind'] == 'explore'); out['by_map'][m] = t
    t = table(all_objs); t['explore_choices'] = sum(1 for o in all_objs if o['kind'] == 'explore')
    t['mapping_ms_median_over_runs'] = float(np.median([r['mapping_ms']['median'] for r in runs]))
    t['planning_ms_median_over_runs'] = float(np.median([r['planning_ms_per_call']['median'] for r in runs if r['planning_ms_per_call']['median'] is not None]))
    out['overall'] = t
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--frames', type=Path, nargs='+', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--large', type=Path, default=None, help='directory for per-run detail JSON (default: --out)')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    detail_dir = a.large or a.out
    geom_cache, runs = {}, []
    for f in a.frames:
        t0 = time.perf_counter()
        r = evaluate_run(f, a.out, geom_cache)
        runs.append(r)
        (detail_dir / f'{r["name"]}.json').write_text(json.dumps(r, indent=1))
        ov = summarise([r])['overall']
        print(f"{r['name']}: {r['frames']} frames ({r['frames_skipped']} skipped), {ov['objectives']} objectives, seen {ov['seen_view']}, "
              f"strict {ov['reachable_strict']}, optimistic {ov['reachable_optimistic']}, doors {ov['reachable_doors']}, "
              f"map {r['mapping_ms']['median']:.1f} ms, plan {r['planning_ms_per_call']['median']} ms, {time.perf_counter() - t0:.0f} s", flush=True)
    summary = summarise(runs)
    (a.out / 'milestone1.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary['overall'], indent=1))


if __name__ == '__main__':
    main()
