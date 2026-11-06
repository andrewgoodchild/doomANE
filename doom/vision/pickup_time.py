"""Where does pickup time go in MAP01 games? (offline diagnostic, no games, no model)

Pickup tics are grouped into episodes: consecutive telemetry rows executing pickup on the same target, across
re-decisions. Per episode: the item's name and category (from the resource planner's target, else this tic's item list), its length in tics,
whether the player came within REACH_UNITS of the item's listed position (reached), and the executor's last
detail. Names never listed in the engine games of the same map are counted as impossible (a phantom class).
Run from ext/doomLaya-dev:
  python3 ../../doom/vision/pickup_time.py pix1_engine pix3_gain pix3_oracle --out ../../results/vision/pickup_time.json
"""
import argparse
import glob
import json
import math
from collections import Counter
from pathlib import Path

REACH_UNITS = 40


def target(r, oid):
    """The selected item as the resource planner holds it (the item memory), else as listed on this tic."""
    t = (r.get('resource') or {}).get('target')
    if t and str(t.get('id')) == oid:
        return t
    return next((i for i in r['items'] if str(i['id']) == oid), None)


def episodes(run):
    out, cur = [], None
    for line in open(run + 'telemetry.jsonl'):
        r = json.loads(line)
        ex = r.get('execution') or {}
        if not isinstance(ex, dict) or ex.get('action') != 'pickup':
            cur = None
            continue
        key = str(ex.get('target_id'))
        if cur is None or cur['key'] != key:
            item = target(r, key)
            cur = dict(key=key, tics=0, name=item and item['name'], category=item and item.get('category'),
                       x=item and item['x'], y=item and item['y'], reached=False, closest=1e9, detail=None, hp=r['hp'])
            out.append(cur)
        cur['tics'] += 1
        item = target(r, key)
        if item:
            cur['x'], cur['y'] = item['x'], item['y']
        if cur['x'] is not None:
            cur['closest'] = min(cur['closest'], math.hypot(r['x'] - cur['x'], r['y'] - cur['y']) / 32)
            cur['reached'] |= cur['closest'] * 32 <= REACH_UNITS
        cur['detail'] = ex.get('status') + (': ' + ex['detail'][:60] if ex.get('status') != 'executing' and ex.get('detail') else '')
    return out


def summary(suffix, engine_names):
    runs = sorted(glob.glob(f'runs/2*-{suffix}-map01-seed*/'))
    tics = sum(1 for run in runs for _ in open(run + 'telemetry.jsonl'))
    eps = [e for run in runs for e in episodes(run)]
    ptics = sum(e['tics'] for e in eps)
    share = lambda f: round(sum(e['tics'] for e in eps if f(e)) / max(1, ptics), 3)
    lengths = sorted(e['tics'] for e in eps)
    return dict(runs=len(runs), pickup_tic_share=round(ptics / tics, 3), episodes=len(eps),
                median_episode_tics=lengths[len(lengths) // 2] if lengths else None,
                median_closest_m_unreached=sorted(round(e['closest'], 1) for e in eps if not e['reached'])[len([e for e in eps if not e['reached']]) // 2] if eps else None,
                episodes_reached=round(sum(e['reached'] for e in eps) / max(1, len(eps)), 3),
                tics_on_unreached=share(lambda e: not e['reached']),
                tics_on_impossible_names=None if engine_names is None else share(lambda e: e['name'] not in engine_names),
                tics_by_category={k: round(v / max(1, ptics), 3) for k, v in Counter(
                    c for e in eps for c in [e['category']] * e['tics']).most_common()},
                tics_by_name={k: round(v / max(1, ptics), 3) for k, v in Counter(
                    n for e in eps for n in [e['name']] * e['tics']).most_common(10)},
                end_of_unreached={k: v for k, v in Counter(e['detail'] for e in eps if not e['reached']).most_common(6)})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('suffixes', nargs='+')
    p.add_argument('--engine', default='pix1_engine')
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    engine_names = {i['name'] for run in glob.glob(f'runs/2*-{a.engine}-map01-seed*/')
                    for line in open(run + 'telemetry.jsonl') for i in json.loads(line)['items']}
    report = {s: summary(s, None if s == a.engine else engine_names) for s in a.suffixes}
    report['engine_item_names'] = sorted(engine_names)
    print(json.dumps(report, indent=1))
    if a.out:
        a.out.write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
