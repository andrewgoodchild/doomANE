"""Decision-time features for the judgement POC, parsed from the text Laya saw at a movement decision.

Every feature comes from the record's `state`, `raw_state` and `raw_question.instructions` (the combat facts),
all of which the agent had before the rollout. Nothing from `outcomes`, `scores` or `label` is read here.
"""
import re

MOVES = ['stationary', 'strafe_left', 'strafe_right', 'backward']
HITSCAN = {'Zombieman', 'ShotgunGuy', 'ChaingunGuy', 'WolfensteinSS'}
MELEE = {'Demon', 'Spectre', 'LostSoul'}
# everything else throws projectiles (DoomImp, HellKnight, BaronOfHell, Cacodemon, Revenant, Arachnotron, Fatso, ...)
WEAPONS = ['melee', 'pistol', 'shotgun', 'super_shotgun', 'chaingun', 'rocket_launcher', 'plasma_rifle', 'bfg']
RANGES = ['melee', 'close', 'medium', 'far']
HEALTHS = ['healthy', 'wounded', 'critical']
COMMANDS = ['attack', 'pickup', 'use_switch', 'explore', 'retreat', 'wait', 'look_back', 'open_door', 'exit']
SIDES = ['left', 'right', 'center']
FAR = 99.0

_ENEMY = re.compile(r'([A-Za-z]+)#\d+ ([\d.]+)m \((visible|last seen)\)')
_ITEM = re.compile(r'([A-Za-z]+)#\S+ \[(\w+)\] ([\d.]+)m')


def _num(pattern, text, default=None, flags=re.M):
    m = re.search(pattern, text, flags)
    return float(m.group(1)) if m else default


def _yes(pattern, text):
    m = re.search(pattern, text, re.M)
    return 1.0 if m and m.group(1) == 'yes' else 0.0


def features(record, include_chosen=True):
    s, raw = record['state'], record['raw_state']
    f = {}
    # --- the movement question's own state (7 lines) ---
    f['known_enemies'] = _num(r'^Known enemies: (\d+)', s, 0.0)
    f['visible_enemies'] = _num(r'visible enemies: (\d+)\.', s, 0.0)
    f['nearest_threat_m'] = _num(r'remembered threat if none visible: ([\d.]+) meters', s, FAR)
    f['threat_within_12'] = _yes(r'closer than twelve meters: (yes|no)', s)
    for side in ('left', 'right', 'back'):
        f[f'clear_{side}_m'] = _num(rf'^Body clearance {side}: ([\d.]+) meters', s, 0.0)
    move = re.search(r'^Current movement: (\w+)\.', s, re.M)
    for m in MOVES:
        f[f'cur_{m}'] = 1.0 if move and move.group(1) == m else 0.0
    # --- the full packet state ---
    f['hp'] = _num(r'^HP (\d+); armor (\d+)', raw, 100.0)
    f['armor'] = _num(r'^HP \d+; armor (\d+)', raw, 0.0)
    cmd = re.search(r'^Current command: (\w+)(?: #\S+ \S+ ([\d.]+)m)?', raw, re.M)
    for c in COMMANDS:
        f[f'cmd_{c}'] = 1.0 if cmd and cmd.group(1) == c else 0.0
    f['cmd_target_m'] = float(cmd.group(2)) if cmd and cmd.group(2) else FAR
    rng = re.search(r'^Enemy range: (\w+); health: (\w+); space: (.*)\.$', raw, re.M)
    for r in RANGES:
        f[f'range_{r}'] = 1.0 if rng and rng.group(1) == r else 0.0
    for h in HEALTHS:
        f[f'health_{h}'] = 1.0 if rng and rng.group(2) == h else 0.0
    space = rng.group(3) if rng else ''
    f['space_left_open'] = 1.0 if 'left open' in space else 0.0
    f['space_right_open'] = 1.0 if 'right open' in space else 0.0
    inv = dict(re.findall(r'(\w+) (\d+) ammo', re.search(r'^Inventory: (.*)$', raw, re.M).group(1))) if re.search(r'^Inventory:', raw, re.M) else {}
    for w in WEAPONS:
        f[f'ammo_{w}'] = float(inv.get(w, 0))
        f[f'owned_{w}'] = 1.0 if w in inv else 0.0
    f['ammo_total'] = sum(float(v) for k, v in inv.items() if k != 'melee')
    f['keys'] = 0.0 if re.search(r'^Collected keys: none', raw, re.M) else float(len(re.findall(r'\w+', (re.search(r'^Collected keys: (.*)$', raw, re.M) or [None, ''])[1] if re.search(r'^Collected keys:', raw, re.M) else '')))
    enemies = _ENEMY.findall((re.search(r'^Enemies: (.*)$', raw, re.M) or [None, ''])[1])
    vis = [(n, float(d)) for n, d, v in enemies if v == 'visible']
    mem = [(n, float(d)) for n, d, v in enemies if v != 'visible']
    f['n_vis'] = float(len(vis))
    f['n_mem'] = float(len(mem))
    f['nearest_vis_m'] = min([d for _, d in vis], default=FAR)
    f['nearest_mem_m'] = min([d for _, d in mem], default=FAR)
    for lim in (5, 10, 20):
        f[f'vis_within_{lim}'] = float(sum(1 for _, d in vis if d <= lim))
    f['vis_hitscan'] = float(sum(1 for n, _ in vis if n in HITSCAN))
    f['vis_melee'] = float(sum(1 for n, _ in vis if n in MELEE))
    f['vis_projectile'] = float(sum(1 for n, _ in vis if n not in HITSCAN and n not in MELEE))
    f['threat_sum'] = sum(1.0 / max(d, 1.0) for _, d in vis)
    items = _ITEM.findall((re.search(r'^Items: (.*)$', raw, re.M) or [None, ''])[1])
    health = [float(d) for n, c, d in items if c == 'Health']
    f['nearest_health_m'] = min(health, default=FAR)
    f['n_health_items'] = float(len(health))
    # --- combat facts from the question's instructions ---
    cf = dict(re.findall(r'(\w+)=([^;.]+)', re.search(r'Combat facts: (.*)$', record['raw_question']['instructions']).group(1)))
    f['incoming'] = float(cf.get('incoming', 0) or 0)
    secs = cf.get('incoming_seconds', 'none')
    f['incoming_seconds'] = float(secs) if secs not in ('none', None) else -1.0
    f['incoming_known'] = 0.0 if secs in ('none', None) else 1.0
    f['incoming_miss_yes'] = 1.0 if cf.get('incoming_miss') == 'yes' else 0.0
    f['incoming_miss_no'] = 1.0 if cf.get('incoming_miss') == 'no' else 0.0
    for side in SIDES:
        f[f'incoming_{side}'] = 1.0 if cf.get('incoming_side') == side else 0.0
    f['melee_nearest'] = float(cf.get('melee_nearest', FAR))
    f['hitscan_visible'] = float(cf.get('hitscan_visible', 0))
    # --- Laya's own choice, known at decision time (ablated separately) ---
    if include_chosen:
        for m in MOVES:
            f[f'chosen_{m}'] = 1.0 if record['chosen'] == m else 0.0
    return f


def label_a(record):
    o = record['outcomes'][record['chosen']]
    return 1 if (o['died'] or o['lost'] >= 20) else 0


def label_died(record):
    return 1 if record['outcomes'][record['chosen']]['died'] else 0


def vectorize(records, include_chosen=True):
    import numpy as np
    rows = [features(r, include_chosen) for r in records]
    names = list(rows[0].keys())
    X = np.array([[row[n] for n in names] for row in rows], dtype=np.float32)
    return X, names
