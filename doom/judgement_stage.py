"""Read the phase 3 judgement stages (docs/LAB_NOTEBOOK.md, 2026-11-08): paired scenarios, then the MAP04 gate.

scenarios: pairs the two arms of a doom/scenarios.py output on the same saved states and reports deaths, health
lost, damage dealt, the discordant pairs and a one-sided sign test (binomial, p = 0.5) on them, against the
pre-registered rule (judgement deaths <= 80 % of the baseline's and p <= 0.10). Mechanism numbers come from the
judgement arm's run directories: attack decisions answered yes, fall-back tics, health change per tic during
fall-back against other attack tics, and the authority audit on every judgement run.

games: the same reading for a repeat_map.py result file with both configs (paired by seed): total deaths, the
seed-wise sign, completions and the time to the first key.

Run from ext/doomLaya-dev (the run directories are resolved against it):
  .venv/bin/python ../../doom/judgement_stage.py scenarios --out ../../results/doom/scenarios-judge.jsonl
  .venv/bin/python ../../doom/judgement_stage.py games --result ../../results/doom/repeat_map04_judge.json
"""
import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

BASE, JUDGE = 'planner_ext_safe', 'planner_ext_safe_judge'


def sign_test(worse, better):
    """One-sided binomial probability of at most `better` successes... reported as P(X <= better) where X ~ Bin(n, 0.5)
    counts the discordant pairs in which the judgement arm did worse; small when the judgement arm is better."""
    n = worse + better
    if n == 0:
        return 1.0
    return sum(math.comb(n, k) for k in range(0, worse + 1)) / 2 ** n


def mechanism(run_dir):
    """Judgement numbers of one run: decisions, yes answers, fall-back tics, health change per tic by movement."""
    decisions = [json.loads(l) for l in (run_dir / 'decisions.jsonl').read_text().splitlines()]
    rows = [json.loads(l) for l in (run_dir / 'telemetry.jsonl').read_text().splitlines()]
    attack = [d for d in decisions if d['directive']['action'] == 'attack']
    judged = [d for d in attack if d.get('judgement')]
    yes = [d for d in judged if d['judgement']['noul'] >= d['judgement']['threshold']]
    applied = [d for d in judged if d['judgement']['applied']]
    capped = [d for d in yes if not d['judgement']['applied']]
    hp_change = Counter()
    tics = Counter()
    previous = None
    for row in rows:
        execution = row.get('execution') or {}
        if previous is not None and row['episode'] == previous['episode'] and execution.get('action') == 'attack':
            kind = 'fall_back' if execution.get('movement') == 'fall_back' else 'other_attack'
            hp_change[kind] += (row['hp'] + row['armor']) - (previous['hp'] + previous['armor'])
            tics[kind] += 1
        previous = row
    runs, length = 0, []
    current = 0
    for row in rows:
        if (row.get('execution') or {}).get('movement') == 'fall_back':
            current += 1
        elif current:
            runs += 1
            length.append(current)
            current = 0
    if current:
        runs += 1
        length.append(current)
    return dict(decisions=len(decisions), attack_decisions=len(attack), judged=len(judged), yes=len(yes), applied=len(applied), capped=len(capped),
                probabilities=[round(d['judgement']['noul'], 3) for d in judged],
                fall_back_tics=tics['fall_back'], other_attack_tics=tics['other_attack'],
                hp_per_tic_fall_back=round(hp_change['fall_back'] / tics['fall_back'], 4) if tics['fall_back'] else None,
                hp_per_tic_other_attack=round(hp_change['other_attack'] / tics['other_attack'], 4) if tics['other_attack'] else None,
                fall_backs=runs, mean_fall_back_tics=round(sum(length) / len(length), 1) if length else None)


def audit(run_dir):
    sys.path.insert(0, str(Path.cwd()))
    from tools.check_authority import check
    result = check(run_dir)
    return dict(passed=result['pass'], errors=result['errors'][:5])


def scenarios(a):
    rows = [json.loads(l) for l in a.out.read_text().splitlines()]
    by = {}
    for r in rows:
        if r.get('command', 'model') == 'model':
            by.setdefault(r['state'], {})[r['config']] = r
    paired = {s: v for s, v in by.items() if BASE in v and JUDGE in v}
    base = [v[BASE] for v in paired.values()]
    judge = [v[JUDGE] for v in paired.values()]
    deaths = {BASE: sum(r['died'] for r in base), JUDGE: sum(r['died'] for r in judge)}
    worse = sum(1 for v in paired.values() if v[JUDGE]['died'] and not v[BASE]['died'])
    better = sum(1 for v in paired.values() if v[BASE]['died'] and not v[JUDGE]['died'])
    p = sign_test(worse, better)
    ratio = deaths[JUDGE] / deaths[BASE] if deaths[BASE] else float('nan')
    result = dict(states=len(paired), deaths=deaths, death_ratio=round(ratio, 3), discordant=dict(judge_worse=worse, judge_better=better), sign_test_p=round(p, 4),
                  rule=dict(death_ratio_max=0.8, sign_test_p_max=0.10), passed=bool(deaths[BASE] and ratio <= 0.8 and p <= 0.10),
                  health_lost={BASE: round(sum(r['lost'] for r in base) / len(base), 1), JUDGE: round(sum(r['lost'] for r in judge) / len(judge), 1)},
                  dealt={BASE: round(sum(r['dealt'] for r in base) / len(base), 1), JUDGE: round(sum(r['dealt'] for r in judge) / len(judge), 1)},
                  per_map={})
    for m in sorted({r['map'] for r in base}):
        b = [v for v in paired.values() if v[BASE]['map'] == m]
        result['per_map'][m] = dict(states=len(b), deaths={BASE: sum(v[BASE]['died'] for v in b), JUDGE: sum(v[JUDGE]['died'] for v in b)})
    mech = Counter()
    probabilities = []
    audits = Counter()
    audit_errors = Counter()
    hp = {'fall_back': [0.0, 0], 'other_attack': [0.0, 0]}
    lengths = []
    for r in judge:
        run_dir = Path('runs') / r['scenario']
        if not run_dir.exists():
            continue
        m = mechanism(run_dir)
        for k in ('decisions', 'attack_decisions', 'judged', 'yes', 'applied', 'capped', 'fall_back_tics', 'other_attack_tics', 'fall_backs'):
            mech[k] += m[k]
        probabilities += m['probabilities']
        if m['hp_per_tic_fall_back'] is not None:
            hp['fall_back'][0] += m['hp_per_tic_fall_back'] * m['fall_back_tics']
            hp['fall_back'][1] += m['fall_back_tics']
        if m['hp_per_tic_other_attack'] is not None:
            hp['other_attack'][0] += m['hp_per_tic_other_attack'] * m['other_attack_tics']
            hp['other_attack'][1] += m['other_attack_tics']
        if m['mean_fall_back_tics'] is not None:
            lengths.append(m['mean_fall_back_tics'] * m['fall_backs'])
        if a.audit:
            verdict = audit(run_dir)
            audits['passed' if verdict['passed'] else 'failed'] += 1
            for e in verdict['errors']:
                audit_errors[e] += 1
    result['mechanism'] = dict(mech, share_yes=round(mech['yes'] / mech['judged'], 4) if mech['judged'] else None,
                               probability_quartiles=[round(sorted(probabilities)[int(q * (len(probabilities) - 1))], 3) for q in (0.25, 0.5, 0.75)] if probabilities else None,
                               hp_per_tic_fall_back=round(hp['fall_back'][0] / hp['fall_back'][1], 4) if hp['fall_back'][1] else None,
                               hp_per_tic_other_attack=round(hp['other_attack'][0] / hp['other_attack'][1], 4) if hp['other_attack'][1] else None,
                               mean_fall_back_tics=round(sum(lengths) / mech['fall_backs'], 1) if mech['fall_backs'] else None,
                               authority=dict(audits), authority_errors=dict(audit_errors))
    print(json.dumps(result, indent=1))
    if a.save:
        a.save.write_text(json.dumps(result, indent=1))


def games(a):
    report = json.loads(a.result.read_text())
    runs = [r for r in report['runs'] if r['valid'] and r.get('authority', True)]
    by_seed = {}
    for r in runs:
        by_seed.setdefault(r['seed'], {})[r['config']] = r
    paired = {s: v for s, v in by_seed.items() if BASE in v and JUDGE in v}
    deaths = {c: sum(v[c]['deaths'] for v in paired.values()) for c in (BASE, JUDGE)}
    fewer = sum(1 for v in paired.values() if v[JUDGE]['deaths'] < v[BASE]['deaths'])
    more = sum(1 for v in paired.values() if v[JUDGE]['deaths'] > v[BASE]['deaths'])
    ratio = deaths[JUDGE] / deaths[BASE] if deaths[BASE] else float('nan')
    result = dict(seeds=sorted(paired), read_games={c: sum(1 for r in runs if r['config'] == c) for c in (BASE, JUDGE)},
                  unread_games=[dict(config=r['config'], seed=r['seed'], valid=r['valid'], authority=r.get('authority')) for r in report['runs'] if not (r['valid'] and r.get('authority', True))],
                  deaths=deaths, death_ratio=round(ratio, 3), seeds_fewer_deaths=fewer, seeds_more_deaths=more,
                  deaths_by_seed={s: {c: v[c]['deaths'] for c in (BASE, JUDGE)} for s, v in sorted(paired.items())},
                  completions={c: sum(v[c]['completed'] for v in paired.values()) for c in (BASE, JUDGE)},
                  rule=dict(death_ratio_max=0.8, more_seeds_fewer_than_more=True), passed=bool(deaths[BASE] and ratio <= 0.8 and fewer > more))
    first_key = {BASE: [], JUDGE: []}
    mech = Counter()
    hp = {'fall_back': [0.0, 0], 'other_attack': [0.0, 0]}
    for s, v in paired.items():
        for c in (BASE, JUDGE):
            run_dir = Path(v[c]['run'])
            if not run_dir.exists():
                continue
            events = [json.loads(l) for l in (run_dir / 'events.jsonl').read_text().splitlines()]
            keys = [e['game_seconds'] for e in events if e.get('event') == 'key_pickup']
            first_key[c].append(round(min(keys), 1) if keys else None)
            if c == JUDGE:
                m = mechanism(run_dir)
                for k in ('attack_decisions', 'judged', 'yes', 'applied', 'capped', 'fall_back_tics', 'other_attack_tics', 'fall_backs'):
                    mech[k] += m[k]
                if m['hp_per_tic_fall_back'] is not None:
                    hp['fall_back'][0] += m['hp_per_tic_fall_back'] * m['fall_back_tics']
                    hp['fall_back'][1] += m['fall_back_tics']
                if m['hp_per_tic_other_attack'] is not None:
                    hp['other_attack'][0] += m['hp_per_tic_other_attack'] * m['other_attack_tics']
                    hp['other_attack'][1] += m['other_attack_tics']
    result['first_key_seconds'] = first_key
    result['mechanism'] = dict(mech, share_yes=round(mech['yes'] / mech['judged'], 4) if mech['judged'] else None,
                               hp_per_tic_fall_back=round(hp['fall_back'][0] / hp['fall_back'][1], 4) if hp['fall_back'][1] else None,
                               hp_per_tic_other_attack=round(hp['other_attack'][0] / hp['other_attack'][1], 4) if hp['other_attack'][1] else None)
    print(json.dumps(result, indent=1))
    if a.save:
        a.save.write_text(json.dumps(result, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='stage', required=True)
    s = sub.add_parser('scenarios')
    s.add_argument('--out', type=Path, required=True, help='the doom/scenarios.py output')
    s.add_argument('--audit', action='store_true', help='run tools.check_authority on every judgement scenario run')
    s.add_argument('--save', type=Path)
    g = sub.add_parser('games')
    g.add_argument('--result', type=Path, required=True, help='a repeat_map.py result file with both configs')
    g.add_argument('--save', type=Path)
    a = p.parse_args()
    (scenarios if a.stage == 'scenarios' else games)(a)


if __name__ == '__main__':
    main()
