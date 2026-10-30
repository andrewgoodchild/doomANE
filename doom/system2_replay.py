"""Replay logged objective questions from development-map planner runs to the System 2 LLM, offline.

Checks latency and answer validity before a game run, and shows where System 2 and Laya disagree.
Held-out maps (doom/map_split.json) are refused. The game state text is not in the planner log, so
prompts here carry only the options, the re-plan reason and the objective history of that run.

  python doom/system2_replay.py ext/doomLaya/runs/repeat-planner-*-map03-seed54-planner.jsonl --limit 20
"""
import argparse
import json
import re
import time
from pathlib import Path

import system2

REPO = Path(__file__).resolve().parents[1]
HELD_OUT = set(json.loads((REPO / 'doom/map_split.json').read_text())['holdout'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('logs', nargs='+')
    p.add_argument('--limit', type=int, default=20, help='objective questions per log')
    p.add_argument('--out', default=None)
    a = p.parse_args()
    rows, out = [], []
    for log in a.logs:
        found = re.search(r'map\d\d', log)
        if found is None or found.group().upper() in HELD_OUT:
            raise SystemExit(f'{log}: not a development-map log')
        history = []
        for line in Path(log).read_text().splitlines():
            row = json.loads(line)
            if row.get('question', 'objective') != 'objective':
                continue
            if history:
                history[-1]['outcome'] = row['reason']
            rows.append((log, row, list(history)))
            history.append(dict(choice=row['choice'], at=row['game_seconds']))
            if sum(r[0] == log for r in rows) >= a.limit:
                break
    agree = valid = 0
    latencies = []
    for log, row, history in rows:
        criteria = row['options']
        user = (f"{row['game_seconds']:.0f} s into the level. Re-planning because: {row['reason']}.\n"
                f"Objectives so far: {system2.describe(history)}.\n\nOptions:\n"
                + "\n".join(f"- {k}: {v}" for k, v in criteria.items()))
        messages = [{"role": "system", "content": system2.SYSTEM}, {"role": "user", "content": user}]
        started = time.perf_counter()
        text = system2.System2._chat(None, messages)
        ms = (time.perf_counter() - started) * 1000
        choice, why, error = system2.System2._parse(text, criteria)
        latencies.append(ms)
        valid += error is None
        agree += choice == row['choice']
        out.append(dict(log=Path(log).name, tick=row['tick'], reason=row['reason'], laya=row['choice'], system2=choice,
                        why=why, error=error, latency_ms=round(ms), options=len(criteria)))
        print(f"{Path(log).name[:48]:48} {row['game_seconds']:6.0f}s  laya={row['choice']:12} s2={str(choice):12} {ms:5.0f}ms  {why or error}")
    latencies.sort()
    print(f"\n{len(rows)} questions: valid {valid}, agree with Laya {agree}, "
          f"latency median {latencies[len(latencies) // 2]:.0f} ms, max {latencies[-1]:.0f} ms")
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
