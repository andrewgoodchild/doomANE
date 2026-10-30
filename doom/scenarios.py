"""Development scenarios: play short games from saved engine states and compare configurations.

Each scenario loads a saved state (agent.py --load-state, e.g. a fight state from a --fight-states run),
plays it for --seconds with the full stack of a repeat_map configuration and ends at the first death. A batch
of states takes minutes instead of the hours of whole games; the result is survival and health lost per
configuration on the same states. These are screening numbers, not completions; held-out maps are refused.

With --commands each state is also played with the first command forced to each listed one ('model' is the
model's own choice; the model still answers target, weapon and movement). These runs fail the authority audit
by design and are training labels for the command head only, never results.

Run from a doomLaya checkout (ext/doomLaya-dev or ext/snap/<name>/ext/doomLaya):
  .venv/bin/python ../../doom/scenarios.py --runs runs/<run>... --before-death 10 --limit 20 \\
      --configs planner_ext_safe combat_move1 --parallel 4 --port 8030 --out ../../results/doom/scenarios-x.jsonl
"""
import argparse
import concurrent.futures
import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import repeat_map as rm  # noqa: E402  (sets ROOT to the doomLaya checkout)


def states(runs, before_death, limit):
    """(run, saved state, map) for movement-question states at most before_death seconds before a death."""
    for run in map(Path, runs):
        cfg = json.loads((run / 'config.json').read_text())
        name = cfg['args']['map'].upper()
        if name in rm.SPLIT['holdout'] and not cfg['args'].get('wad'):
            raise SystemExit(f'{run}: {name} is a held-out map')
        deaths = [e['game_seconds'] for e in map(json.loads, (run / 'events.jsonl').read_text().splitlines()) if e.get('event') == 'death']
        rows = [json.loads(line) for line in (run / 'decisions.jsonl').read_text().splitlines()]
        rows = [r for r in rows if r.get('fight_state') and (not before_death or any(0 <= d - r['game_seconds'] <= before_death for d in deaths))]
        rows = rows[::max(1, len(rows) // limit)][:limit] if limit else rows
        for r in rows:
            yield run, run / r['fight_state'], name


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--runs', nargs='+', required=True, help='collection runs with saved fight states')
    p.add_argument('--before-death', type=float, default=10, help='only states this many seconds before a death (0: all)')
    p.add_argument('--limit', type=int, default=20, help='states per run (0: all)')
    p.add_argument('--configs', nargs='+', required=True, choices=list(rm.CONFIGS))
    p.add_argument('--seconds', type=float, default=30)
    p.add_argument('--parallel', type=int, default=4)
    p.add_argument('--port', type=int, default=8030)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--commands', nargs='+', default=['model'], help="first commands to force per state ('model': none)")
    p.add_argument('--ane-servers', type=int, default=0, help='servers on the ANE; the others run the same model on the GPU')
    a = p.parse_args()
    todo = list(states(a.runs, a.before_death, a.limit))
    done = set()
    if a.out.exists():
        done = {(r['config'], r['state'], r.get('command', 'model')) for r in map(json.loads, a.out.read_text().splitlines())}
    server_args = (rm.REPO / 'doom/doomlaya_args.txt').read_text().split()
    lock = threading.Lock()
    logs = rm.REPO / 'results/logs'
    logs.mkdir(parents=True, exist_ok=True)
    for config in a.configs:
        jobs = [(run, state, name, command) for run, state, name in todo for command in a.commands
                if (config, str(state), command) not in done]
        if not jobs:
            continue
        ports = [a.port + i for i in range(min(a.parallel, len(jobs)))]
        busy = [port for port in ports if not rm.port_free(port)]
        if busy:
            raise SystemExit(f'ports {busy} are already in use')
        servers, _ = rm.start_servers(config, ports, rm.config_server_args(config, server_args), logs / f'scenarios_{config}.log', a.ane_servers)
        try:
            free = queue.Queue()
            for port in ports:
                free.put(port)

            def job(run, state, name, command):
                port = free.get()
                try:
                    tag = f'scenario-{config}-{name.lower()}-{state.stem}' + ('' if command == 'model' else f'-{command}')
                    force = [] if command == 'model' else ['--force-first-command', command]
                    client = rm.REPO / rm.CLIENTS[config] if config in rm.CLIENTS else rm.ROOT / 'agent.py'
                    env = dict(os.environ, PLANNER_LOG=os.devnull, **rm.CLIENT_ENV.get(config, {}))
                    agent = subprocess.run([sys.executable, str(client), '--model', 'doom-adapted', '--endpoint', f'http://127.0.0.1:{port}/predict',
                                            *rm.FLAGS, '--map', name, '--seconds', str(a.seconds), '--load-state', str(state), '--tag', tag,
                                            *rm.CLIENT_FLAGS.get(config, []), *force], cwd=rm.ROOT, capture_output=True, text=True, env=env)
                    out = next((line[4:] for line in agent.stdout.splitlines() if line.startswith('RUN ')), None)
                    if agent.returncode or out is None:
                        # e.g. a state saved on the death tic: no live player to play from
                        tail = (agent.stderr or agent.stdout).strip().splitlines()[-1:] or ['']
                        print('SKIP', config, command, state, tail[0][:200], flush=True)
                        return
                    out = rm.ROOT / out
                    summary = json.loads((out / 'summary.json').read_text())
                    telemetry = (out / 'telemetry.jsonl').read_text().splitlines()
                    first, last = json.loads(telemetry[0]), json.loads(telemetry[-1])
                    died = summary['deaths'] > 0
                    result = dict(config=config, command=command, state=str(state), run=run.name, map=name, scenario=out.name, died=died,
                                  survived_seconds=summary['game_seconds'], start_hp=first['hp'],
                                  lost=first['hp'] - (0 if died else last['hp']) + max(0, first['armor'] - last['armor']),
                                  dealt=last['damagecount'] - first['damagecount'], kills=summary['kills'])
                    (out / 'video.mp4').unlink(missing_ok=True)
                    with lock:
                        with a.out.open('a') as handle:
                            handle.write(json.dumps(result) + '\n')
                        print('SCENARIO', json.dumps(result), flush=True)
                finally:
                    free.put(port)
            with concurrent.futures.ThreadPoolExecutor(len(ports)) as pool:
                for future in [pool.submit(job, *j) for j in jobs]:
                    future.result()
        finally:
            for server in servers:
                server.terminate()
                server.wait()
    rows = [json.loads(line) for line in a.out.read_text().splitlines()]
    arms = [(c, m) for c in a.configs for m in a.commands]
    shared = set.intersection(*[{r['state'] for r in rows if (r['config'], r.get('command', 'model')) == arm} for arm in arms])
    for config, command in arms:
        mine = [r for r in rows if (r['config'], r.get('command', 'model')) == (config, command) and r['state'] in shared]
        if mine:
            print('SUMMARY', config, command, f'states={len(mine)} died={sum(r["died"] for r in mine)} '
                  f'lost={sum(r["lost"] for r in mine) / len(mine):.1f} dealt={sum(r["dealt"] for r in mine) / len(mine):.1f}', flush=True)


if __name__ == '__main__':
    main()
