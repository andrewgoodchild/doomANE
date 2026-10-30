"""Repeat doomLaya maps across seeds and server configurations to measure completion rates.

Run from ext/doomLaya-dev with its venv:
  .venv/bin/python ../../doom/repeat_map.py --maps MAP03 MAP01 --seeds 54 1 2 3 --configs fast commit
Each configuration starts its own ane_doomlaya.py server and plays every map and seed on it;
results go to results/doom/repeat_<map>.json, one file per map.
With --campaign SECONDS one game starts on the first --maps map and plays on through the following maps
(each from a pistol start, as the engine starts a new episode per map) until the budget runs out;
results go to results/doom/repeat_campaign-<map>.json. Held-out maps, and campaigns (which reach them),
are played only from a frozen snapshot (doom/snapshot_stack.sh) with --evaluation.
"""
import argparse
import concurrent.futures
import json
import os
import queue
import socket
from pathlib import Path
import subprocess
import sys
import threading
import time
import urllib.request

ROOT = Path.cwd()
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.regression_suite import CASES, FLAGS

CONFIGS = {
    'slow': ['--encoder', 'ane'],
    'fast': ['--encoder', 'ane', '--heads', 'ane', '--skip-forced'],
    'memory': ['--encoder', 'ane', '--heads', 'ane', '--skip-forced',
               '--command-memory', str(REPO / 'results/doom_heads/command-memory-v1')],
    'commit': ['--encoder', 'ane', '--heads', 'ane', '--skip-forced', '--commit-route', '20'],
    'resume': ['--encoder', 'ane', '--heads', 'ane', '--skip-forced', '--commit-route', '20', '--commit-resume', '15'],
    'resume2': ['--encoder', 'ane', '--heads', 'ane', '--skip-forced', '--commit-route', '20', '--commit-resume', '15',
                '--commit-floor', '0.02', '--commit-pickups'],
    'resume3': ['--encoder', 'ane', '--heads', 'ane', '--skip-forced', '--commit-route', '20', '--commit-resume', '15',
                '--commit-floor', '0.02', '--commit-pickups', '--commit-margin', '0.3'],
    'planner': ['--encoder', 'ane', '--heads', 'ane', '--skip-forced'],
    'planner_ext': ['--encoder', 'ane', '--heads', 'ane', '--skip-forced'],
    'planner_run': ['--encoder', 'ane', '--heads', 'ane', '--skip-forced'],
    'planner_ext_run': ['--encoder', 'ane', '--heads', 'ane', '--skip-forced'],
    'planner_ext_safe': ['--encoder', 'ane', '--heads', 'ane', '--skip-forced'],
    'planner_ext_fast': ['--encoder', 'ane', '--heads', 'ane', '--skip-forced'],
    'random_ext_safe': ['--encoder', 'ane', '--heads', 'ane', '--skip-forced', '--random-chooser', '0'],  # baseline: uniform random answers
    'combat_move1': ['--encoder', 'ane', '--heads', 'ane', '--skip-forced'],
    'fight': ['--encoder', 'ane', '--heads', 'ane', '--skip-forced'],
    'planner6': ['--encoder', 'ane', '--heads', 'ane', '--skip-forced'],
    'legacyfire': ['--encoder', 'ane', '--heads', 'ane', '--skip-forced'],
    'legacyweapon': ['--encoder', 'ane', '--heads', 'ane', '--skip-forced'],
    'diaghold': ['--encoder', 'ane', '--heads', 'ane', '--skip-forced'],
    'v031': ['--encoder', 'ane', '--heads', 'ane', '--skip-forced'],  # run from a clean azalio v0.3.1 checkout (ext/doomLaya as setup.sh clones it)
    'system2': ['--encoder', 'ane', '--heads', 'ane', '--skip-forced'],
    'system2_27b': ['--encoder', 'ane', '--heads', 'ane', '--skip-forced'],
    'sample': ['--encoder', 'ane', '--heads', 'ane', '--skip-forced',
               '--choice-temperature', '1', '--sample-questions', 'command', 'switch'],
}
# configs whose game client is a wrapper around agent.py (same arguments)
CLIENTS = {'planner': 'doom/planner.py', 'planner_ext': 'doom/planner.py', 'planner_run': 'doom/planner.py', 'planner_ext_run': 'doom/planner.py', 'planner_ext_safe': 'doom/planner.py', 'random_ext_safe': 'doom/planner.py', 'planner_ext_fast': 'doom/planner.py', 'combat_move1': 'doom/planner.py', 'fight': 'doom/planner.py', 'planner6': 'doom/planner_v6.py', 'legacyfire': 'doom/planner.py', 'legacyweapon': 'doom/planner.py', 'diaghold': 'doom/planner.py',
           'system2': 'doom/system2.py', 'system2_27b': 'doom/system2.py'}  # planner6: v6 frozen for A/B
CLIENT_ENV = {'fight': {'PLANNER_FIGHT': '1'}, 'legacyfire': {'DOOM_FIRE_LEGACY': '1'},
              'legacyweapon': {'DOOM_WEAPON_TEXT_LEGACY': '1'},
              'diaghold': {'DOOM_DIAG_AMBUSH_HOLD': '1'},
              'system2': {'SYSTEM2_MODEL': 'mlx-community/Qwen3-8B-4bit'},
              'system2_27b': {'SYSTEM2_MODEL': 'mlx-community/Qwen3.8-27B-8bit'}}
# extra agent.py flags per config; planner_ext: every vanilla door, lift and floor special (doomlib/specials.py);
# *_run: run (SPEED) during the model's strafe, backward and retreat movements; *_fast: also along routes (always-run)
CLIENT_FLAGS = {'planner_ext': ['--extended-mechanisms'], 'planner_run': ['--run-combat-movement'],
                'planner_ext_run': ['--extended-mechanisms', '--run-combat-movement'],
                'planner_ext_safe': ['--extended-mechanisms', '--run-combat-movement', '--crusher-safety'],
                'random_ext_safe': ['--extended-mechanisms', '--run-combat-movement', '--crusher-safety'],
                'planner_ext_fast': ['--extended-mechanisms', '--run-combat-movement', '--crusher-safety', '--run-navigation'],
                'combat_move1': ['--extended-mechanisms', '--run-combat-movement', '--crusher-safety', '--combat-facts']}
# question heads that replace those in doom/doomlaya_args.txt for a config
# combat_move1: movement head fine-tuned on rollout outcomes (doom/fight_rollouts.py), reads --combat-facts
HEAD_OVERRIDES = {'combat_move1': {'movement': str(REPO / 'results/doom_heads/laya-combat-movement-v1')}}
# system2*: the objective is chosen by a local LLM served by MLX on SYSTEM2_PORT, started here (doom/system2.py)
SYSTEM2_PORT = 8011
MLX_SERVER = Path.home() / '.local/share/uv/tools/mlx-lm/bin/mlx_lm.server'
# legacyfire: pre-2026-10-21 fire gate; legacyweapon: 'Ammo: N' weapon text;
# diaghold: diagnostic only, code holds navigation after nearby areas open, not a model result
# maps beyond the regression suite (development and held-out, see doom/map_split.json)
SPLIT = json.loads((REPO / 'doom/map_split.json').read_text())
HOLDOUT = tuple((m, None, SPLIT['seconds']) for m in sorted(SPLIT['development'] + SPLIT['holdout'])
                if m not in {c[0] for c in CASES})
SUMMARY = ('status', 'game_seconds', 'kills', 'deaths', 'decisions', 'latency_ms_median', 'final_map', 'stalled',
           'levels_completed')


def config_server_args(config, server_args):
    heads = HEAD_OVERRIDES.get(config, {})
    return [f'{q}={heads[q]}' if q in heads and '=' in arg and arg.split('=')[0] == q else arg
            for arg in server_args for q in [arg.split('=')[0]]]


def gpu_flags(flags):
    """The same model in PyTorch on the GPU (MPS): Core ML holds the ANE models of only two processes at once."""
    out = list(flags)
    for flag, value in (('--encoder', 'torch'), ('--heads', 'cpu')):
        if flag in out:
            out[out.index(flag) + 1] = value
    return out


def start_servers(config, ports, server_args, log, ane_servers=1):
    """Model servers one after another; the first ane_servers on the ANE, the others on the GPU (same answers on
    40 of 40 replayed requests, 77 ms median against 85 ms). Returns the servers and port -> device."""
    servers, devices = [], {}
    with log.open('a') as handle:
        try:
            for i, port in enumerate(ports):
                ane = i < ane_servers
                devices[port] = 'ane' if ane else 'gpu'
                servers.append(subprocess.Popen([sys.executable, str(REPO / 'doom/ane_doomlaya.py'),
                                                 *(CONFIGS[config] if ane else gpu_flags(CONFIGS[config])),
                                                 '--port', str(port), '--device', 'cpu' if ane else 'mps', *server_args],
                                                cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT))
                wait_up(port, servers[-1])
        except BaseException:
            for server in servers:
                server.terminate()
            raise
    return servers, devices


def port_free(port):
    with socket.socket() as s:
        return s.connect_ex(('127.0.0.1', port)) != 0


def wait_up(port, server, timeout=900):
    deadline = time.monotonic() + timeout
    while True:
        if server.poll() is not None:
            raise RuntimeError('server exited')
        if time.monotonic() > deadline:
            raise RuntimeError(f'server not healthy after {timeout} s')
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=5):
                return
        except OSError:
            time.sleep(3)


def wait_mlx(server, model, timeout=600):
    """Wait until the MLX server answers a chat request, which also loads and warms the model."""
    body = json.dumps({'model': model, 'messages': [{'role': 'user', 'content': 'Say OK.'}], 'max_tokens': 4,
                       'chat_template_kwargs': {'enable_thinking': False}}).encode()
    deadline = time.monotonic() + timeout
    while True:
        if server.poll() is not None:
            raise RuntimeError('MLX server exited')
        if time.monotonic() > deadline:
            raise RuntimeError(f'MLX server not answering after {timeout} s')
        try:
            request = urllib.request.Request(f'http://127.0.0.1:{SYSTEM2_PORT}/v1/chat/completions', body,
                                             {'content-type': 'application/json'})
            with urllib.request.urlopen(request, timeout=120):
                return
        except OSError:
            time.sleep(3)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--maps', nargs='+', default=None, choices=[c[0] for c in CASES + HOLDOUT], help='default MAP03 unless --wads')
    p.add_argument('--seeds', type=int, nargs='+', default=[54, 1, 2, 3])
    p.add_argument('--configs', nargs='+', default=list(CONFIGS), choices=list(CONFIGS))
    p.add_argument('--port', type=int, default=8003, help='first model server port; --parallel N uses N consecutive ports')
    p.add_argument('--parallel', type=int, default=1, help='games played at once, each against its own model server')
    p.add_argument('--ane-servers', type=int, default=1, help='of the parallel servers, how many run on the ANE (the others on the GPU); Core ML takes two ANE processes at most')
    p.add_argument('--suffix', default='', help='separate results file, e.g. after changing doomLaya itself')
    p.add_argument('--keep-videos', action='store_true', help='keep the video of runs that did not complete')
    p.add_argument('--wads', nargs='+', default=[], type=Path, help='PWADs with a MAP01 each (e.g. Obsidian maps), played as extra maps')
    p.add_argument('--wad-seconds', type=float, default=900)
    p.add_argument('--extra-flags', nargs='+', default=[], help="agent flags added to every run, quoted after '=', e.g. --extra-flags='--fight-states --combat-facts'")
    p.add_argument('--evaluation', action='store_true', help='allow held-out maps and campaigns (only from a frozen snapshot, doom/snapshot_stack.sh)')
    p.add_argument('--campaign', type=float, metavar='SECONDS', help='one game from the first --maps map on through the next maps, for this many game seconds')
    a = p.parse_args()
    frozen = a.evaluation and '/ext/snap/' in str(ROOT)
    if a.campaign and not frozen:
        raise SystemExit('a campaign reaches held-out maps: run it from ext/snap/<name> with --evaluation')
    if not frozen and set(a.maps or ()) & set(SPLIT['holdout']):
        raise SystemExit('held-out maps are played only by a frozen stack: run from ext/snap/<name> with --evaluation')
    a.extra_flags = [flag for value in a.extra_flags for flag in value.split()]  # "--x --y" given as one value
    budget = {m: t for m, _, t in CASES + HOLDOUT}
    wads = {}  # map key -> PWAD; Obsidian sets live in $OBLIGE_MAPS/<set>/seed<N>.wad (doom/oblige_maps.py)
    for wad in a.wads:
        key = f'OBL-{wad.parent.name}-{wad.stem}'.upper()
        if 'HOLDOUT' in key and not (a.evaluation and '/ext/snap/' in str(ROOT)):
            raise SystemExit(f'{wad}: held-out Obsidian sets are played only by a frozen stack: run from ext/snap/<name> with --evaluation')
        wads[key] = wad.resolve()
        budget[key] = a.wad_seconds
    a.maps = (a.maps or ([] if wads else ['MAP03'])) + list(wads)
    engine_map = {m: m for m in a.maps}
    if a.campaign:
        if wads or len(a.maps) != 1:
            raise SystemExit('--campaign takes one start map and no --wads')
        start = a.maps[0]
        a.maps = [f'CAMPAIGN-{start}']
        engine_map = {a.maps[0]: start}
        budget[a.maps[0]] = a.campaign
    server_args = (REPO / 'doom/doomlaya_args.txt').read_text().split()
    reports = {}
    for game_map in a.maps:
        out = REPO / 'results/doom' / f'repeat_{game_map.lower()}{"_" + a.suffix if a.suffix else ""}.json'
        flags = [f for f in FLAGS if not (a.campaign and f == '--stop-after-level')] + a.extra_flags
        setup = dict(map=game_map, seconds=budget[game_map], flags=flags, server_args=server_args)
        if game_map in wads:
            setup['wad'] = str(wads[game_map])
        report = json.loads(out.read_text()) if out.exists() else dict(setup, configs={}, runs=[])
        changed = [k for k, v in setup.items() if report.get(k, v) != v]
        # a config's server flags may not change once it has runs; new configs can be added
        changed += [c for c, flags in report.get('configs', {}).items() if c in CONFIGS and flags != CONFIGS[c]
                    and any(r['config'] == c for r in report['runs'])]
        if changed:
            raise SystemExit(f'{out} was recorded with different {", ".join(changed)}; move it aside to start over')
        report.update(setup)
        report['configs'] = {**report.get('configs', {}), **{c: CONFIGS[c] for c in a.configs}}
        reports[game_map] = (out, report)
    logs = REPO / 'results/logs'
    logs.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()  # report files and RESULT lines, shared by parallel games

    def play(config, game_map, seed, port, device='ane'):
        out, report = reports[game_map]
        tag = f'repeat-{config}{"-" + a.suffix if a.suffix else ""}-{game_map.lower()}-seed{seed}'
        client = REPO / CLIENTS[config] if config in CLIENTS else ROOT / 'agent.py'
        env = dict(os.environ, PLANNER_LOG=str(ROOT / 'runs' / f'{tag}-planner.jsonl'), **CLIENT_ENV.get(config, {}))
        agent = subprocess.run([sys.executable, str(client), '--model', 'doom-adapted',
                                '--endpoint', f'http://127.0.0.1:{port}/predict', *report['flags'],
                                '--map', 'MAP01' if game_map in wads else engine_map[game_map], '--seed', str(seed), '--seconds', str(budget[game_map]),
                                *(['--wad', str(wads[game_map])] if game_map in wads else []),
                                '--tag', tag, *CLIENT_FLAGS.get(config, [])],
                               cwd=ROOT, capture_output=True, text=True, env=env)
        run = next((line[4:] for line in agent.stdout.splitlines() if line.startswith('RUN ')), None)
        if agent.returncode or run is None:
            raise RuntimeError(f'{tag} failed:\n{agent.stdout[-2000:]}{agent.stderr[-2000:]}')
        run = ROOT / run
        with (run / 'verify.log').open('w') as vlog:
            verified = subprocess.run([sys.executable, '-m', 'tools.verify_run', str(run)], cwd=ROOT,
                                      stdout=vlog, stderr=subprocess.STDOUT)
        if not (run / 'verification.json').exists():
            raise RuntimeError(f'{tag}: verifier exited {verified.returncode} without a report; see {run}/verify.log')
        verification = json.loads((run / 'verification.json').read_text())
        summary = json.loads((run / 'summary.json').read_text())
        result = dict(config=config, seed=seed, run=str(run.relative_to(ROOT)), server=device,
                      completed=verification['level_completed'], valid=verification['experiment_valid'],
                      authority=verification['authority']['pass'], verifier_exit=verified.returncode,
                      **{k: summary.get(k) for k in SUMMARY})
        with lock:
            report['runs'].append(result)
            out.write_text(json.dumps(report, indent=2) + '\n')
            print('RESULT', game_map, json.dumps(result), flush=True)
        if not result['completed'] and not a.keep_videos:
            (run / 'video.mp4').unlink(missing_ok=True)  # verified; failed-run videos are 60 % of the disk

    for config in a.configs:
        todo = [(m, s) for m in a.maps for s in dict.fromkeys(a.seeds)
                if (config, s) not in {(r['config'], r['seed']) for r in reports[m][1]['runs']}]
        if not todo:
            continue
        if config.startswith('system2') and a.parallel > 1:
            raise SystemExit('system2 configs share one MLX server: run them with --parallel 1')
        # one model server per parallel game; each game keeps its own server for its whole run
        ports = [a.port + i for i in range(min(a.parallel, len(todo)))]
        busy = [port for port in ports if not port_free(port)]
        if busy:
            raise RuntimeError(f'ports {busy} are already in use; stop those servers first')
        servers, devices = start_servers(config, ports, config_server_args(config, server_args), logs / f'doomlaya_repeat_{config}.log', a.ane_servers)
        mlx = None
        if config.startswith('system2'):
            if not port_free(SYSTEM2_PORT):
                raise RuntimeError(f'port {SYSTEM2_PORT} is already in use; stop that server first')
            mlx = subprocess.Popen([str(MLX_SERVER), '--model', CLIENT_ENV[config]['SYSTEM2_MODEL'], '--port', str(SYSTEM2_PORT)],
                                   stdout=(logs / f'system2_server_{config}.log').open('a'), stderr=subprocess.STDOUT)
        try:
            if mlx is not None:
                wait_mlx(mlx, CLIENT_ENV[config]['SYSTEM2_MODEL'])
            free = queue.Queue()
            for port in ports:
                free.put(port)

            def job(game_map, seed):
                port = free.get()
                try:
                    play(config, game_map, seed, port, devices[port])
                finally:
                    free.put(port)
            with concurrent.futures.ThreadPoolExecutor(len(ports)) as pool:
                for future in [pool.submit(job, m, s) for m, s in todo]:
                    future.result()
        finally:
            for server in servers:
                server.terminate()
                server.wait()
            if mlx is not None:
                mlx.terminate()
                mlx.wait()
    for game_map, (out, report) in reports.items():
        rates = {}
        for r in report['runs']:
            # a completion only counts when the experiment was valid and the model kept authority
            rates.setdefault(r['config'], []).append(r['completed'] and r['valid'] and r.get('authority', True))
        for config, values in rates.items():
            print('RATE', game_map, config, f'{sum(values)}/{len(values)}', flush=True)


if __name__ == '__main__':
    main()
