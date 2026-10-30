"""Generate single-level Doom 2 WADs (MAP01) with the Obsidian level maker, one per seed, reproducibly.

Obsidian (the successor to Oblige) is built from a clone of Obsidian ($OBSIDIAN_SOURCE, default ext/obsidian-src) with the local patch
doom/patches/obsidian-macos-arm64.patch and run from its internal-disk copy in ext/obsidian (macOS may
not let it getcwd on an external volume). The patch fixes an arm64 crash in LuaJIT and removes the
dependence on heap addresses, so the output depends only on seed and settings. Maps target Freedoom 2 / Doom 2 in the
limit-removing binary format: no UDMF, no ACS, no Boom generalized linedefs.

  python3 doom/oblige_maps.py dev1 --seeds 1-10                 # writes $OBLIGE_MAPS/dev1/seed<N>.wad (default ext/oblige-maps)
  python3 doom/oblige_maps.py dev1 --seeds 11 12 --set float_mons=1.5 health=more
  python3 doom/oblige_maps.py dev1 --check-repro                # regenerate the first seed twice, compare hashes
  ext/doomLaya-dev/.venv/bin/python doom/oblige_maps.py dev1 --verify   # load each WAD in ViZDoom
Each set has manifest.json next to its WADs with the exact settings, command line, Obsidian commit, binary
and patch hashes, and per-map hash and lump statistics. A set keeps one group of settings: adding seeds
with different settings is refused unless --force.
"""
import argparse
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OBSIDIAN = REPO / 'ext/obsidian'
BINARY = OBSIDIAN / 'obsidian'
SOURCE = Path(os.environ.get('OBSIDIAN_SOURCE', REPO / 'ext/obsidian-src'))
SOURCE_COMMIT = 'f4c93451e10fd3ab13395dfb8e7b0416c1cd6633'  # recorded in case git on the source volume fails
PATCH = REPO / 'doom/patches/obsidian-macos-arm64.patch'
OUT_ROOT = Path(os.environ.get('OBLIGE_MAPS', REPO / 'ext/oblige-maps'))
VIZDOOM_PYTHON = REPO / 'ext/doomLaya-dev/.venv/bin/python'

# Game settings (top-level config keys). Obsidian's own defaults apply to everything not listed; the
# full effective configuration is in each seed<N>.log ("CONFIG FILE" block) and in the WAD's OBSIDATA lump.
SETTINGS = {
    'game': 'doom2',
    'engine': 'idtech_1',
    'port': 'limit_removing',  # binary map format, vanilla specials only; 'limit_enforcing' switches to SLUMP
    'length': 'single',        # one level, MAP01
    'theme': 'original',
    'float_size': '36',        # map width/height in grid squares, 10-75
    'float_mons': '1',         # monster quantity multiplier, 0-10
    'float_strength': '1',     # monster strength, 0.55-12
    'bosses': 'medium',
    'traps': 'mixed',
    'health': 'normal',        # none scarce less bit_less normal bit_more more heaps
    'ammo': 'normal',          # same choices as health
    'weapons': 'normal',       # none very_soon sooner normal later very_late
    'items': 'normal',         # none rare less normal more heaps
    'secrets': 'mixed',
    'bool_pistol_starts': '1',
}
# Module options, passed as @module=on/off followed by its options. limit_removing raises an error
# ("uses a very basic set of prefabs") unless the debugger module suppresses it.
MODULES = {
    'debugger': ('1', {'bool_suppress_nolimit_warning': '1'}),
    'music_swapper': ('0', {}),  # would add a DEHACKED music shuffle; not needed
}
# Program options file: written per run so that no state is shared between runs and nothing is
# read from or written to ~/Library/Application Support/Obsidian (that path is also the Obsidian notes app's).
OPTIONS = {
    'language': 'AUTO', 'create_backups': '0', 'overwrite_warning': '0', 'debug_messages': '0',
    'limit_break': '0', 'preserve_old_config': '0', 'randomize_architecture': '0', 'randomize_monsters': '0',
    'randomize_pickups': '0', 'randomize_misc': '0', 'random_string_seeds': '0', 'password_mode': '0',
    'mature_word_lists': '0', 'filename_prefix': '0', 'builds_per_run': '1',
}

KEYS = {5: 'blue_card', 6: 'yellow_card', 13: 'red_card', 38: 'red_skull', 39: 'yellow_skull', 40: 'blue_skull'}
EXITS = {11: 'exit_switch', 51: 'secret_exit_switch', 52: 'exit_walk', 124: 'secret_exit_walk'}
MAX_VANILLA_SPECIAL = 141


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_seeds(items):
    seeds = []
    for item in items:
        if '-' in item:
            lo, hi = item.split('-')
            seeds += range(int(lo), int(hi) + 1)
        else:
            seeds.append(int(item))
    return seeds


def obsidian_commit():
    try:
        out = subprocess.run(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], capture_output=True, text=True,
                             cwd=REPO, timeout=30)
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return SOURCE_COMMIT


def command(output, seed, settings, modules, config, options, log):
    """The exact Obsidian command line. Plain settings come first: after @module, arguments belong to it."""
    cmd = [str(BINARY), '--config', str(config), '--options', str(options), '--log', str(log),
           '--batch', str(output), f'seed={seed}']
    cmd += [f'{k}={v}' for k, v in settings.items()]
    for name, (on, opts) in modules.items():
        cmd.append(f'@{name}={on}')
        cmd += [f'{k}={v}' for k, v in opts.items()]
        cmd.append('@@')
    return cmd


def generate(seed, settings, modules, dest, log_dest=None):
    """Build one WAD into dest. Obsidian writes tmp.wad next to the output and renames it, which fails
    across volumes, so it builds in a temporary directory on the internal disk and the result is copied."""
    with tempfile.TemporaryDirectory(prefix='oblige_') as tmp:
        tmp = Path(tmp)
        (tmp / 'OPTIONS.txt').write_text(''.join(f'{k} = {v}\n' for k, v in OPTIONS.items()))
        out, log = tmp / 'out.wad', tmp / 'obsidian.log'
        cmd = command(out, seed, settings, modules, tmp / 'CONFIG.txt', tmp / 'OPTIONS.txt', log)
        # cwd must be on the internal disk (getcwd can fail on an external volume)
        res = subprocess.run(cmd, cwd=OBSIDIAN, capture_output=True, text=True, timeout=1800)
        if res.returncode != 0 or not out.exists():
            tail = (res.stdout + res.stderr)[-2000:]
            raise RuntimeError(f'obsidian failed for seed {seed} (exit {res.returncode}):\n{tail}')
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(out, dest)
        if log_dest:
            shutil.copyfile(log, log_dest)
    return cmd


def wad_lumps(path):
    data = Path(path).read_bytes()
    ident, count, offset = struct.unpack_from('<4sii', data, 0)
    lumps = []
    for i in range(count):
        pos, size, name = struct.unpack_from('<ii8s', data, offset + 16 * i)
        lumps.append((name.rstrip(b'\0').decode('ascii', 'replace'), data[pos:pos + size]))
    return ident.decode(), lumps


def wad_stats(path):
    """Counts from the MAP01 lumps: things, linedef specials, keys, exits, non-vanilla specials."""
    ident, lumps = wad_lumps(path)
    names = [n for n, _ in lumps]
    maps = [n for n in names if len(n) == 5 and n.startswith('MAP') and n[3:].isdigit()]
    i = names.index('MAP01')
    level = {}
    for name, data in lumps[i + 1:]:
        if name not in ('THINGS', 'LINEDEFS', 'SIDEDEFS', 'VERTEXES', 'SEGS', 'SSECTORS', 'NODES', 'SECTORS',
                        'REJECT', 'BLOCKMAP', 'TEXTMAP', 'BEHAVIOR', 'ZNODES', 'ENDMAP'):
            break
        level[name] = data
    things = [struct.unpack_from('<hhhhh', level['THINGS'], o) for o in range(0, len(level['THINGS']), 10)]
    lines = [struct.unpack_from('<HHHHHHH', level['LINEDEFS'], o) for o in range(0, len(level['LINEDEFS']), 14)]
    specials = [ln[3] for ln in lines if ln[3]]
    keys = {}
    for t in things:
        if t[3] in KEYS:
            keys[KEYS[t[3]]] = keys.get(KEYS[t[3]], 0) + 1
    exits = {}
    for s in specials:
        if s in EXITS:
            exits[EXITS[s]] = exits.get(EXITS[s], 0) + 1
    return {
        'wad_type': ident,
        'maps': maps,
        'format': 'udmf' if 'TEXTMAP' in level else 'binary',
        'acs': 'BEHAVIOR' in level,
        'things': len(things),
        'player_starts': sum(1 for t in things if t[3] == 1),
        'linedefs': len(lines),
        'sectors': len(level['SECTORS']) // 26,
        'linedef_specials': len(specials),
        'special_types': sorted(set(specials)),
        'keys': keys,
        'exits': exits,
        'non_vanilla_specials': sorted({s for s in specials if s > MAX_VANILLA_SPECIAL}),
        'lumps': len(lumps),
    }


def verify(set_dir, tics):
    """Load each WAD in ViZDoom with freedoom2.wad, run some tics and record the sector count.
    Needs vizdoom, so run it with the doomLaya venv python."""
    import vizdoom as vzd
    iwad = Path(vzd.__file__).parent / 'freedoom2.wad'
    manifest_path = set_dir / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory(prefix='oblige_vzd_') as tmp:
        os.chdir(tmp)  # ViZDoom writes _vizdoom.ini into the working directory
        try:
            for name, entry in sorted(manifest['maps'].items(), key=lambda kv: kv[1]['seed']):
                game = vzd.DoomGame()
                game.set_doom_game_path(str(iwad))
                game.set_doom_scenario_path(str(set_dir / entry['wad']))
                game.set_doom_map('MAP01')
                game.set_window_visible(False)
                game.set_mode(vzd.Mode.PLAYER)
                game.set_sectors_info_enabled(True)
                game.set_available_buttons([vzd.Button.MOVE_FORWARD])
                result = {'iwad': iwad.name, 'iwad_sha256': sha256(iwad), 'tics': tics}
                try:
                    game.init()
                    game.new_episode()
                    for _ in range(tics):
                        if game.is_episode_finished():
                            break
                        game.make_action([0])
                    state = game.get_state()
                    result.update(ok=True, sectors=len(state.sectors) if state else None,
                                  episode_tic=game.get_episode_time(), dead=game.is_player_dead())
                except Exception as e:  # report and continue with the next map
                    result.update(ok=False, error=str(e))
                finally:
                    game.close()
                entry['vizdoom'] = result
                print(name, 'ok' if result['ok'] else 'FAILED', 'sectors', result.get('sectors'),
                      'tic', result.get('episode_tic'), result.get('error', ''))
        finally:
            os.chdir(cwd)
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('set_name')
    p.add_argument('--seeds', nargs='+', default=[], help='seeds or ranges, e.g. 1-10 42')
    p.add_argument('--set', nargs='+', default=[], metavar='KEY=VALUE', help='override a game setting')
    p.add_argument('--out-root', type=Path, default=OUT_ROOT)
    p.add_argument('--overwrite', action='store_true', help='regenerate seeds that already exist')
    p.add_argument('--force', action='store_true', help='allow settings that differ from the set manifest')
    p.add_argument('--check-repro', action='store_true', help='regenerate the first seed twice and compare')
    p.add_argument('--verify', action='store_true', help='load every WAD in ViZDoom (needs the vizdoom venv)')
    p.add_argument('--tics', type=int, default=100)
    a = p.parse_args()

    set_dir = a.out_root / a.set_name
    if a.verify:
        verify(set_dir, a.tics)
        return

    settings = dict(SETTINGS)
    for kv in a.set:
        k, v = kv.split('=', 1)
        settings[k] = v
    modules = {k: (on, dict(opts)) for k, (on, opts) in MODULES.items()}
    manifest_path = set_dir / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        # compare as JSON: tuples come back from the manifest as lists
        if (manifest['settings'] != settings or manifest['modules'] != json.loads(json.dumps(modules))) and not a.force:
            sys.exit(f'{manifest_path}: settings differ from this set; use another set name or --force')
    else:
        manifest = {'maps': {}}
    manifest.update({
        'set': a.set_name,
        'generator': 'Obsidian level maker (successor to Oblige), batch mode',
        'obsidian_commit': obsidian_commit(),
        'obsidian_source': str(SOURCE),
        'local_patch': str(PATCH.relative_to(REPO)),
        'local_patch_sha256': sha256(PATCH),
        'binary': str(BINARY.relative_to(REPO)),
        'binary_sha256': sha256(BINARY),
        'settings': settings,
        'modules': modules,
        'options_file': OPTIONS,
        'iwad': 'freedoom2.wad (or doom2.wad)',
        'command_template': command('<out.wad>', '<seed>', settings, modules, '<tmp>/CONFIG.txt',
                                    '<tmp>/OPTIONS.txt', '<tmp>/obsidian.log'),
        'notes': 'Same seed + settings + binary give a byte-identical WAD. Obsidian defaults apply to every '
                 'key not in settings; the full config is in seed<N>.log and in the OBSIDATA lump.',
    })

    seeds = parse_seeds(a.seeds)
    for seed in seeds:
        name = f'seed{seed}'
        wad = set_dir / f'{name}.wad'
        if wad.exists() and not a.overwrite and name in manifest['maps']:
            print(name, 'exists, skipped')
            continue
        t0 = time.time()
        cmd = generate(seed, settings, modules, wad, set_dir / f'{name}.log')
        stats = wad_stats(wad)
        manifest['maps'][name] = {'seed': seed, 'wad': wad.name, 'log': f'{name}.log', 'sha256': sha256(wad),
                                  'bytes': wad.stat().st_size, 'seconds': round(time.time() - t0, 1),
                                  'generated': time.strftime('%Y-%m-%dT%H:%M:%S'), 'command': cmd, 'stats': stats}
        print(name, f"{stats['sectors']} sectors, {stats['linedef_specials']} specials, keys {stats['keys']}, "
              f"exits {stats['exits']}, non-vanilla {stats['non_vanilla_specials']}, {time.time() - t0:.1f} s")
        set_dir.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')

    if a.check_repro:
        seed = seeds[0] if seeds else min(m['seed'] for m in manifest['maps'].values())
        hashes = []
        with tempfile.TemporaryDirectory(prefix='oblige_repro_') as tmp:
            for i in range(2):
                out = Path(tmp) / f'repro{i}.wad'
                generate(seed, settings, modules, out)
                hashes.append(sha256(out))
        stored = manifest['maps'].get(f'seed{seed}', {}).get('sha256')
        same = len(set(hashes + ([stored] if stored else []))) == 1
        manifest['repro_check'] = {'seed': seed, 'hashes': hashes, 'manifest_sha256': stored, 'identical': same,
                                   'checked': time.strftime('%Y-%m-%dT%H:%M:%S')}
        print('repro seed', seed, 'identical' if same else 'DIFFERENT', hashes, stored)
        set_dir.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
        if not same:
            sys.exit(1)


if __name__ == '__main__':
    main()
