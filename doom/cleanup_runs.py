"""Free disk used by doomLaya runs: delete videos once a run is finished and verified.

Videos are ~60 % of every run (1 GB for a 30-minute MAP02 game). The verifier needs a run's video,
so a video is only deleted after the run has its verification.json (or, for runs that are never
verified such as teacher runs, its summary.json and an hour without writes). Telemetry, decisions,
events, summaries and verification reports are always kept, so every analysis still works.

Kept videos: the newest completed run per map, plus any run named in doom/keep_videos.txt.

  .venv/bin/python ../../doom/cleanup_runs.py            # dry run: what would go, and how much
  .venv/bin/python ../../doom/cleanup_runs.py --apply    # delete
Options: --keep-completed keeps every completed run's video; --gzip-telemetry also compresses
telemetry.jsonl of runs older than --gzip-days (default 3), `gunzip` restores it.
"""
import argparse
import gzip
import json
import re
import shutil
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / 'ext/doomLaya/runs'
KEEP = REPO / 'doom/keep_videos.txt'
QUIET_SECONDS = 3600


def finished(run):
    if (run / 'verification.json').exists():
        return True
    newest = max((p.stat().st_mtime for p in run.rglob('*') if p.is_file()), default=0)
    return (run / 'summary.json').exists() and time.time() - newest > QUIET_SECONDS


def run_map(run):
    m = re.search(r'map0\d', run.name)
    return m.group(0) if m else None


def completed(run):
    v = run / 'verification.json'
    return v.exists() and json.loads(v.read_text()).get('level_completed')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--apply', action='store_true')
    p.add_argument('--keep-completed', action='store_true')
    p.add_argument('--gzip-telemetry', action='store_true')
    p.add_argument('--gzip-days', type=float, default=3)
    a = p.parse_args()
    runs = sorted(r for r in RUNS.iterdir() if r.is_dir())
    keep = {line.strip() for line in KEEP.read_text().splitlines()} if KEEP.exists() else set()
    newest = {}
    for r in runs:
        if completed(r) and run_map(r):
            newest[run_map(r)] = r.name  # names start with a timestamp, so the last one wins
    keep |= set(newest.values())
    freed = 0
    for r in runs:
        if not finished(r):
            continue
        video = r / 'video.mp4'
        if video.exists() and r.name not in keep and not (a.keep_completed and completed(r)):
            freed += video.stat().st_size
            print('video', r.name, f'{video.stat().st_size / 1e9:.2f} GB')
            if a.apply:
                video.unlink()
        telemetry = r / 'telemetry.jsonl'
        if a.gzip_telemetry and telemetry.exists() and time.time() - telemetry.stat().st_mtime > a.gzip_days * 86400:
            size = telemetry.stat().st_size
            print('gzip', r.name, f'{size / 1e9:.2f} GB')
            if a.apply:
                with telemetry.open('rb') as src, gzip.open(str(telemetry) + '.gz', 'wb') as dst:
                    shutil.copyfileobj(src, dst)
                telemetry.unlink()
                freed += size - Path(str(telemetry) + '.gz').stat().st_size
            else:
                freed += size * 0.85  # JSON telemetry compresses roughly 7:1
    print(('freed' if a.apply else 'would free'), f'{freed / 1e9:.1f} GB; kept videos:', ', '.join(sorted(keep)) or 'none')


if __name__ == '__main__':
    main()
