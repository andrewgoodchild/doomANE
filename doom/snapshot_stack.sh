#!/bin/sh
# Freeze the current stack (doom/ scripts + the doomLaya-dev code) into ext/snap/<name>, so a game batch
# keeps running the same code while development goes on. Weights, the venv, run recordings and results
# are shared through symlinks; every game still snapshots its own sources into its run directory.
#   doom/snapshot_stack.sh exp1 && cd ext/snap/exp1/ext/doomLaya && .venv/bin/python ../../doom/repeat_map.py ...
set -eu
name=$1
repo=$(cd "$(dirname "$0")/.." && pwd)
snap="$repo/ext/snap/$name"
[ -e "$snap" ] && { echo "$snap exists" >&2; exit 1; }
mkdir -p "$snap/ext"
rsync -a --exclude __pycache__ "$repo/doom" "$snap/"
cp "$repo/coreml_cache.py" "$snap/"
ln -s "$repo/results" "$snap/results"
rsync -a --exclude .git --exclude .venv --exclude checkpoints --exclude runs --exclude __pycache__ \
  "$repo/ext/doomLaya-dev/" "$snap/ext/doomLaya/"
for link in .venv checkpoints runs; do ln -s "$repo/ext/doomLaya/$link" "$snap/ext/doomLaya/$link"; done
(cd "$repo/ext/doomLaya-dev" && git rev-parse HEAD && git diff --stat | tail -1) > "$snap/SNAPSHOT"
(cd "$repo" && git rev-parse HEAD && git status --short doom | head -20) >> "$snap/SNAPSHOT"
echo "$snap"
