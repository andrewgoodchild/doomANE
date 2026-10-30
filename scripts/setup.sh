#!/bin/sh
# Set up doomANE on an Apple Silicon Mac: doomLaya v0.3.1 with our patch, its environment, the fine-tuned
# checkpoints, and the Core ML packages that put the encoder and question heads on the Neural Engine.
#
#   brew install uv ffmpeg cmake boost sdl2 openal-soft      # once (doomLaya's build needs boost and openal-soft)
#   scripts/setup.sh
#
# Layout it creates (all under ext/, which git ignores):
#   ext/doomLaya       clean doomLaya v0.3.1: the shared .venv, checkpoints/ and runs/
#   ext/doomLaya-dev   a git worktree of the same commit with doom/patches/doomLaya-dev.patch applied;
#                      .venv, checkpoints and runs are symlinks into ext/doomLaya
# Re-running skips the steps that are already done and resumes one that failed part way.
set -eu
repo=$(cd "$(dirname "$0")/.." && pwd)
cd "$repo"
mkdir -p ext results/coreml results/logs

if [ ! -d ext/doomLaya ]; then
  git clone --branch v0.3.1 https://github.com/azalio/doomLaya.git ext/doomLaya
fi
cd ext/doomLaya
if [ ! -f .venv/.doomane-ready ]; then
  [ -x .venv/bin/python ] || uv venv --python 3.13
  uv pip install --python .venv/bin/python -r requirements-lock.txt
  uv pip install --python .venv/bin/python -r requirements-model.txt
  uv pip install --python .venv/bin/python coremltools==9.0
  # ViZDoom 1.3.1 overflows a 128-line array on Freedoom MAP03; doomLaya's script builds the fixed engine
  .venv/bin/python scripts/install_vizdoom_sector_fix.py
  touch .venv/.doomane-ready
fi
# About 13 GB; the repo's training-assets/ (another 10 GB) is not needed here. hf resumes a partial download.
if [ ! -f checkpoints/laya-doom-v031/.doomane-ready ]; then
  .venv/bin/hf download azalio/laya-doom-map03 --revision v0.3.1 --local-dir checkpoints/laya-doom-v031 \
    --exclude 'training-assets/*'
  touch checkpoints/laya-doom-v031/.doomane-ready
fi
mkdir -p runs

cd "$repo"
[ -d ext/doomLaya-dev ] || git -C ext/doomLaya worktree add --detach ../doomLaya-dev v0.3.1
if ! git -C ext/doomLaya-dev apply --reverse --check ../../doom/patches/doomLaya-dev.patch 2>/dev/null; then
  git -C ext/doomLaya-dev apply ../../doom/patches/doomLaya-dev.patch
fi
for link in .venv checkpoints runs; do
  [ -e "ext/doomLaya-dev/$link" ] || [ -L "ext/doomLaya-dev/$link" ] || ln -s "../doomLaya/$link" "ext/doomLaya-dev/$link"
done
# The trained movement residual (config combat_move1) shares the encoder and head weights of doomLaya's movement head.
for part in model.safetensors encoder tokenizer; do
  [ -e "results/doom_heads/laya-combat-movement-v1/$part" ] || [ -L "results/doom_heads/laya-combat-movement-v1/$part" ] ||
    ln -s "../../../ext/doomLaya/checkpoints/laya-doom-v031/laya-v031-movement-numeric-v1/$part" \
      "results/doom_heads/laya-combat-movement-v1/$part"
done

# Core ML: one encoder package per sequence length, merged into one multifunction package, then the heads.
cd ext/doomLaya
if [ ! -d "$repo/results/coreml/doomlaya_enc_fp16.mlpackage" ]; then
  for length in 128 256 320 384 448 512 640; do
    .venv/bin/python ../../doom/export_doom_encoder.py --length "$length"   # one process per length
  done
  .venv/bin/python ../../doom/merge_doom_encoder.py --delete-sources
fi
.venv/bin/python ../../doom/export_doom_heads.py --missing

cd "$repo/ext/doomLaya-dev"
.venv/bin/python -m unittest discover -s tests -q
echo "doomANE is set up. Next: README.md, 'Play a map'."
