# doomANE

## Core ML: never load an .mlpackage with `ct.models.MLModel(...)` for prediction

`ct.models.MLModel("x.mlpackage")` compiles the package into a fresh `$TMPDIR/x_<uuid>.mlmodelc` on every load
and never removes it, and Core ML's ANE cache keeps a new entry for every new compiled path (hundreds of GB in a
few days). Load through `coreml_cache.py` (`coreml_cache.load(path, "CPU_AND_NE")`, or `compiled_path(path)`).
`MLModel(..., skip_model_load=True)` only reads the spec and is fine.

## Rules of the experiments

- The model chooses the command, target, weapon and movement; the executor carries them out and never
  assigns combat priority, item usefulness or the exit. An API error is never replaced by a heuristic.
  `tools/check_authority.py` (in the doomLaya checkout) audits every run.
- Wall time and game time agree within 5 %.
- Held-out maps (`doom/map_split.json`, plus Obsidian holdout sets) are played only from a frozen snapshot
  (`doom/snapshot_stack.sh`), and their results are never used for fixes. No map names, line numbers or
  coordinates in the stack's code.
- No hosted model sees game states or produces training data.
- Changes to doomLaya go into `ext/doomLaya-dev` and are saved as `doom/patches/doomLaya-dev.patch`
  (`git diff --binary -- . ':!.venv' ':!checkpoints' ':!runs'` from `ext/doomLaya-dev`, after `git add -N` for new files).
- `.env` is never printed or committed.
