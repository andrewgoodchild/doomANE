# POC 3: a Doom next-frame model on the Neural Engine ("doomANE: dream")

**Question.** Can a small action-conditioned next-frame model of Freedoom, trained in under an hour on this Mac,
run on the Apple Neural Engine at 35 frames per second and produce a recognisable one-second rollout?

**Pass criterion, fixed before training:** median at most 28 ms per frame on the Neural Engine including the
autoregressive loop overhead; one-step PSNR at least 25 dB on held-out frames; the 35-frame rollout still looks
like a Doom view (PSNR at step 35 reported, with an honest description of the GIF).

**Verdict: two of three pass, the rollout fails.** Speed: 0.64 ms per frame on the Neural Engine with the
autoregressive loop, 44 times under the budget. One-step quality: 28.6 dB on held-out frames against 24.9 dB for
copying the last frame. Rollout: within about ten frames the prediction decays into a brown-grey smear with the
weapon sprite and the on-screen text as the only sharp parts; PSNR at step 35 is 18.5 dB, one dB above holding
the first frame. So the Neural Engine is not the constraint at all; a deterministic 4-frame model is. The
timing headroom says a model 20 to 40 times this cost, or 10 to 20 denoising steps, still fits 35 fps.

## Data (`record.py`)

ViZDoom 1.3.1, Freedoom 2, development maps only (`doom/map_split.json`): MAP01, MAP02, MAP03. God mode, skill 3
so monsters move. Each episode warps the player to a random standable point (the `collect_frames.py` helpers,
unchanged) and walks a random policy with momentum for 700 tics: an action pattern over the seven buttons
MOVE_FORWARD, MOVE_BACKWARD, TURN_LEFT, TURN_RIGHT, MOVE_LEFT, MOVE_RIGHT, ATTACK, held for 3 to 20 tics,
biased to moving forward and turning. Every tic records the 256x192 view with no HUD, downscaled 2x to 128x96
RGB uint8, the button vector, x, y, angle and health, and the episode id. Training never crosses an episode
boundary.

| Map | Frames | Episodes | Recording time |
|---|---|---|---|
| MAP01 | 70,000 | 100 | 48 s |
| MAP02 | 45,000 | 65 | 30 s |
| MAP03 | 45,000 | 65 | 51 s |

About 900 to 1,500 frames per second headless, 129 s in all. Files: `ext/poc-data/dream/frames_<MAP>.npy` (uint8 memmap),
`meta_<MAP>.npz` (actions, variables, episode), `record_summary.json`.

## Model (`model.py`)

A conv U-Net with widths 48, 96, 192, 384 at 96x128, 48x64, 24x32, 12x16. Input: the last 4 frames as 12
channels in [0, 1] plus the 7-button action, mixed by a linear layer and broadcast as 7 channels. Output: the
next frame as a residual on the last frame, clamped to [0, 1]. Only ops the Neural Engine runs well: 3x3
convolutions, batch norm (folded at export), SiLU, average pooling, nearest upsampling, concatenation.
4,391,195 parameters. Deterministic, L1 loss. No diffusion step was tried within the time budget.

## Training (`train.py`)

The last 10 % of frames of each map are the test split. 143,176 training samples, 15,896 test samples.
AdamW, learning rate 1e-3 with a 300-step warm-up and a cosine decay over the wall-time budget, batch 64,
55 minutes on the GPU (MPS, torch 2.7). The checkpoint with the best test PSNR at a 500-step evaluation is
kept as `model.pt`.

18,824 steps in 55.0 minutes (5.7 steps per second, batch 64, about 8.4 passes over the training split).
Test PSNR on 1,024 fixed held-out samples, evaluated every 500 steps:

| Minutes | 1.5 | 5.9 | 14.7 | 23.5 | 32.3 | 41.0 | 49.7 | 54.1 |
|---|---|---|---|---|---|---|---|---|
| Test PSNR (dB) | 24.8 | 26.5 | 27.3 | 27.9 | 28.3 | 28.5 | 28.7 | 28.7 |

Still rising slowly at the end: more time or a wider model would help one-step quality, but see the rollout.
Files: `ext/poc-data/dream/run1/{model.pt,model_last.pt,log.json,train_summary.json}`, log `run1_train.log`.

## Export and timing (`export_time.py`)

`torch.jit.trace`, then coremltools 9.0 to an ML Program with fixed shapes and float16 weights and compute,
macOS 15 target. Loaded through `coreml_cache.load` (a compiled copy at a stable path; never
`ct.models.MLModel(package)`, see `CLAUDE.md`). The timing loop feeds each prediction back as the newest frame,
so the figures include the roll and the numpy to Core ML hand-off. 300 calls after 20 warm-up calls.

| Compute units | Median ms per frame | p90 ms | Frames per second at the median | Mean abs. difference from torch |
|---|---|---|---|---|
| CPU_AND_NE (Neural Engine) | **0.64** | 0.65 | 1,570 | 0.0035 |
| CPU_AND_GPU | 1.20 | 1.36 | 836 | 0.0002 |
| CPU_ONLY | 3.90 | 3.98 | 257 | 0.0013 |

Package 8.8 MB (float16). The GPU figure was taken with the GPU otherwise idle. The Neural Engine's
difference from torch (0.0035 on a [0, 1] scale) is float16 arithmetic, not an export fault.
Files: `ext/poc-data/dream/run1/{dream.mlpackage,export.json}`, the compiled copy in `results/coreml/compiled/`.

## Evaluation (`evaluate.py`)

All on the held-out 10 % of each map; the model never saw these frames. `results/poc/dream/evaluation.json`.

**One-step PSNR** (3,000 held-out samples), against copying the last frame:

| | MAP01 | MAP02 | MAP03 | All |
|---|---|---|---|---|
| Model | 28.3 | 30.3 | 27.5 | **28.6** |
| Copy the last frame | 24.6 | 26.3 | 24.2 | 24.9 |

**Autoregressive rollout** (24 held-out starts, the recorded actions, the prediction fed back as the newest frame):

| Step | 1 | 5 | 10 | 20 | 35 |
|---|---|---|---|---|---|
| Model PSNR (dB) | 28.3 | 22.4 | 20.4 | 19.6 | **18.5** |
| Hold the first frame | 24.0 | 20.2 | 19.1 | 17.9 | 17.4 |

**What the media show.** `rollout_strip.png` (real above, predicted below, at t+1, t+5, t+10, t+20, t+35) and
`rollout.gif` (2 s at 35 fps, real above, predicted below). The t+1 frame is sharp and close to the real one.
By t+5 edges soften; by t+10 the view is a brown-grey smear in which the weapon sprite and the static
"A SECRET IS REVEALED!" text (copied through from the context) are the only sharp parts, and they stay so to
t+35 while the real view turns and walks on. The L1 objective averages over an uncertain future, and once a
blurred frame is fed back the model has no sharp input to copy from. This is the known behaviour of
deterministic next-frame models and the reason the field uses diffusion with several steps.

**Action response** (200 held-out contexts; turn left against turn right from the same 4 frames, horizontal
shift of the upper view by column-profile correlation): mean shift +3.0 px for turn left and −2.9 px for turn
right, the correct directions, but only 36.5 % of contexts show both shifts non-zero and opposite. The model
has learned what turning does on average and applies it weakly, which is also what the blur suggests.

## How to run

```bash
# 1. record (one ViZDoom instance, about 2 minutes)
cd ext/doomLaya-dev && .venv/bin/python ../../poc/dream/record.py --maps MAP01 MAP02 MAP03 --frames 70000 45000 45000 --episode-tics 700 --out ../../ext/poc-data/dream
cd ../..
# 2. train for 55 minutes on the GPU
ext/vision-venv/bin/python poc/dream/train.py --minutes 55 --batch 64 --out ext/poc-data/dream/run1
# 3. export to Core ML and time per compute unit
ext/vision-venv/bin/python poc/dream/export_time.py ext/poc-data/dream/run1
# 4. PSNR, rollouts, GIF, strip, action response
ext/vision-venv/bin/python poc/dream/evaluate.py ext/poc-data/dream/run1 --out results/poc/dream
```

## What this POC does not show

- **Memory.** The model sees 4 frames, about a ninth of a second. It says nothing about keeping a corridor
  consistent after looking away, which is the open problem for every interactive world model. GameNGen holds
  about 3 s with 64 frames of context.
- **An agent.** Nothing plays inside the model, and nothing here bears on doomANE's completion results.
- **Diversity.** Three development maps and a random policy. Human-like trajectories, fights at range and
  the other 14 development maps are not covered. Held-out maps were not touched.
- **Sampling.** A deterministic L1 model blurs where the future is uncertain (monsters, projectiles, doors).
  A diffusion or flow variant with 2 to 4 steps would be the next step, and the timing headroom reported above
  says how many steps the budget allows.
