# Lab notebook: a local decision model playing Doom from the Apple Neural Engine

The dated record of every batch, from the zero-shot baseline (2026-10-18) to the end (2026-10-25). The summary is `WRITEUP.md`; paths below are relative to the repository root.

## Baseline (2026-10-18)

Laya, the 421M decision model, plays ViZDoom scenarios through a Jev-compatible `/v1/systemone` API
(the ANE server lived in the companion `ane-decision-model` project; `laya-serve` from the `laya` package
speaks the same API on the CPU or GPU), with every forward pass on the Neural Engine (Core ML,
`CPU_AND_NE`); the model never uses the GPU. This is the **zero-shot baseline**: the published checkpoints,
no Doom training, no prompt tuning after the first run.

Prior art: several projects have a decision model play Doom (TypeSafe's hosted Jev demo at about 10
decisions per second; `dylanbstorey/laya-doom` on the Mac GPU at about 62 ms per decision and 8.75
decisions per second; `azalio/doomLaya`, a fine-tuned Laya that finishes Freedoom MAP01 to MAP03 with a
scripted navigation and aiming controller, about 380 ms per decision on MPS or CUDA; others on MLX
or CUDA). None of them runs the model on the Neural Engine.

## How it works (`doom/play.py`)

Each decision, ViZDoom's object labels are turned into plain English: health, ammo, and each target
in view with its direction relative to the crosshair ("Demon slightly right (9 degrees), 160 units
away"). That text and one or two typed questions go to `/v1/systemone`:

- **aim** (choice: left / right / hold) and **fire** (noul) in `basic` and `defend_the_center`;
- **steer** (choice: left / right / forward) in `health_gathering`.

The answers (argmax) become button presses held for 4 tics, 8.75 decisions per second of game time.
Four policies use exactly the same interface: `random`; `scripted`, the rule the questions describe
applied to the same features (the ceiling for this interface); and the two Laya checkpoints. Scores
come from ViZDoom's synchronous mode (10 episodes each, seeds 1000 to 1009), so they do not depend on
latency; `doom/realtime.py` then checks the loop against the live 35 fps game.

## Results

| scenario | policy | reward | kills | survival s | agreement with the rule | ms p50 / p95 |
|---|---|---|---|---|---|---|
| basic | random | -69.0 | 0.8 | 1.8 | aim 0.36, fire 0.52 | – |
| basic | scripted | 73.0 | 1.0 | 0.0 | aim 1.00, fire 1.00 | – |
| basic | laya | 60.8 | 1.0 | 0.0 | aim 1.00, fire 0.27 | 48 / 54 |
| basic | laya-typed-decisions | 60.8 | 1.0 | 0.0 | aim 1.00, fire 0.27 | 47 / 49 |
| defend_the_center | random | 0.3 | 1.3 | 9.5 | aim 0.34, fire 0.49 | – |
| defend_the_center | scripted | 16.3 | 17.3 | 31.7 | aim 1.00, fire 1.00 | – |
| defend_the_center | laya | 0.7 | 1.7 | 9.0 | aim 0.13, fire 0.94 | 47 / 54 |
| defend_the_center | laya-typed-decisions | 0.8 | 1.8 | 9.1 | aim 0.11, fire 0.93 | 48 / 56 |
| health_gathering | random | 485.6 | 0.0 | 16.8 | steer 0.34 | – |
| health_gathering | scripted | 2100.0 | 0.0 | 60.0 | steer 1.00 | – |
| health_gathering | laya | 284.0 | 0.0 | 11.0 | steer 0.52 | 25 / 30 |
| health_gathering | laya-typed-decisions | 284.0 | 0.0 | 11.0 | steer 0.52 | 25 / 29 |

"Agreement with the rule" is the share of decisions where the policy chose what the scripted rule
would have chosen in the same state. In `basic` the survival column is not meaningful (ViZDoom
reports 0 when the episode ends on the kill).

Real time (ViZDoom `ASYNC_PLAYER`, the game runs at 35 tics per second regardless of the model):

| scenario | game seconds | wall seconds | decisions per second | ms p50 / p95 |
|---|---|---|---|---|
| defend_the_center | 9.69 | 9.38 | 14.9 | 48 / 54 |
| defend_the_center | 10.29 | 9.97 | 15.3 | 48 / 56 |
| defend_the_center | 7.0 | 6.68 | 13.9 | 51 / 58 |
| health_gathering | 11.0 | 10.95 | 18.8 | 26 / 29 |
| health_gathering | 11.03 | 10.97 | 19.6 | 25 / 31 |
| health_gathering | 11.0 | 10.94 | 19.7 | 25 / 29 |

Game time tracks wall time, so the loop keeps up: about 15 decisions per second with two questions
(48 ms) and about 20 with one (25 ms), every 2 to 3 game frames, against 8.75 to 10 for the other
Doom decision-model demos, without the model touching the GPU.

## What the baseline shows

- **Speed is solved; play is not.** The ANE answers fast enough for real-time control, but zero-shot
  Laya plays at the level of the random policy in `defend_the_center` (1.7 kills vs 1.3 random and
  17.3 scripted) and below random in `health_gathering` (11 s survival vs 17 s random and the full
  60 s scripted). It does solve `basic` (one monster, always in view): reward 61 vs 73 scripted.
- **It reads directions well.** When an enemy is in view, it picks the correct turn direction 97 of 98
  times (traced on one rule-driven episode).
- **It fails on the conditions inside option definitions**, the documented weakness of these models
  (arXiv 2609.26758: they follow the option's name, not its definition):
  - "Turn left: ... or no enemy is in view (turn to look for one)": with nothing in view it chose
    `hold` 27 of 27 times, so it stands still while enemies close in from behind.
  - "Walk forward: the nearest medikit is in the crosshair or slightly off it": it never chose
    `forward` (0 of 80 centred states); the state says "slightly left", so it picks `left`.
- **The fire question barely separates.** P(fire) is about 0.82 with the enemy in the crosshair and
  0.46 to 0.62 without, so at the 0.5 threshold it fires most of the time and wastes ammo.
- The fine-tuned `laya-typed-decisions` behaves the same as the general checkpoint here.

## Next, as planned on day one (superseded)

The same day the work moved to azalio's fine-tuned doomLaya on full Freedoom maps (next section), and
these scenarios were not played again. Neither step below was done here.

- **Prompt fix**: move conditions out of the option definitions into the state ("No enemies in view:
  search.") and give `forward` its own wording; refit the fire threshold. *Later:* the idea carried over
  to the full maps as planner facts and option wording tested offline first (weapon text, melee
  options); the baseline table was not re-run.
- **Fine-tune**: behaviour-clone the scripted rule and re-run this table. *Later:* not done; the
  fine-tunes that followed were the command-memory residual and the outcome-trained movement head, both
  on full maps.
- Videos of these runs (every game frame, the model's answer, probability and latency overlaid) were
  written with `doom/play.py --video <file>`; they are not in the repository.

## Fine-tuned model on full Freedoom levels, from the Neural Engine (2026-10-18)

`azalio/doomLaya` v0.3.1 (Apache-2.0) is a Laya fine-tuned on Freedoom with eight question heads
over one frozen encoder, plus small numeric heads; a controller executes the model's commands
(route, aim, use) but chooses none of them. Its published run used the Mac GPU (MPS).

Here its shared encoder runs on the Neural Engine: `doom/ane_doomlaya.py` replaces the one encoder
module all eight heads share (Laya 0.3.4 calls only `encoder(...).last_hidden_state`), with seven
fixed-length Core ML packages (128 to 640 tokens) exported by `doom/export_doom_encoder.py`
(cosine 0.999+ against PyTorch, 9 to 74 ms per pass). Heads, numeric residuals and all of
doomLaya's code run unchanged on the CPU. On doomLaya's own regression cases the ANE server agreed
with the CPU reference on 508 of 512 answers (99.2 %), with identical pass counts on seven of eight
case files, about 8x faster than the CPU. Stock ViZDoom 1.3.1 crashes on MAP03 (a sector with 168
lines overflows a 128-line array); doomLaya's patched build (`scripts/install_vizdoom_sector_fix.py`,
needs Homebrew boost and openal-soft) fixes that and was used for the runs below.

Their real-time suite (`tools/regression_suite.py`, skill 3, wall time equals game time):

| map | seed | result | time | kills | deaths | decision ms p50 / p90 | azalio's MPS run |
|---|---|---|---|---|---|---|---|
| MAP01 | 48 | **completed** | 147.8 s | 17 | 0 | 337 / 413 | completed, 168.4 s, 266 ms |
| MAP02 | 54 | **completed** | 396.3 s | 56 | 0 | 357 / 445 | completed, 420.4 s, 296 ms |
| MAP03 | 54 | not completed (1,200 s limit, alive) | – | 67 | 0 | 515 / 546 | completed, 347.1 s, 309 ms |

On MAP03 the model collected both keycards by 170 s, then from 296 s chose `use_switch` on all
remaining 1,508 decisions while alternating between three switches (332, 295, 2839) at full
confidence, and the controller shuttled between them until time ran out. Cause not established:
candidates are the ANE's small answer differences (0.8 % on the regression cases) compounding over a
long run, slower decisions (MAP03 prompts average about 1,900 tokens per decision, and the CPU, which
runs the heads, was shared with another job using about 13 cores), or ordinary run-to-run variance
(azalio reports a single series and says the maps and seeds were used in development).

An earlier attempt on stock ViZDoom also completed MAP01 (120.3 s, 16 kills, 0 deaths) and finished
MAP02 alive after 1,133 s, before the engine crashed loading MAP03.

Videos were written to each run directory (`ext/doomLaya/runs/`, not in the repository).

### What the MAP03 loop actually was, and where the time goes (2026-10-18)

**The loop is the numeric residual overriding Laya.** After 296 s the bot holds the blue and yellow keys, never the red one. In 829 of the 1,649 later decisions, `RedCard#45` is in "Reachable items", at a median of 4.9 m, and `pickup` is offered. Laya's own command answer (`numeric_residual.semantic_answer`) picks `pickup` in 347 of those 829. The small command MLP that adds a learned correction on top of Laya's logits (`laya-command-numeric-residual-v2`) turns every one of them into `use_switch`. The bot then rides lift #332, whose route needs the red key it never collects. More generally, the residual changes Laya's answer often: command 50 % on MAP01, 28 % on MAP02, 21 % on MAP03; item 72 %, 33 % and 40 %. The final decision is therefore often made by the MLP over parsed numbers, not by the language model.

**There is no vision in the loop.** The state is text built from engine internals: object lists, distances, mechanism ids and reachability. Perception is effectively perfect. The failure is in decision-making, not in seeing.

**Neural Engine share of a decision.** I replayed 80 recorded MAP03 packets against `ane_doomlaya.py --encoder ane`, with machine load average about 31–46 from other jobs:

| | per decision |
|---|---|
| wall time | 596 ms mean |
| encoder calls (all on the ANE, 0 fallbacks) | 6.2, each batch 1, run one after another |
| tokens encoded | about 2,220 (padding waste about 8 %) |
| time inside Core ML `predict` | about 250 ms (42 %) |
| everything else on the CPU: tokenisation, 2-layer decision heads per question, numeric residual MLPs, JSON | about 345 ms (58 %) |

Of the encoder time, 160 of the 494 calls were at L640, about 74 ms each, and they make up about half of it. The ANE is idle more than half of each decision. Even while it works it runs a 150M-parameter encoder at batch 1, so its average utilisation is low. Ways to put more on it: export the decision heads to Core ML too; run the independent question groups concurrently, or batch them; drop the extra `look_gate` and enemy-ranking passes where a question doesn't need them.

### Moving the whole decision onto the Neural Engine, and three MAP03 fixes (2026-10-18)

**Speed.** I profiled in-process by replaying 40 recorded MAP03 packets through `doom/profile_doomlaya.py`, with the heads on the CPU as in the game runs. Tokenisation is about 3 ms per decision, so it stays on the CPU. The Neural Engine can't run string code anyway.

| change | ms / decision | Neural Engine share | answers identical to CPU heads |
|---|---|---|---|
| encoder on ANE, heads on CPU (previous runs) | 612 | 41 % | – |
| + 2-layer heads and scorer on ANE (`doom/export_doom_heads.py`, `--heads ane`) | 276 | 98 % | 210 / 210 |
| + skip one-option base questions (`--skip-forced`; `combat` is "hold"-only in 92 % of requests) | 206 | 98 % | 210 / 210 |
| concurrent Core ML calls (2 threads), or the batch API | no change: 36.1 vs 36.8 ms; 8.3 vs 8.1 ms per row | – | – |

The ANE runs one model at a time, and a single 384-token encoder call already keeps it busy. So concurrency and batching don't help. In the game the server now answers in a median of 110–140 ms, against about 420 ms before. HTTP adds only about 6 ms. The seven per-length encoder packages are now one multifunction package (`doom/merge_doom_encoder.py`): outputs identical at every length, 5.3 GB down to 0.76 GB.

**Fixes tried on MAP03.**
- **Guard** (`--residual-guard command=0.7`: keep Laya's answer when its top probability is at least 0.7). It got the red key at 285 s, but then lost: it overrode the residual 629 times after the keys, from `use_switch` (exit switch #2839) to `pickup`, so the bot zig-zagged between the exit switch and nearby items until the time limit. Rejected.
- **Repair fine-tune with memory** (`doom/repair_command_residual.py`, `doom/command_memory.py`, `--command-memory`). The command residual gets five memory and observation inputs, zero-initialised, and is trained on our recorded runs only. Repair label: `pickup` where a key not held is reachable and no enemy is visible. Retention: every decision from the completed MAP01 and MAP02 runs and the MAP03 prefix. Offline it picks `pickup` on 249 / 249 held-out states and keeps 1,615 / 1,615 retained decisions. Same-map data, so this is not generalisation.

| suite (fast pipeline + repaired memory residual) | result | time | kills | deaths | server ms p50 / p90 |
|---|---|---|---|---|---|
| MAP03 | **completed** | 327.7 s | 73 | 0 | 131 / 166 |
| MAP01 | completed | 145.4 s | 18 | 0 | 109 / 140 |
| MAP02 | completed | 1,124.2 s | 187 | 4 | 127 / 154 |

**But the fine-tune didn't cause the MAP03 win.** Replaying the run, the original residual makes the same choice on all 643 MAP03 decisions. This time the bot had the red key at 261 s, before the state that trapped the earlier run, so the repair never fired. On MAP02 only 2 of 2,206 decisions differ. That MAP02 run, slower and with 4 deaths against 396 s and 0 before, is most likely variance, not the repair. MAP03 was won either by faster decisions or by run-to-run luck. An ablation run with the fast pipeline and the original residual is the control.

**Control: fast pipeline with the original residual** (no guard, no memory):

| map | fast + original residual | fast + repaired memory residual | earlier slow run (encoder-only ANE) |
|---|---|---|---|
| MAP03 | completed, 780.0 s, 3 deaths | completed, 327.7 s, 0 deaths | timed out (switch loop) |
| MAP01 | completed, 162.9 s | completed, 145.4 s | completed, 147.8 s |
| MAP02 | **timed out** at 1,800 s (stuck between switches #728 and #805) | completed, 1,124.2 s, 4 deaths | completed, 396.3 s |

One run per cell, and outcomes flip between configurations whose decisions are nearly identical: the repair changed 0 MAP03 decisions and 2 MAP02 decisions. So a single suite can't rank these variants. The trajectories are chaotic, and one different decision, or a different decision timing, sends the run down a different path. Faster decisions completed MAP03 in both fast runs, against 0 of 1 slow runs, which is suggestive, not proof. To rank variants you need several runs per configuration, with varied seeds or timing.

## Repeated runs, executor fixes, commitment and a System 1 planner (2026-10-19)

**Evaluation.** `doom/repeat_map.py` plays maps × seeds × server configurations. It verifies each run with doomLaya's `tools/verify_run`, and counts a completion only when the run is also valid and passes the authority check. Results go to `results/doom/repeat_<map>[_<suffix>].json`, with a new suffix whenever doomLaya itself changed. Seeds 54, 1, 2 and 3 are used throughout.

**The rule-based player is not perfect either.** `training/map3_teacher.py` (azalio's label rules, no model) completes MAP03 3/4 at its usual 16-tick latency and 4/4 at 5 ticks, the latency of our setup, in 441–953 s with 1–5 deaths. Seed 1 is hard for everyone.

**Why the failed runs failed.** The failures were mostly the executor and Laya trapping each other, not judgement:

| failure | cause | fix |
|---|---|---|
| MAP03 exit loop: exit → `open_door` → lift #332 → exit, every ~2 s for 10 minutes | Door 298 is an open-wait-close door. It opens, the executor reports `open_door` "unavailable" (its closed target is gone), Laya then prefers the lift 0.9/0.1, and the door shuts again | `--commit-resume`: return to a door-blocked command after `open_door` |
| Frozen at a switch for 1,019 s | The executor pressed USE from behind the switch line, so the navigator's "no path" result never became `blocked` | executor: only USE from the front side |
| Standing 0.37 m from a Shell for 1,700 s | The executor reported "executing" with no progress | executor: a 10 s no-progress watchdog reports `blocked` and sets the target aside for 30 s. A first version was undone on the next tick |
| MAP02 lift #805 ↔ two shotguns, every 1.5 s | Near-tie flips of command and item target | `--commit-pickups` with `--commit-margin 0.3` |
| Holding against a 0.99999 pickup (MAP01 timeouts) | Unbounded holds | `--commit-floor 0.02` and the margin |
| Valid completions ruled invalid (2 runs) | The navigator braked and strafed during a melee approach, and the authority audit requires those movements to come from the model | executor: forward and turn only |
| MAP03 seeds 1 and 54: 6–12 deaths | Big fights, and each death restarts the map | not addressed |

The executor changes are in `doom/patches/doomLaya-executor-fixes.patch`, applied to `ext/doomLaya`. Each run's source snapshot records them, and doomLaya's 336 tests pass.

**Screens (4 seeds, completions that pass the checks):**

| setup | MAP01 | MAP02 | MAP03 |
|---|---|---|---|
| fast, original executor | – | – | 2/4 (but all 4 runs fail the authority audit, so 0/4) |
| commit / sample / memory, original executor | – | – | 2/4 / 3/4 / 2/4 |
| resume (switch-side fix) | 3/4 | 2/4 | 3/4 |
| fast (switch-side fix) | 4/4 | 2/4 | 1/4 |
| resume2 (+ watchdog, floor, pickups) | 1/4 | **4/4** | 3/4 |
| fast (+ watchdog) | 4/4 | 2/4 | 2/4 |
| resume3 (+ margin, watchdog v3) | 3/4 | 3/4 (+1 invalid) | 2/4 |

With 4 seeds a cell is noisy, but holding commands clearly helps MAP02 and MAP03, and unbounded holds hurt MAP01. No single setup yet finishes all three maps reliably.

**New System 1 questions, zero-shot** (`doom/eval_questions.py`, labels from recorded telemetry and outcomes, 200 balanced cases each). Laya answers at chance:

| question | label | accuracy | AUROC |
|---|---|---|---|
| making progress toward the target? | target > 1 m closer in 5 s | 0.44 | 0.36 |
| pull back to heal? | died within 6 s vs survived 20 s | 0.49–0.51 | 0.47–0.48 |
| keep going instead of switching? | abandoned target re-chosen and not reached vs switch paid off within 10 s | 0.47–0.49 | 0.43–0.52 |

The results are the same as `noul` and as two-option `choice`. Laya says yes 70–95 % of the time whatever the facts. Progress is a measurement and stays in code (the watchdog). Retreat and keep would need trained heads and better labels.

**A System 1 planner** (`doom/planner.py`, a wrapper around `agent.py`, config `planner`). Code lists the objectives reachable now: missing keys, unused switches and lifts, the exit, and explore. Each carries what it opens two steps ahead, from what-if copies of doomLaya's navigator (key held, remote door open, lift raised), plus its history. Laya picks one through the same `/predict` endpoint as an ordinary `choice` question, only on events. The objective then narrows the 2 Hz route options, while fighting, doors and nearby pickups stay free. Each plan takes 90–125 ms the first time and 6–17 ms when cached. Zero-shot at the start of each map, Laya picks door switch #839 (MAP01), #662 (MAP02, which opens lift #728 and so the red key) and #1850 (MAP03, which opens the blue key): the first steps of the real routes. An unreachable key was preferred until only reachable objectives were offered. The first planner game finished MAP01 seed 54 in 112 s, against 155–168 s for the other setups. The full screen is in `results/doom/repeat_*_exec4.json`.

### Planner v3–v6 and a fight question (2026-10-20)

Completions out of 4 seeds (54, 1, 2, 3). Each death restarts the map from a pistol start.

| Setup | MAP01 | MAP02 | MAP03 | Total |
|---|---|---|---|---|
| fast (switchfix) | 4 | 2 | 1 | 7/12 |
| resume2 (commitment layer) | 1 | 4 | 3 | 8/12 |
| planner v3 | 4 | 3 | 0 | 7/12 |
| **planner v6** | **4** | **3** | **3** | **10/12** |
| planner v6 + fight question, no back-off | – | – | 1 | – |

The changes from v3 to v6 are all code-side facts and filters. Laya still picks one option.
- Lifts are offered even when already activated, and a lift's `activated` flag is left out of the state signature. The flag had toggled the signature and caused ping-pong on lift #332.
- A lift counts as done once its ride ends, and is then set aside for 45 s.
- A what-if lowers the lift to its lowest neighbouring floor and offers a lift only if that opens something new.
- The stuck-item clock counts only time spent targeting that item, and is forgotten after 30 s.
- Unowned weapons are errands only within 15 m.
- An `open_door` whose door doomLaya has already failed to route to is dropped from the 2 Hz options. MAP03 seed 54 had re-chosen an unroutable door #181 2,003 times.

The two v6 failures are deaths, not plans:
- MAP02 seed 54: 8 deaths, 7 of them in the demon room on the way to switch #662. The bot stands still at 1.6 m while health falls 50 → 6 → dead.
- MAP03 seed 1: 4 deaths, each restarting the map. One planning flaw was also found. When nothing is reachable, every objective is offered, and Laya took "red key, not reachable" (p 0.42–0.54) over "lift #332, opens red key" (0.08).

**Fight question** (`PLANNER_FIGHT=1`, config `fight`). This is engage/avoid recast as a code-written options question, the same way as the planner. The options are fight, back off, heal and move on. Each carries one outcome sentence that code computes: whether the player outlasts the visible enemies, from health against observed and nominal damage rates, and enemy health against weapon DPS.
- Offline, Laya chose fight in 14–18 of 18 cases whenever the numbers appeared in the option text, whatever the outcome. With a bare outcome sentence it followed the outcome: back off or heal in 16 of 30 cases when "the player dies", fight in 24 of 30 when "the player survives".
- The estimator flags only 41 of 94 pre-death moments as losing.
- In play, v1 with back-off scored MAP03 0/4, with 6–11 deaths per run. In 17 of 22 matched deaths a recent answer was "back off", and doomLaya's `retreat` walks backwards without firing.
- Back-off is now behind `PLANNER_BACK_OFF=1`. Without it, MAP03 scored 1/4 with 2–8 deaths, still worse than plain v6.
- The decision layer works. The executor can't yet carry out "avoid" safely, so the question stays off by default.

Disk: `doom/cleanup_runs.py` (a dry run unless `--apply`) deletes verified videos except the newest completed run per map, and `--gzip-telemetry` compresses old telemetry. `repeat_map.py` deletes a failed run's video after verification unless `--keep-videos` is given.

### Planner v7 and v8 did not beat v6; deaths are run-to-run noise plus combat (2026-10-21)

The v7 and v8 changes were in the development history (not in this repository's history):
- v7: set-asides escalate; an optional item set aside twice is dropped for the map; in the nothing-reachable fallback, the step that opens a target is offered instead of the target.
- v8: re-plan when health falls below 50 or 25, and a separate "keep going" vs "heal first" question.

`doom/planner.py` is back at v6. `doom/planner_v6.py` (config `planner6`) is the frozen copy used for same-session comparisons.

| MAP03 | Completed | Deaths per run |
|---|---|---|
| v6, all runs (3 sessions) | 13/25 | 3.0 |
| v7 | 5/13 | 3.7 |
| v8, same session as v6 | 3/10 | 3.7 |
| v6, same session as v8 | 6/10 | 3.2 |

Deaths vary a lot between repeats of the same seed: v6 seed 5 had 0 deaths, then 6, then 4. The 6/10 from the first 10-seed v6 screen was on the lucky side; MAP03 is about 50 %. *Later:* every MAP03 arm of the stacks from exec5 on pools to 16/51 (31 %).

Three possible causes were checked and ruled out:
- **Model loader:** the new loader's encoder outputs are bit-identical to `MLModel`'s.
- **CPU load from other jobs:** no run missed a tick, because the game steps in lockstep with the client.
- **v7's own changes:** frozen v6 did as badly as v7 in the same session.

Deaths on MAP03, 91 in total:
- 5 s before dying, median health was 45, and 56 % were below 50.
- A third had no enemy in view 1 s before death.

Health choice:
- Offline, inside the objective list, Laya picked an offered health item in only 12 of 43 decisions below 25 health.
- As a separate two-option question ("Keep going: X." / "Heal first: Y; health is N.") it picked heal in 35 of 43. Across all 122 decisions where health was offered (below 50), it picked heal 82 % of the time, so the effect comes mostly from the framing, not from how low health is.
- In play, v8 healed after 81 of 99 heal-first questions, and only 1 of 37 deaths happened while it was heading for health. More healing did not reduce deaths, and the detours cost completions.

The remaining failures are fights the executor loses, not choices the planner makes.

### Executor fire gate (exec5, 2026-10-21)

Two executor changes in doomLaya (`doomlib/combat.py`, patch in `doom/patches/doomLaya-executor-fixes.patch`). Neither chooses a target or a weapon; they only decide whether the chosen weapon fires at the chosen target this tick.
- **Fire inside the target's body.** The aim tolerance is the target's angular half-width, `max(5°, 0.8·atan(20 units / distance))`: 17° at 1.6 m, 5° at 10 m. Before, a fixed narrow gate held fire in 37 % of ticks with the target under 2.5 m, and the bot fired on only 48 % of them.
- **No rockets inside the blast radius.** The rocket launcher fires only beyond 4 m, and the weapon question says "Can hit visible enemies from here: no" inside that. Before, 20 of 186 MAP03 deaths followed a rocket choice, 16 with the target within 6 m.

`DOOM_FIRE_LEGACY=1` (config `legacyfire`) restores the old gates. Same session, MAP03, seeds 54 and 1–9 (`results/doom/repeat_map03_exec5.json`):

| MAP03 | Completed | Deaths | Deaths per run |
|---|---|---|---|
| new fire gate (`planner`) | 4/10 | 31 | 3.1 |
| legacy fire gate (`legacyfire`) | 1/10 | 47 | 4.7 |

All 20 runs are valid with authority intact. The legacy arm's 1/10 is low next to v6's pooled 13/25, so this session ran hard, but the same-session gap holds: a third fewer deaths, and wins on seeds 1, 2, 6 and 9 against only seed 3 for legacy.

The new gate doesn't hurt the easier maps (`results/doom/repeat_map0{1,2}_exec5.json`, same seeds):

| Map | New fire gate | v6 screen with the old gate |
|---|---|---|
| MAP01 | 10/10, 0 deaths | 10/10 |
| MAP02 | 8/10, 39 deaths | 9/10, 40 deaths |

The MAP02 failures are different seeds each time (now 1 and 5, then 54), so this is within the noise.

### Weapon options as shots left (exec6, 2026-10-22)

Deaths in exec5 showed Laya's weapon head choosing a weapon that cannot hit once the chaingun runs dry: of 1,084 such decisions with a loaded shotgun, 41 % chainsaw, 12 % a rocket launcher with 2 rockets, 9 % the empty chaingun. The head's training data never contained a chainsaw, and a 50-shell shotgun and a 2-rocket launcher both read "Powerful ranged gun… Ammo: N". The options now end in "Shots left: N" (ammo divided by the cost per shot) or "No ammunition needed." for melee (`doomlib/policy.py`). Laya still chooses. Offline, on 200 logged questions with an empty chaingun, choices that could not hit fell from 59 to 6; on 200 other weapon questions they stayed at 6. `DOOM_WEAPON_TEXT_LEGACY=1` (config `legacyweapon`) restores the old text.

Same session, MAP03, seeds 54 and 1–9 (`results/doom/repeat_map03_exec6.json`), both arms with the new fire gate:

| MAP03 | Completed | Deaths | Deaths holding chainsaw, rockets or an empty gun | Chainsaw choices |
|---|---|---|---|---|
| shots left (`planner`) | 3/10 | 44 | 1 | 0 % |
| ammo (`legacyweapon`) | 3/10 | 38 | 5 | 2 % |

The change removes the failure it targets but does not change completions. The difference in deaths is within run-to-run noise: single seeds range from 0 to 8 deaths. Pooled over exec5 and exec6, the new fire gate completes MAP03 in 7 of 20. In this session 23 of 44 deaths in the shots-left arm were in the nukage area (x 2200–2600), against 14 of 38.

### MAP03 deaths: the yellow-key ambush (2026-10-22)

All 113 deaths from the 30 MAP03 runs with the new fire gate (exec5 `planner`, exec6 both arms):
- **40 % are one trap.** 45 deaths come 4–18 s after the yellow key pickup at the north end of the long corridor (x ≈ 2400, y ≈ 680). The pickup opens two monster closets (sectors 132 and 189, reported in `geometry_changes` 0.2 s later) about 16 m back down the corridor, holding 4 ShotgunGuys and 4 imps. With no enemy in view yet, Laya returns to the planner objective and walks south past the closets, and the ShotgunGuys hit it from about 6 m: 85 → 10 HP in 4 s in one trace. The bot survives 35 of 80 such encounters. Survivors entered with a median of 74 HP, the dead with 54; the fights look the same otherwise (same targets, movement, fire rate).
- **The yellow key is required.** All 10 winning lives collected blue, yellow and red, in that order, yellow at 150–220 s into the life. Each death restarts the map, so every life has to get through this trap.
- Only a minority of the deaths in this area involve damaging floor; the earlier "nukage area" label was wrong.

Offline (`/predict` replays of 463 command decisions in the 3 s after the key with no enemy in view), Laya's choice did not change in any case when the state said "Hidden areas opened nearby…" or `wait` was reworded as "Hold this position…": the command head reads its own projection of the state and a numeric residual, not free text. As a separate hold/go question, the answers followed the wording, not the situation (one framing chose hold in 22 of 161 ambush states but 47 of 80 ordinary states). Zero-shot wording does not reach this decision; holding would need a trained gate like `look_gate`.

### Diagnostic: a forced hold after the yellow key (diag1, 2026-10-22)

To measure what holding would be worth before training anything, the executor was made to hold the bot in place (turning towards the opened sector, still free to fire) for 8 s whenever hidden areas open within 30 m (`DOOM_DIAG_AMBUSH_HOLD=1`, config `diaghold`). This is code choosing the movement, against the authority rule, so it is a diagnostic only, not a model result. Same session, MAP03, seeds 54 and 1–9 (`results/doom/repeat_map03_diag1.json`):

| MAP03 | Completed | Deaths | Yellow-key trap survived (no death within 30 s) | Deaths in the trap room | Lives reaching the red key |
|---|---|---|---|---|---|
| forced hold (`diaghold`) | 0/10 | 56 | 9/33 | 24 | 0 |
| `planner` | 3/10 | 43 | 6/31 | 27 | 3 |

Earlier planner arms (exec5, exec6, both arms) survived the trap in 35 of 80 cases, so neither arm here is unusual and the hold does not help. The ShotgunGuys do not come out of their closets while the bot waits at the dead end: in the traces the bot stands for 8 s with nothing in view, then walks south and is shot from 6–8 m as before. Several "survivors" by a 20 s count died at 21–26 s, so the hold mostly delays the death. The hold also fired 27 times away from the trap (doors and lifts near the start area); only 4 deaths followed one within 20 s, so the extra deaths elsewhere are mostly noise.

Waiting is therefore not the missing behaviour. What separates survivors is health on entry (median 74 vs 54) and fighting the ShotgunGuys at the closets rather than walking past them. The next things to try are getting health before the key, or advancing against the closets in combat movement rather than navigation. *Later:* neither was tried as such; the combat work went to movement and command rollouts from saved states (2026-10-25).

### Held-out map: MAP04 (holdout1, 2026-10-22)

To check for overfitting, the current stack (same heads, planner, executor; config `planner`) played MAP04, which was never used for training or tuning, on fresh seeds 101–105 with a 1200 s budget (`results/doom/repeat_map04_holdout1.json`; `doom/repeat_map.py` now accepts MAP04 and MAP05 as held-out maps).

| MAP04 | Completed | Deaths per run | Keys collected | Grid cells visited (4 m) |
|---|---|---|---|---|
| `planner`, seeds 101–105 | 0/5 | 1, 11, 6, 3, 0 | 0 | 126–132 |

All runs are valid and pass the authority check. Fighting is not the problem (seed 105 has no deaths). The bot never finds the way on: the map knowledge lists a red and a blue key, door switches #389 and #540 and the exit, but the navigator finds no path to any of them ("No traversable route with the current keys"; switches "Unreachable" in about 6,700 of 11,700 decisions). The planner cycles through these objectives, each failing in about a second, and falls back to `explore` (50–80 % of command decisions). Explore covers the same ~130 cells in every run and finds nothing new. The bot does see both keys (as close as 19 m) but they never become reachable.

So the MAP01–03 results do not carry over to a new map, and on MAP04 the first gap is route knowledge, not the decision heads: whatever opens the way out of the start area (a mechanism the navigator does not model, or a route explore never takes) is missing. The heads have not yet been tested on a new map past that point.

### MAP04 route fixes, and a map split (2026-10-23)

MAP04 with switch type 102 understood (`repeat_map04_holdout2.json`, same seeds 101–105): 0/5, but every run now collects the red key (before: none). The route is then blocked at the blue key. The key stands on a fenced pedestal in the middle of a room. Doom picks an item up when the two bodies' boxes overlap (16 + 20 units), also across a fence, but the navigator only accepted a contact point within 30 units of the item, so the key was masked as unreachable. In the development copy (`ext/doomLaya-dev`, `doom/patches/doomLaya-dev.patch`), `pickup_point` now also accepts a reachable cell up to 44 units from the item on each axis, and on arrival the executor presses on towards the item. Simulated on the map data, the whole MAP04 route is then connected: switch 883 → red key → red door → blue key across the fence → blue door switch 427 → switch 540 (doors 14 and 79) → switch 389 → exit. The game test waited until the held-out evaluation freed the server; it is `dev1` below.

From here maps are split (`doom/map_split.json`): development MAP01–04 and the even maps, held-out the odd maps from MAP05 plus MAP31–32, which are only played by a frozen stack. The first held-out evaluation (`heldout1`: MAP05, 07, 09, 11, 13, 15, seeds 201–203) runs the stack as committed on that day in the development history (`a391cf9`, not in this repository) from `ext/doomLaya`, then clean azalio v0.3.1 (config `v031`, from a separate clean checkout) on the same maps and seeds.

### Held-out evaluation heldout1: our stack vs clean doomLaya v0.3.1 (2026-10-23)

Six held-out maps, seeds 201–203, 1200 s each; `planner` is the stack at `a391cf9`, `v031` is azalio's v0.3.1 unchanged (`results/doom/repeat_map{05..15}_heldout1.json`, per-run details in `results/doom/heldout1_compare.json`). Neither stack finished a map: 0/18 each.

| Map | Config | Valid | Deaths/run | Kills/run | Doors opened/run | Cells visited (4 m)/run | Keys per run |
|---|---|---|---|---|---|---|---|
| MAP05 | planner | 3/3 | 4.3 | 114 | 17.7 | 179 | –, blue, blue+yellow |
| MAP05 | v031 | 3/3 | 1.3 | 70 | 8.0 | 161 | blue+yellow, blue, blue+yellow |
| MAP07 | planner | 3/3 | 20.0 | 270 | 57.7 | 149 | red, red, red |
| MAP07 | v031 | 3/3 | 17.3 | 263 | 59.7 | 138 | –, red, red |
| MAP09 | planner | 3/3 | 2.0 | 63 | 39.3 | 103 | blue, –, blue |
| MAP09 | v031 | 3/3 | 0.3 | 30 | 22.7 | 101 | – |
| MAP11 | planner | 3/3 | 26.7 | 246 | 20.7 | 162 | – |
| MAP11 | v031 | 0/3 | 24.3 | 230 | 19.3 | 143 | – |
| MAP13 | planner | 3/3 | 22.0 | 262 | 26.3 | 34 | – |
| MAP13 | v031 | 1/3 | 19.3 | 239 | 27.0 | 39 | – |
| MAP15 | planner | 2/3 | 1.3 | 10 | 3.7 | 23 | – |
| MAP15 | v031 | 1/3 | 0.0 | 1 | 1.0 | 9 | – |

Totals: `planner` 8 keys, 496 doors, 1,949 cells, 229 deaths, 17/18 valid, 18/18 authority; `v031` 7 keys, 413 doors, 1,774 cells, 188 deaths, 11/18 valid, 13/18 authority (v0.3.1 failed the authority check in five runs, e.g. strafes and backward attacks with no model decision; the other invalid runs did not sustain real time). Our stack explores and fights a little more, and dies a little more; on the new maps the two are about level. The MAP01–03 gains over v0.3.1 therefore come from the development maps and do not transfer. On these maps nobody got past the first key or two, so the limit is still route knowledge (mechanisms the navigator does not model), as on MAP04. These runs are not used to drive fixes; the next held-out evaluation will use other odd maps or new seeds. *Later:* there was no second held-out evaluation before the project was parked, so the generic route rules below have not been played on a held-out map.

### System 2, planner what-ifs, and MAP04 dev1–dev3 (2026-10-23/24)

System 2 (`doom/system2.py`) asks the planner's objective question of a local LLM through MLX (`mlx_lm.server` on port 8011) instead of Laya. It runs in a background thread so the game stays real time. Until an answer arrives, the previous objective stays. After a done or failed objective there is no objective, which narrows nothing. An invalid answer is asked again once; a second invalid answer or an API error stops the run (no heuristic). An answer that arrives after the options changed is discarded and asked again. The prompt also gives the objectives so far and how each ended. Configs: `system2` (Qwen3-8B-4bit) and `system2_27b` (Qwen3.8-27B-8bit). In offline replay of 20 logged dev-map questions (`doom/system2_replay.py`, held-out maps refused), all answers were valid; median latency was 0.4 s (8B) and 2.4 s (27B).

Planner facts (`doom/planner.py`): a switch what-if now rebuilds the navigator with the moved floor or door, because closed sectors have no cells. Keys and items count as reached by touch (`pickup_point`), and key options list the key-locked switches they unlock. At the MAP04 start the options now say "switch 883 opens red key (which opens blue key)". In the navigator (dev patch), edges rejected after a stuck now expire after 60 s.

| MAP04, seeds 101–103, 1200 s | Completed | Deaths/run | Keys/run | Blue key |
|---|---|---|---|---|
| dev1 `planner` | 0/3 | 9, 8, 2 | 2, 1, 3 | 0 |
| dev1 `system2` | 0/3 | 10, 10, 10 | 2, 1, 1 | 0 |
| dev1 `system2_27b` | 0/3 | 11, 8, 7 | 2, 3, 0 | 0 |
| dev2 `planner` | 0/3 | 2, 12, 10 | 1, 0, 1 | 0 |
| dev2 `system2` | 0/3 | 11, 2, 9 | 1, 2, 2 | 0 |
| dev2 `system2_27b` | 0/3 | 9, 9, 9 | 2, 2, 0 | 0 |
| dev3 `planner` | 0/3 | 10, 8, 10 | 1, 1, 4 | 0 |
| dev3 `system2_27b` | 0/3 | 6, 9, 12 | 2, 2, 0 | 0 |

All runs are valid and pass the authority check. All keys collected are red (counted again after each death, since a death restarts the level from a pistol start). In dev1 the 8B model looped on weapon pickups. From dev2 on, every config chooses the blue key once red is held, and the option says "reachable now".

dev3 rules out blocked edges. When the blue-key objective failed, the planner's `failure_check` found no blocked edges, and the key was unreachable with or without them. Of the eight blue-key attempts in dev3, five ended in a death, and the other three ran into the 30 s no-progress limit while fighting. The closest the bot got was 604 units, in an attempt that started at 76 HP: near (62, 1095) it lost about 50 HP and fell back. After the red key the bot usually has 15–50 HP. So on MAP04 the limit is now survival on the way to the blue key, plus a death resetting all progress. The choice of objective is no longer the limit (planner, 8B and 27B fail alike). Results: `results/doom/repeat_map04_dev{1,2,3}.json`.

### Route knowledge and combat for unseen maps: rules and pipeline (2026-10-24)

The research survey (`docs/research/Doom AI research survey/README.md`) ranked two groups of changes: generic route knowledge (a table of linedef specials, effect checks, prerequisite chaining, one-way drops) and combat reflexes that the model chooses. Both are built so that they work the same on a map nobody has looked at:

- **One rule set for every map.** What a line does comes from `doomlib/specials.py`, a table of the vanilla Doom 2 linedef specials taken from the engine's own rules: trigger (use, walk-over, gun, manual door), once or repeatable, the mover, where it stops (lowest or highest neighbouring floor, lowest ceiling minus 4 for doors, and so on), and the key. No map name, line number, sector number or coordinate is used in the stack's logic or configuration (some comments name the map where a problem was found). The map file's geometry, specials, tags and thing positions are read as before (doomLaya already used geometry, switches, key positions and the exit). The live engine supplies sector heights and what is on screen. This is still more than a first-time human player knows: it is like a full automap that also says what every line does. A stricter "only what has been seen" mode would be a separate experiment.
- **Development and evaluation are kept apart.**
  - Fixes and tuning use only the development maps: Freedoom MAP01–04 and the even maps, plus the Obsidian set `dev1`, generated with `doom/oblige_maps.py` (Obsidian, Oblige's successor, built for arm64 with `doom/patches/obsidian-macos-arm64.patch`; the same seed and settings give the same WAD).
  - The odd Freedoom maps and the Obsidian set `holdout1` (seeds 1001–1010) are played only by a frozen snapshot (`doom/snapshot_stack.sh`). `repeat_map.py` refuses held-out Obsidian WADs unless it runs from `ext/snap/` with `--evaluation`.
- **The model still chooses.** Code lists options and facts; Laya (or System 2) picks the objective, command, target, weapon and movement. Mechanisms beyond doomLaya's own set reach the 2 Hz questions only once the objective question has chosen them.
- **Combat facts are what the player can see.** `doomlib/combat_facts.py` uses only objects in the labels buffer (on screen): a projectile is any visible moving object of category Explosive, with its engine velocity. From these it reports incoming projectiles (closest approach within a second, which side they pass), health, the nearest melee-only monster and the number of visible hitscan shooters.
- **Training labels come from the game.** `agent.py --fight-states` saves the engine state at every movement question (ZDoom save, about 6 ms and 60 KB). `doom/fight_rollouts.py` reloads each state once per movement (stationary, strafe left, strafe right, backward), carries out the attack as the executor would, and scores health and armor lost, damage dealt and death. Rollouts from a saved state are deterministic. `training/finetune_combat_movement.py` trains a new movement residual on these outcomes (projection `movement-compact-v3-combat`) and picks the epoch by validation regret on other runs. No hosted model is involved.

Route knowledge so far, measured with the static check `doom/route_solver.py`: from the start, collect reachable keys, use reachable mechanisms with the table's moves, ride lifts both ways, test the exit. It is optimistic: it ignores monsters and timing and does not notice a one-way drop that would strand the player. `--diagnose` shows where the rules stop each missing target. On the 17 Freedoom development maps, 10 solve with doomLaya's own switch set and 12 with the full table (MAP06 and MAP28 added). On the first 10 Obsidian development maps, 8 solve.

Fixes found with the solver's diagnostics and the MAP04 runs:
- **Nested floor areas:** the navigator dropped a sector's floor faces nested inside another sector's ring, which made MAP20's start room an island. Faces are now also kept by the even-odd rule. Nothing changes on MAP01–10.
- **Crushers:** two crushing ceilings on MAP04 opened and closed every ~2 s once started. Each change rebuilt the navigator and cut the route to the blue key, so after the red key every objective read "not reachable" (run `ext1` seed 101). Crusher sectors now stay in the geometry.
- **Prerequisite chaining:** when a key or the exit is unreachable, a relaxed navigator with every mechanism applied finds the route there. The mechanisms on that route are named in their options as "On the way to: …".
- **One option per mechanism:** several lines often work one mechanism; the planner now offers one option per mechanism, and lift what-ifs start on the lift itself.
- **Weapon order:** the weapon head favours the first option. With fists listed first it chose them over a loaded pistol, the MAP02 regression in `regress1` (1/4, 12–14 deaths per run). In 150 replayed questions, listing melee last cut choices that cannot hit from 54 to 0.

### Weapon text back to "Ammo: N"; headroom in the navigator; MAP04 ext2 (2026-10-24)

**MAP02 regression found: the weapon wording, not the order.** With melee listed last (`wfix`), MAP02 still completed 1 of 6 at 10–11 deaths per run, against 8 of 10 in exec5. The run sources differed in the weapon option text: exec6 had replaced "Ammo: N" with "Shots left: N". In the failed wfix runs, 1,075 of 9,004 attack decisions chose the shotgun while its option read "Loaded: no … Shots left: 0", which never happened with "Ammo: 0". In 200 replayed MAP02 weapon questions, choices that cannot hit were 35 with the shots wording and 0 with "Ammo: N", with melee listed either first or last. So the fists choices in `regress1` also came from the wording; listing melee last is kept, as it is harmless. The weapon head was trained on "Ammo: N", which is the default again (`doomlib/policy.py`); `DOOM_WEAPON_TEXT_SHOTS=1` gives the exec6 wording for A/B runs. On MAP03 (exec6) the shots wording had not changed completions (3/10 in both arms).

| MAP02, `planner` | Completed | Deaths per run |
|---|---|---|
| exec5, seeds 54, 1–9 | 8/10 | 6, 4, 4, 2, 4, 11, 4, 1, 1, 2 |
| wfix (shots text, melee last), seeds 54, 1–5 | 1/6 | 3, 11, 10, 11, 11, 10 |
| ammofix (snapshot exp2), seeds 1–5 | 4/5 | 7, 0, 3, 2, 0 |

**MAP04 with extended mechanisms (`ext2`, exp2, seeds 101–103): 0/3**, deaths 7, 6, 0, all three hold the red key. Seeds 101 and 103 spent their last 5–15 minutes trapped in a lift cage: the navigator planned a drop from the cage floor (−32) into a pocket at −160 whose ceiling (−96) lies below the floor the player stands on, which in the engine is a wall. The bot shuttled between the cage walls trying to reach the blue key, a medikit and a rocket launcher beyond it. Seed 102 died in fights (chaingunners and imps near the red key area and at the start).

**Fix: headroom (dev patch).** A navigator step now needs 56 units between the lowest fixed ceiling the body overlaps and the higher of the two floors, as in the engine's `P_TryMove`. Doors, lifts and crushers have moving ceilings and are left out. A lift that lowers on the way (a walk-over line, or the legacy SR lift) is judged at its lowest floor; a use-only lift is judged at its current floor, because its switch may face the other side (MAP04's line 909). `mission.navigator_doors` builds this list for the executor and `doom/route_solver.py`. Route solver with the rule: Freedoom development maps unchanged at 12/17 (the same maps solve; MAP12 took 9,086 s), Obsidian `dev1` 8/10 unchanged.

### MAP04 route works end to end; combat is the blocker; faster experiments (2026-10-25)

**Route fixes (dev patch and planner).** A USE lift's switch line, not its platform, decides whether the planner can work it. A ride down ends when a lift boarded on the upper floor comes to rest at the bottom. A repeatable lift ridden up and stepped off at the top can be called again (the planner had chosen lift #463 to ride down to the blue key while the policy hid it as used, and Laya explored for 770 s). A key option lists every line locked to its colour. Health bonuses (+1) are errands only within 6 m: alternating between distant bonuses and floor switch #883, Laya reached neither for 20 minutes. With these, MAP04 runs collect both keys and open the exit route (switches #427 and #540); they die before the exit.

**Melee wording.** With "Loaded: yes … Ammo: 0" on melee options, Laya chose the chainsaw over a loaded shotgun once the chaingun ran dry (130 s against an imp on a ledge in MAP03). Melee options now state only range and reach. Replayed attack questions with an empty chaingun: 161 of 200 chose a weapon that cannot hit before, 2 after; on 200 other questions 0 before, 2 after. **Rocket safe distance** is 5.5 m (was 4 m): splash reaches 128 units past the target's edge. *Later:* this change came after the batches in the table below and the skill test; of the game batches it is in `cm1` only (and the scenarios and command rollouts).

| Batch (snapshot) | MAP02 | MAP03 | MAP04 | MAP04 deaths per run |
|---|---|---|---|---|
| regress2 (exp9), seeds 54, 1, 2 | 3/3 | 0/3 | | |
| lift4 (exp9), seeds 101–103 | | | 0/3 | 5, 8, 5 |
| melee1 (exp10), MAP02/03 seeds 54, 1, 2; MAP04 101–103 | 3/3 | 1/3 | 0/3 | 9, 1, 3 |
| lift5 (exp11), seeds 101–105 | | | 0/5 (blue key in 4) | 6, 5, 7, 6, 4 |
| fight1 (exp11, `--combat-facts`), MAP02/03 seeds 3, 4; MAP04 106–109 | 0/2 | 1/2 | 0/4 | 0, 4, 5, 7 |

MAP02 seeds 3 and 4 failed with `--combat-facts` (12 and 9 deaths), seeds that passed before without it; the extra facts in the movement question may hurt and are tested before training on them. *Later:* the outcome-trained movement head reads these facts; in games (`cm1`) it made no difference, and MAP02 was not re-tested with them.

**Deaths before the exit are survivable by another movement.** On 560 fight states from the 10 s before a death (fight1), each movement was replayed for 3 s from the saved state (`doom/fight_rollouts.py --before-death 10 --tics 105`). Laya's movement died in 179; in 85 of those another movement survived. Mean loss (health lost + 100 per death): Laya's choice 50.1, best of the four 25.7 (an upper bound, one deterministic stream); standing still is the worst (65.4, strafes and backward 48–52). A first pass was wrong: a load does not revive a dead player, so after a rollout death every later rollout of a run had died at once; the rollouts now start a new episode first. **Movement head trained on outcomes** (`training/finetune_combat_movement.py`, 7,056 rollout-labelled states, validation runs MAP04 seed 109 and MAP03 seed 4): validation regret 1.92 against 2.24 for the current head and 2.15 for the recorded choices. Config `combat_move1` plays it.

**Experiment speed.** Whole games in real time, one at a time, were the bottleneck. Now:
- `repeat_map.py --parallel N` plays N games at once, each against its own server. Core ML holds the ANE models of two processes at most (a third fails with "`.functionName` must be nil unless the model type is ML Program"), so servers beyond `--ane-servers` run the same model in PyTorch on the GPU: identical answers on 40 of 40 replayed requests, 77 ms median against 85 ms. Results record the server device. Seven games at once kept wall and game time equal (ratio 1.000). Reported results stay ANE-only. *Later:* not quite: two of the three skill-1 MAP04 wins (`skill1`, seeds 102 and 103) and the `fast1` batches ran on GPU servers, and the scenarios and command rollouts used GPU servers.
- `doom/scenarios.py` plays short games from saved states (`agent.py --load-state`, which ends at the first death) for several configs on the same states.
- `--stall-seconds N` (screening only) ends a run after N game seconds without a new 4 m cell, key or mechanism in the current life.
- `--run-navigation` (config `planner_ext_fast`): routes are followed running, as players keep always-run on; the audit allows SPEED only for navigation commands under that flag.

**Player guides** (`docs/research/Doom player guides/notes.md`, written this morning) turned generic player and engine knowledge into ten ranked proposals, each checked against the authority rule. Tried: P1, outcome-scored rollouts (begun for movement the day before; the movement head and the command rollouts below); P2, running everywhere (`fast1`, no change); and the rocket distance of P6 (5.5 m, from `cm1` on). Not tried before the park: hit-chance and attack-class facts for enemies (P3), a break-line-of-sight retreat (P4), health and armour effect facts (P5), the rest of P6 (switch delay, shots to kill), re-asking the movement question for close projectiles (P7), and door, awareness and infighting facts (P8–P10). P3 and P4 are among the open directions in `docs/WRITEUP.md`.

**Working method from here.**
1. *Offline first.* A change goes to a game batch only after it shows a gain offline: question text on replayed logged questions (`weapon_replay`-style), movement or tactics on rollouts or scenarios from saved states, planner changes on recorded objective timelines.
2. *Screen, then evaluate.* Screening batches run in parallel with `--stall-seconds 300`; a result we report comes from a frozen snapshot, ANE servers only, full time budget.
3. *Decide before the batch.* Each experiment states what result continues it and what result stops it:
   - Skill test (MAP04, skills 2 and 1, seeds 101–103): a completion at either skill means route and planning work and combat is the blocker; none at skill 1 means route problems remain.
   - Run everywhere (`fast1`): kept if MAP04 deaths per run and stalls are no worse than lift5 (deaths 6, 5, 7 on these seeds).
   - Movement head (`combat_move1`): scenarios on the validation runs' pre-death states first; it goes to games if deaths fall by at least a fifth with health lost no higher.
   - Overall: if by 2026-10-28 MAP04 has no completion at skill 3, the results are written up and the project is parked. *Later:* it was parked on 2026-10-25, after the command rollouts below.

**Skill test result: combat is the blocker.** MAP04, seeds 101–103, frozen exp11: skill 1 completed 3/3 with no deaths (508, 647 and 698 s); skill 2 0/3 with 3, 2 and 1 deaths; skill 3 (lift5) 0/3 with 6, 5 and 7. The route takes 500–700 s even without deaths, and a death restarts the map, so in 20 minutes a death after the keys almost always ends the run (skill 2 seed 101 opened switch #389, which opens the exit, at 1028 s and then died). MAP04 at skill 3 needs about zero deaths, not fewer.

**Run everywhere (`fast1`, exp12): no change.** MAP04 seeds 101–103: 0/3 with 3, 9 and 6 deaths (lift5: 6, 5, 7). Its servers ran on the GPU under load, at about 215 ms median decision latency against about 125 ms on the ANE, which may hide an effect.

**Outcome-trained movement head in scenarios.** 44 states from the 10 s before a death in the two validation runs, 20 s each, same states for both heads (exp13/exp15, GPU servers): deaths 18 → 13, health lost 20.4 → 15.0 per scenario, damage dealt 113 → 121. MAP04 12 → 8 deaths, MAP03 6 → 5. Of the 13 states where the outcomes differ, 9 favour the new head. The head's epoch was chosen on these runs, so the gain is optimistic. It meets the rule above and goes to games (`cm1`).

**Movement head in games (`cm1`, exp15): no change.** MAP04 seeds 101–105, skill 3: 0/5 with 9, 8, 4, 4 and 3 deaths, 28 in all, as with the current head (lift5: 6, 5, 7, 6, 4). The scenario gain (18 → 13 deaths over 44 states, 13 differing outcomes) did not carry into games: the scenarios start inside fights already being lost, while over a game the choice of which fights to take and when to break off is a command choice, not a movement. Run everywhere on MAP03 (seeds 101–103, no baseline on these seeds): 0/3 with 6, 6 and 3 deaths.

**Command rollouts (exp16): no command beats Laya's beyond the noise.** 209 fight states from the 25 s before a death (MAP02 seeds 3, 4; MAP03 seeds 3, 4; MAP04 seeds 107–109), the first command forced to each offered option, then 15 s of the unchanged stack (`doom/scenarios.py --commands`, `agent.py --force-first-command`; labels only, these runs fail the audit by design). Laya's own command: 71 deaths, mean loss 56.9. Forcing the same command again changes the death outcome in 43 of 207 states, so single rollouts are about a fifth noise. Best alternative `wait`: 57 deaths, loss 49.1, sign test on discordant deaths 34 vs 20 (p = 0.08); explore, attack, pickup and use_switch are level (p ≥ 0.5); retreat and look_back are worse. The table and conclusion are in `docs/WRITEUP.md`. Per the park rule, the results are written up there and MAP04 work stops.

**Open competitors zero-shot on Doom questions (`doom/opendecider_check.py`, separate venv with `opendecider` 0.9.0).** OpenDecider-nano (Ettin 400M encoder, Apache-2.0) and OpenDecider-small (4B decoder, MLX 4-bit) answered the same Doom questions with no Doom training. Movement: 1,362 outcome-labelled fight states of the validation runs (MAP04 seed 109, MAP03 seed 4); regret is the best movement's score minus the chosen one's (lower is better). Weapon: 300 logged attack weapon questions from `cm1` with at least one gun that can hit.

| Model | Movement regret | Weapon choices that cannot hit (of 300) | ms per question (this Mac) |
|---|---|---|---|
| Laya, outcome-trained movement head | 1.92 | – | – |
| Laya, doomLaya's Doom-trained heads | 2.24 (2.15 as played) | 6 | about 100 (ANE server, two questions) |
| OpenDecider-small 4B | 2.63 | 78 | about 170 (GPU) |
| OpenDecider-nano 400M | 2.74 | 95 | about 50 (GPU) |
| uniform random movement | 3.49 | – | – |

Both are better than random but well behind the Doom-trained Laya, and both choose a gun that cannot hit in about a quarter to a third of weapon questions (often melee or an empty gun). General decision-model quality does not stand in for Doom training; a swap would mean fine-tuning a competitor on the same data, for at best the small gains seen on the general benchmarks.

### Second held-out evaluation and a campaign, decided before the batch (2026-10-29)

After the park, two questions remained open: do the generic route rules work on maps they were not
shaped on, and can the final stack get past MAP04 at all? Both are played from one frozen snapshot
(`ext/snap/eval2`, the stack of this commit), ANE servers only, config `planner_ext_safe`, skill 3
unless stated. Nothing in these runs is used for fixes.

- **heldout2:** the six maps of heldout1 (MAP05, 07, 09, 11, 13, 15) on new seeds 204–206, 1200 s
  each, 18 games, against heldout1's 0/18 for both our earlier stack and clean v0.3.1. Any completion
  says the route rules carry over to unseen maps; 0/18 with more keys and doors per run than heldout1
  (8 keys, 496 doors) says they help but survival still decides; no gain says they do not transfer.
- **Campaign** (`repeat_map.py --campaign`): one game from MAP01 onward, 3600 s, each map from a pistol
  start (the engine starts a new episode per map, so inventory is not carried as in a real campaign).
  Skill 3, seeds 301–302: the expected end is MAP04, from 0/79. Skill 1, seeds 303–304: MAP04 went 3/3
  at skill 1, so this measures how far the stack gets past it; maps from MAP05 on are held-out maps.

### eval2 results: no held-out completion, MAP04 at skill 1 inside a campaign (2026-10-29)

Played as pre-registered from `ext/snap/eval2`, but on one ANE server, one game at a time: a second
server could no longer load all its models on this day (Neural Engine capacity taken by another
process), so 22 games took about 11 hours. Every game passed the authority audit.

**Campaigns** (`results/doom/repeat_campaign-map01_skill{3,1}.json`; time is game time at the exit,
deaths per map):

| Skill | Seed | Maps finished | Per map | Ended on |
|---|---|---|---|---|
| 3 | 301 | 2 | MAP01 192 s, 0 deaths; MAP02 2401 s, 12 deaths | MAP03, 11 deaths |
| 3 | 302 | 3 | MAP01 149 s, 0; MAP02 1307 s, 8; MAP03 1880 s, 3 | MAP04, 9 deaths |
| 1 | 303 | 4 | MAP01 107 s, 0; MAP02 515 s, 0; MAP03 829 s, 0; MAP04 1413 s, 0 | MAP05, 0 deaths |
| 1 | 304 | 2 | MAP01 162 s, 0; MAP02 623 s, 0 | MAP03, 0 deaths |

- Skill 3 ended where expected: MAP04 at best, reached once, with 9 deaths in the 1720 s left.
- Skill 1, seed 303 finished MAP04 inside a campaign on an ANE server, with no deaths on any map, then
  spent 2187 s on held-out MAP05 without finishing it. Seed 304 spent about 3000 s alive on MAP03
  without finishing it: with no deaths, what stopped it there was not combat.
- Every map started from a pistol start, so MAP02–05 were played without the weapons found earlier.

**heldout2** (`results/doom/repeat_map{05..15}_heldout2.json`; columns as in heldout1, cells are
4 m squares visited):

| Map | Valid | Deaths/run | Kills/run | Doors opened/run | Cells visited/run | Keys per run |
|---|---|---|---|---|---|---|
| MAP05 | 3/3 | 2.7 | 61 | 22.7 | 144 | blue+yellow, –, blue+yellow |
| MAP07 | 3/3 | 21.0 | 321 | 56.3 | 127 | red, –, – |
| MAP09 | 3/3 | 2.3 | 59 | 27.3 | 77 | blue, –, – |
| MAP11 | **0/3** | 23.7 | 278 | 27.7 | 192 | – |
| MAP13 | 3/3 | 27.7 | 402 | 28.7 | 128 | –, –, blue |
| MAP15 | **0/3** | 2.3 | 20 | 7.0 | 38 | – |

- **No completion: 0/18**, and 0/12 among valid games. Over all 18 games there were 7 keys and 509
  doors, against 8 keys and 496 doors for the heldout1 stack. By the pre-registered rule this is **no
  gain: the generic route rules did not transfer** to these maps in a measurable way. MAP04 shaped them,
  and on these six maps they gave no measurable help.
- **MAP11 and MAP15 did not sustain real time** (wall 1943–2098 s and 1555–1721 s for 1200 s of game),
  so those six games are invalid and the two maps have no result. In heldout1 the stack of 2026-10-23
  kept real time on MAP11 (3/3 valid) and on MAP15 in 2 of 3 games, so the later stack is slower on
  some maps. Server latency was normal (median about 130 ms), so the time goes in the agent loop. A
  cause can be looked for only on development maps; these games are not used for it.
- On MAP09, seeds 205 and 206 ended with 7 kills and no deaths. These runs are not inspected.

So after eval2 the summary stands: the stack completes MAP01–03 most of the time and MAP04 only at
skill 1, and nothing it learned on the development maps finishes a held-out map.

### Inventory carry-over for campaigns (development, 2026-10-29)

ViZDoom ends the episode at the exit, and `new_episode()` starts a new game, so every map of an eval2
campaign began from a pistol start. With `agent.py --carry-inventory` a small ACS library
(`assets/carry_inventory.acs`, built with the pinned ACC by `tools/build_weapon_sensor.py --target
carry_inventory`) publishes the inventory every tic. The harness keeps the last values from while the
player was alive and hands them to the next map in server CVARs, and the library restores them on
entering it. As in vanilla Doom this covers weapons, ammo, backpack, health, armour (points and type)
and the selected weapon, not keys or powerups; a death still restarts the map from a pistol start. The
harness then compares every field. An `inventory_carried` event records the result, and a mismatch
stops the run. This is the game's rules, not a decision: nothing in the model's questions changes.

- `tools/check_carry_inventory.py` (engine only, no model): 7 inventories carried exactly from MAP01
  to MAP02. They cover green armour below 100 and over 100 with bonuses, blue armour, the backpack,
  soulsphere and megasphere, ammo doubled on skills 1 and 5, and a selected chaingun. A restart after
  each one is a clean pistol start.
- In games (MAP01, `planner_ext_safe`, ANE, suffix `carrytest`): seeds 54 and 2 finished MAP01 and
  entered MAP02 with their inventory, with no mismatch (110 health, 103 armour and the shotgun selected
  for seed 54). Seed 1 stalled on MAP01 and never reached the exit. All three games are valid and pass
  the authority audit.

### eval3: campaigns with carry-over, decided before the batch (2026-10-29)

The eval2 campaigns again, with `--carry-inventory` the only change: same snapshot code otherwise
(`ext/snap/eval3`, the stack of this commit), `planner_ext_safe`, one ANE server, 3600 s from MAP01,
skill 3 on seeds 301–302 and skill 1 on seeds 303–304 (`results/doom/repeat_campaign-map01_carry-skill{3,1}.json`).
Maps from MAP05 on are held-out maps, and nothing in these runs is used for fixes.

- eval2 reference, maps finished: skill 3 2 and 3 (5 in all, MAP02 took 1158–2209 s with 8–12 deaths);
  skill 1 4 and 2 (6 in all).
- Seeds give a different game as soon as one decision differs, so this is a paired look at 2 seeds per
  skill, not a test. Read it as: **carry-over helps** if both skills finish more maps in all than in eval2
  and the skill-3 deaths per finished map fall; **no difference** if the totals are within one map;
  anything else is reported as it comes out.
- The headline would be a skill-3 MAP04 completion, which has not happened in 79 single-map runs and 2
  campaigns.

### eval3 results: mixed, within the noise of 2 seeds (2026-10-30)

All four games are valid, pass the authority audit and carried the inventory at every exit with no
mismatch (`results/doom/repeat_campaign-map01_carry-skill{3,1}.json`).

| Skill | Seed | eval2 (pistol start), maps finished | eval3 (carry-over), maps finished | eval3 per map | Ended on |
|---|---|---|---|---|---|
| 3 | 301 | 2 | 2 | MAP01 258 s, 0 deaths; MAP02 3337 s, 20 | MAP03, 2 deaths |
| 3 | 302 | 3 | 1 | MAP01 161 s, 0 | MAP02, 24 deaths |
| 1 | 303 | 4 | 4 | MAP01 110 s, 0; MAP02 542 s, 0; MAP03 986 s, 0; MAP04 1973 s, 0 | MAP05, 0 deaths |
| 1 | 304 | 2 | 4 | MAP01 155 s, 0; MAP02 505 s, 0; MAP03 856 s, 0; MAP04 1874 s, 0 | MAP05, 0 deaths |

- By the pre-registered reading this is neither "helps" nor "no difference": skill 3 finished 3 maps
  against 5, skill 1 finished 8 against 6.
- **Skill 3:** carry-over did not help where the bot dies. A death restarts the map from a pistol start,
  so the carried inventory lasts only for the first life on each map. On MAP02 that life ended early in
  both games (20 and 24 deaths, against 12 and 8 in eval2).
- **Skill 1:** both games finished MAP01–04 with no deaths and reached held-out MAP05; seed 304 got past
  the MAP03 stall of eval2. Without deaths the carried weapons stay for the whole campaign, but a single
  differing decision also changes a game, so one seed going from stuck to through is not evidence on its own.
- No skill-3 MAP04 completion: none in 79 single-map runs and 4 campaigns.

### eval4: a no-model baseline, and the held-out maps at skill 1, decided before the batch (2026-10-30)

Two claims in the README are argued rather than measured: that most of the competence is in the code
around the model, and that the route rules do not transfer to held-out maps. Both are played from one
frozen snapshot (`ext/snap/eval4`, the stack of this commit), one ANE server, one game at a time.

- **Random baseline** (config `random_ext_safe`): `planner_ext_safe` with every choice answer (command,
  targets, weapon, movement, the planner's objective) replaced by a uniform random option
  (`ane_doomlaya.py --random-chooser 0`). The model still runs, so decisions take as long as Laya's,
  and Laya's answer stays in the record. Played with `planner_ext_safe` itself in the same session:
  MAP01, MAP02, MAP03, seeds 54, 1, 2, 3, skill 3, the usual budgets (180, 1800, 1200 s), 12 games each.
  The random baseline passes the authority audit by construction (the server's answer is what is
  executed); its completions are a baseline, never a result for the model.
  - **The model matters** if Laya finishes at least 4 more of the 12 than the random baseline.
  - **The competence is in the code** if the random baseline finishes within 2 of Laya.
  - In between, report it as it comes out.
- **heldout3** (held-out maps at skill 1): MAP05, 07, 09, 11, 13, 15, seeds 204–206 as in heldout2, 1200 s,
  `planner_ext_safe --skill 1`. Skill 1 took the deaths out of MAP01–04 in the campaigns, so this separates
  the route from survival on unseen maps.
  - Any completion: the route rules carry over at least in part once survival is taken out.
  - 0/18 with few deaths (under 3 per run): the route, not combat, is the limit on unseen maps.
  - 0/18 with many deaths: combat decides even at skill 1.

### eval4 results: the model matters; on unseen maps the route is the limit, with one combat map (2026-10-30)

**Random baseline** (`results/doom/repeat_map0{1,2,3}_baseline.json`). Every game kept real time
(wall time within 1.1 s of game time).

| Map | Laya finished | Laya deaths per seed (54, 1, 2, 3) | Random finished | Random deaths per seed |
|---|---|---|---|---|
| MAP01 (180 s) | 3/4 | 0, 0, 0, 0 | 0/4 | 2, 3, 1, 1 |
| MAP02 (1800 s) | 2/4 | 10, 3, 1, 0 | 0/4 | 59, 63, 82, 69 |
| MAP03 (1200 s) | 2/4 | 9, 2, 5, 15 | 0/4 | 25, 22, 23, 20 |
| All | **7/12** | 45 | **0/12** | 370 |

- By the pre-registered reading: **the model matters**. Laya finished 7 more of the 12 than random
  choices, against a threshold of 4. The executor, reflexes and planner client on their own finish
  nothing, not even MAP01, and die eight times as often.
- The pre-registration said the random baseline would pass the authority audit by construction. It does
  not: `tools/check_authority.py` compares the executed command mixture with the model's own output and
  flags every replaced decision ("Command mixture probabilities replaced"), so all 12 random games are
  marked invalid. That is the audit doing its job (an executor that overrides the model fails it), not a
  defect of these games. Laya's 12 games are valid and pass the audit.

**heldout3: held-out maps at skill 1** (`results/doom/repeat_map{05,07,09,11,13,15}_heldout3.json`).
0/18 maps finished. All 18 pass the authority audit.

| Map | Finished | Deaths per seed (204–206) | Kills | Keys (telemetry) | Doors | Wall time, s (1200 s of game) |
|---|---|---|---|---|---|---|
| MAP05 | 0/3 | 0, 0, 0 | 34, 32, 34 | blue+yellow ×3 | 104 | 1262, 1200, 1201 |
| MAP07 | 0/3 | 3, 1, 0 | 128, 99, 81 | blue; blue+red; blue+red | 159 | 1200 ×3 |
| MAP09 | 0/3 | 0, 2, 3 | 5, 58, 94 | –; blue; blue | 127 | 1200 ×3 |
| MAP11 | 0/3 | 11, 13, 12 | 286, 261, 246 | – | 82 | 1789, 1805, 1741 |
| MAP13 | 0/3 | 10, 9, 7 | 198, 180, 188 | –; –; blue | 31 | 1200 ×3 |
| MAP15 | 0/3 | 0, 2, 0 | 1, 29, 4 | – | 9 | 1456, 1347, 1389 |

- Valid games: 11 of 18. MAP11 and MAP15 lost real time again, as at skill 3. The median decision
  latency there (133–154 ms) is in the range of the valid maps (100–174 ms), so the time is lost in the
  game loop on those maps, not in the model; not investigated. MAP05 seed 204 ran 5.1 % slow and is just
  outside the rule.
- Against heldout2 (skill 3, same seeds): 14 keys against 7, 547 doors against 509, no completion in
  either.
- By the pre-registered reading this is mostly **the route is the limit**: on MAP05, MAP07 and MAP09
  (9 games, 5 of them with no deaths and none with more than 3) Laya survived the whole 20 minutes,
  picked up keys in 8 of 9 games and still never found the exit. MAP13 is the exception: 7–10 deaths
  per game even at skill 1, so there combat still decides. MAP11 (11–13 deaths) and MAP15 are invalid
  and are not read.
- Taking the deaths out doubled the keys collected but finished no map. The route rules learned on
  MAP01–04 do not carry over to unseen layouts, and survival was not what hid it.

### pix1: pixel perception, level 1 (pre-registered 2026-10-31)

Question: what does it cost to find enemies and items from the screen instead of the engine? Until now
every enemy and item came from ViZDoom's labels buffer: only what is on screen, but with exact names,
ids, positions and outlines. Level 1 replaces that with a model that sees only pixels
(`doomlib/pixel_perception.py`, agent flag `--pixel-perception`). Everything else stays as it is,
including the map, doors, switches and keys known from the WAD and the object list, the counters, and
projectile facts. So this measures the cost of perception alone, not a human-fair game.

- **Data** (`doom/vision/collect_frames.py`): 250 frames from each of the 17 development maps, warped to
  random points (often near a monster or item), `god` on, random weapon, a few tics of motion and fire.
  The ground truth is the labels buffer. No hosted model is involved, and no held-out map.
- **Model** (`doom/vision/train_seg.py`): it names each 4x4 cell of the view (48x80 cells) with a monster
  or item class. Two encoders get the same decoder: MobileCLIP-S2's image trunk (fine-tuned), and a
  small CNN from scratch. Validation maps are MAP01–04, which are never in training, so the perception
  plays the game comparison on maps it has not seen. Exported to Core ML (`doom/vision/export_seg.py`)
  and run every tic on the Neural Engine.
- **Runtime:** objects are groups of touching cells of one class at a similar depth. Bearing comes from the
  column. Distance comes from the depth buffer along that ray, times 1.07, the median true/ray ratio for
  monsters measured on MAP02. Ids come from a position tracker.
- **Choice of encoder, before any game:** the one with the higher monster object recall on MAP01–04
  plays. If they are within 2 points, the faster one plays.
- **Game comparison:** `planner_ext_safe` with and without `--pixel-perception`, MAP01–03, seeds 54, 1,
  2, 3, skill 3, budgets 180/1800/1200 s, both in the same session from the same stack.
  - **Perception costs little** if pixel finishes within 1 of engine (of 12) and deaths are at most 1.5x.
  - **Perception costs a lot** if pixel finishes 3 or more fewer.
  - In between, report it as it comes out. Games that break the 5 % real-time rule are not read.

**pix1 offline results** (`results/vision/{tiny,mobileclip}/report.json`, `results/vision/recall_by_distance.json`).
Trained 20 epochs on the 13 other development maps, scored on MAP01–04 (1,000 frames). 43 classes: 29
names and the "Other" class of each category.

| Monster found, by distance | 0–5 m | 5–10 m | 10–20 m | > 20 m | False monsters per frame |
|---|---|---|---|---|---|
| MobileCLIP-S2 trunk, fine-tuned | 261/290 (90 %) | 399/454 (88 %) | 366/469 (78 %) | 59/163 (36 %) | 0.33 |
| Small CNN from scratch | 256/290 (88 %) | 340/454 (75 %) | 223/469 (48 %) | 28/163 (17 %) | 0.96 |

- Monster object recall: MobileCLIP 71.6 % against 54.5 %, so by the rule above **MobileCLIP plays**. It
  is ahead for every category and distance. Pretraining matters most at range.
- MobileCLIP, when an object is found: its name is right for 76 % of monsters, 90 % of health, 83 % of
  ammo, 80 % of weapons and 100 % of keys. Its median column error is 0.08°.
- Items are harder than monsters, beyond 10 m most of all: health 41 %, armour 37 %, ammo 32 % and
  weapons 24 % at 10–20 m.
- Core ML: 2.07 ms per frame on the Neural Engine (5.5 ms GPU, 15 ms CPU). It agrees with torch on
  99.93 % of cells. In the agent it takes 2.6 ms per tic, every tic.
- One change before any game, made because of the false-object rate above: an object is reported only
  once its track has been seen on 2 consecutive tics. Weapons and keys the model cannot name (the
  "Other" class) are not reported, because the server needs a weapon's slot (found in a smoke test,
  where an "OtherWeapon" item made the server refuse the packet).

**pix1 game results** (`results/doom/repeat_map0{1,2,3}_pix1_{engine,pixel}.json`, logs in
`results/vision/pix1_{engine,pixel}.log`). The two settings were played one after the other, not side by
side: two Laya servers on the Neural Engine at once fail to load ("`.functionName` must be nil unless the
model type is ML Program"). Engine first (10:00–13:00), pixel after (13:01–15:46), same stack and server.

| Map | Engine: finished | Engine: deaths per game | Pixel: finished | Pixel: deaths per game |
|---|---|---|---|---|
| MAP01 | 3/4 | 0, 0, 0, 0 | 0/4 | 0, 0, 0, 0 (2 kills each) |
| MAP02 | 1/4 | 3, 12, 12, 11 | 0/1 | 17 (seeds 1–3 not played, see below) |
| MAP03 | 2/4 | 10, 3, 6, 3 | 0/4 | 0, 0, 0, 0 (0 kills each; seed 3 invalid for real time) |

- All 12 engine games and 8 of the 9 pixel games were valid and passed the authority audit.
- **Reading: perception costs a lot.** Pixel finished 0 of 9 against 6 of 12. Even if the three unplayed
  MAP02 games had all finished, pixel would be 3 behind, so the reading does not depend on them.
- **Why: phantom monsters that never go away.** On MAP03 the model reports a Zombieman, a Cacodemon,
  Imps and ChaingunGuys around the start area, where the engine has none. Laya does what it is told the
  situation is: `attack` in 2,129 of 2,335 decisions, almost always with near-certainty, until the ammo is
  gone (`hitcount` stays 0), and it never leaves the start area. MAP01 is the same after 2 kills. The
  games on one map are near copies across seeds (MAP01: 2 kills and about 350 decisions each; MAP03:
  0 kills and 2,335 decisions), so the phantoms come from fixed views, not noise. The 2-tic confirmation
  cannot help with a misread texture or decoration, which looks the same on every tic.
- **The crash:** in MAP02 seed 1 the server refused the planner's `objective` question with HTTP 500,
  "question 'objective' options exceed head_max_len=384". The planner writes the known enemies into each
  option, and phantom enemies made the options too long. The same refusal is logged twice earlier in the
  pixel run and never in the engine run. By the rule, the error is not replaced by a heuristic, so the
  game failed and `repeat_map.py` stopped before MAP02 seeds 2 and 3 (MAP03 had been started in parallel
  by then and finished).
- **What the offline test missed:** it counted false objects per frame (0.33) over random views, and
  recall. In play, a missed monster costs little, but one phantom that stays at a fixed place locks the
  policy into attacking it. The offline score that matters is phantoms per view along the paths the
  agent actually walks.
- Pixel perception took 3.2 ms per tic (median, 4.5 ms p90) on the Neural Engine in these games.

### pix2: pixel perception without the phantoms (pre-registered 2026-10-31, after pix1)

Replaying positions from the stuck MAP03 game shows what the phantoms are: wall screens (a globe display
and a text panel) labelled as monsters, in almost every view of the start area. Two causes in pix1:
- **Loss weighting:** "nothing here" cells had the class-balanced weight clamped at 0.05, so a false
  monster cell cost the model a twentieth of a missed one.
- **Data:** three quarters of the frames were warped near a monster or item, and 60 % of empty frames
  were dropped, so the model saw little plain scenery.

Changes (all on the 13 training maps; MAP01–04 stay out of training):
- 250 more frames per training map from random points and facings, every frame kept
  (`collect_frames.py --scenery`, `ext/vision-data/scenery`).
- Background loss weight 1.0, the mean weight of the object classes (`train_seg.py --background-weight 1`).
  Same encoder (MobileCLIP-S2), same epochs, same runtime.
- A new offline test that looks like play: the views at 300 positions per map sampled from the telemetry of
  the four pix1 engine games on each of MAP01–03 (`collect_frames.py --from-runs`, `ext/vision-data/walk`).
  The score is the share of views with at least one false monster object.

Rules, set before training:
- **pix2 plays** if its walked-view false-monster share is at most a quarter of pix1's and its monster
  object recall on the pix1 validation frames stays within 10 points of pix1's 71.6 %. Otherwise the
  offline result is reported and no games are played.
- **Games:** the pix1 pixel comparison again (`planner_ext_safe --pixel-perception`, MAP01–03, seeds 54, 1,
  2, 3, skill 3, 180/1800/1200 s), against the pix1 engine games from the same stack. The same reading as
  pix1: costs little if within 1 finish of engine's 6 of 12 and deaths at most 1.5x; costs a lot if 3 or
  more fewer.

**pix2 offline result: does not play.** (`results/vision/mobileclip_pix2/`, `results/vision/walk_pix{1,2}.json`,
`results/vision/recall_by_distance_pix2.json`; trained on 7,500 frames, 59 minutes on MPS.)

| | pix1 | pix2 | Rule for pix2 |
|---|---|---|---|
| Walked views with a false monster (566 views, MAP01–03) | 13.3 % | 5.8 % | at most 3.3 % |
| Monster object recall, validation frames (MAP01–04) | 71.6 % | 50.5 % | at least 61.6 % |
| Monsters found at 10–20 m, walked views | 80 % | 47 % | |
| False monster objects per walked view | 0.186 | 0.170 | |

- Both conditions fail, so by the rule no games are played.
- Weighting empty cells the same as objects traded recall for precision (monster cell precision 0.84 to
  0.96), and lost most of the range: 10–20 m recall fell from 78 % to 44 %.
- Views with a false monster fell by more than half, but false monster objects per view hardly changed.
  The false objects that remain sit together in fewer views. Not yet checked: how many of them are real
  monsters below the 12-pixel size the evaluation counts, or corpses, rather than wall textures.
- Checking every false monster object on the walked views against the labels buffer: pix1 had 59 on bare
  walls and floors and 44 on other objects (corpses, barrels, lamps, armour bonuses); pix2 has 84 on bare
  scenery and 13 on objects. pix2 learned what corpses and barrels are, but not wall screens, which are not
  on the training maps. Any new map will bring textures the model has not seen, so more frames of the 13
  training maps are not a dependable fix.

### pix3: the depth buffer as a fourth input channel (pre-registered 2026-10-31, after pix2)

A monster or item is a sprite standing in front of whatever is behind it, so its outline is a step in
depth. A texture lies on its wall, so its depth runs on smoothly with the wall's. The runtime already
reads the depth buffer for distance (the player's sense of depth, not the engine's object list), so giving
it to the model keeps level 1's rule: monsters and items are found from what the player sees.

- **Data:** the three sets collected again, the same way, now with the depth buffer saved for every frame
  (`ext/vision-data/{frames_d,scenery_d,walk_d}`).
- **Model:** the MobileCLIP-S2 trunk on RGB, as before, plus a small CNN from scratch on depth. Their
  features are joined at each scale before the same decoder (`train_seg.py --depth`). The loss weights are
  pix1's (the class-balanced weight, at least 0.05, for empty cells), since pix2's weighting cost most of
  the range. Training frames are pix1's plus the scenery set. Same epochs.
- **Rules,** the same as pix2. pix3 plays if its share of walked views with a false monster is at most a
  quarter of pix1's (pix1 is scored again on the new walk set) and its monster object recall on the
  validation frames is within 10 points of pix1's (pix1 again scored on the same new frames). The games
  and the reading are the same as pix1's.

**pix3 offline result: does not play.** (`results/vision/mobileclip_pix3/`, `results/vision/eval_pix3_{walk_d,frames_d}.json`;
7,500 frames, 67 minutes on MPS. The re-collected sets came out frame for frame as before, now with depth,
so pix1 scores the same on them as on the old sets.)

| | pix1 | pix3 | Rule for pix3 |
|---|---|---|---|
| Walked views with a false monster (566 views, MAP01–03) | 13.3 % | 10.8 % | at most 3.3 % |
| Monster object recall, validation frames (MAP01–04) | 71.6 % | 77.3 % | at least 61.6 % |
| Monsters found at 10–20 m / beyond 20 m, walked views | 80 % / 38 % | 89 % / 55 % | |
| False monsters on bare walls and floors, walked views | 59 | 25 | |
| False monsters on other objects, walked views | 44 | 52 | |

- The false-monster condition fails, so by the rule no games are played.
- Depth did what it was for: phantoms on bare walls and floors fell from 59 to 25, and recall rose at
  every range (the largest gains at range: 10–20 m from 80 % to 89 %).
- What is left are mostly real sprites that are not monsters: explosive barrels (17), armour bonuses (11),
  corpses, columns, lamps, keys and ammo. They stand out in depth like monsters do, so depth cannot tell
  them apart. Barrels, corpses, columns and lamps are not target classes (they are trained only as
  "nothing"); armour bonuses, keys and ammo are targets given the monster category by mistake.

### pix4: classes for sprites that are neither monsters nor items (pre-registered 2026-10-31, after pix3)

pix3 with one change: corpses and gibs (Gore), barrels (Hazard), and decorations, vegetation and light
sources (Decoration) get one class each, category "Distractor", taken from the labels buffer like every
other class (`train_seg.py --distractors`). In pix1–3 they were trained as "nothing", and a sprite the model
had to call nothing was often called a monster. The runtime drops Distractor objects, so the stack never
sees them, and the metrics score them as nothing. Projectiles and puffs stay "nothing". Data, depth input,
loss weights, epochs and the rules are pix3's: pix4 plays if at most 3.3 % of walked views show a false
monster and monster object recall on the validation frames is at least 61.6 %.

**pix4 offline result: does not play.** (`results/vision/mobileclip_pix4/`, `results/vision/eval_pix4_{walk_d,frames_d}.json`.)

| | pix1 | pix3 | pix4 | Rule |
|---|---|---|---|---|
| Walked views with a false monster | 13.3 % | 10.8 % | 18.0 % | at most 3.3 % |
| Monster object recall, validation frames | 71.6 % | 77.3 % | 80.7 % | at least 61.6 % |
| Monsters found beyond 20 m, walked views | 38 % | 55 % | 64 % | |
| False monsters per walked view | 0.186 | 0.138 | 0.244 | |

- Recall is the best so far at every range, and phantoms are the worst. Naming the distractors did not stop
  barrels (25) and corpses (19) being called monsters on the walked views; bare-wall phantoms went back up
  from 25 to 60.
- A likely reason, not tested: the extra classes change the class-balanced weights (normalised by the mean
  object-class weight), so the empty class weighs relatively less and the model calls more things. pix1–4
  look like points on one recall/phantom trade-off rather than fixes; the next model should be compared at
  a matched operating point (backlog item 8) and judged on persistent phantoms (backlog item 1),
  `docs/PIXEL_BACKLOG.md`.

### Matched operating point and phantom persistence (backlog items 1 and 8, 2026-11-01)

`doom/vision/operating_point.py`, `doom/vision/collect_sequences.py`, `results/vision/operating_point.json`.
Each model's monster threshold (summed monster-class probability) is set so its monster object recall on
MAP04 (validation, not a game map) is 78.3 %, pix1's at its argmax. Sequences: 6-second stretches every
3 minutes of the twelve pix1 engine games (every 2nd tic; 6,628 frames), and, as a check that the test
sees the failure, stretches of the stuck pix1 pixel games (MAP03 seed 1 every 2 minutes, MAP01 seed 54
every 30 s). A phantom track is a false monster group followed across consecutive frames; 18 frames is
about a second.

| At MAP04 recall 78.3 % | pix1 | pix2 | pix3 | pix4 |
|---|---|---|---|---|
| Threshold | 0.50 | 0.07 | 0.79 | 0.64 |
| Monster recall on walked views, MAP01 / 02 / 03 | 69 / 73 / 89 % | 68 / 66 / 84 % | 74 / 74 / 93 % | 78 / 81 / 95 % |
| Walked views with a false monster | 13.3 % | 17.8 % | **5.1 %** | 9.7 % |
| Engine paths: phantom tracks of 2+ / 9+ frames, longest | 223 / 4, 12 | 1,043 / 13, 18 | **71 / 4, 11** | 116 / 4, 21 |
| Stuck MAP03 path: tracks of 2+ / 9+ / 18+ frames, longest | 194 / 22 / 2, 40 | 230 / 14 / 2, 25 | **13 / 0 / 0, 6** | 64 / 6 / 0, 17 |

- The test sees pix1's MAP03 failure: on the path where it stalled, phantoms last up to 40 frames (80 tics,
  over 2 s); on the engine paths, at most 12.
- At the same recall pix3 is clearly the best: a third of pix1's walked-view phantoms and almost none on the
  MAP03 path. The earlier single-threshold comparison hid this; pix2 and pix4 sat at other points of the
  trade-off. (pix3 at 0.79 is still above the 3.3 % of the pix3 rule, which stands as recorded.)
- **MAP01 failed for a different reason.** On pix1's stuck MAP01 path no phantom lasts more than 3 frames,
  yet the stack listed 23 enemies at 15 s, among them Arachnotrons and a Pain Elemental, and Laya fired
  its ammo at 2 kills and spent 247 of 351 decisions on pickups that went nowhere. In pixel mode an enemy is
  remembered for 70 tics after its last sighting and nothing else removes it (the engine check is off),
  and the tracker starts a new id whenever a detection's position jumps by more than 64 units plus 15 % of
  its distance. Brief phantoms and duplicates of real monsters therefore pile up in memory. A frame test
  cannot see this; it needs the runtime itself on sequences (pixel enemies against engine objects over time).

### Runtime replay and forgetting on negative evidence (backlog items 1b, 1c and 10, 2026-11-01)

`doom/vision/runtime_replay.py` feeds view sequences through doomLaya's own `PixelPerception` and enemy
memory (`Sensors.remember_enemies`), tic by tic as in a game, and compares what the stack is told with the
engine's monsters. A listed enemy farther than 96 units plus 20 % of its distance from every monster is a
phantom; a second entry on the same monster is a duplicate. Sequences: `collect_sequences.py --stride 1`
(every tic; the runtime confirms and forgets by tic, so the stride-2 set of the previous section cannot be
used here, and a first run on it listed nothing at all), same windows as before, in `ext/vision-data/seq_tic`.

**Item 10 found on the way.** The pix1 MAP02 crash (`question 'objective' options exceed head_max_len=384`)
had 129 objective options, almost all "Collect the Shotgun (#100xxxx)": pixel tracker ids. The executor's
item memory drops an item only when the engine reports its object removed, and tracker ids are never in
that list, so every pixel item, phantom, duplicate or already collected, stayed for the rest of the game.
This is also the likely cause of pix1's 247 pickup decisions that went nowhere on MAP01.

**Change (1c, perception only; the model still chooses):** in pixel mode a remembered enemy whose spot is in
clear view (within 40° of the view centre and 15 m, nearer than the wall along that bearing in the depth
buffer) and not detected on 3 tics is forgotten. Items get the same rule within 10 m (items are found
24–41 % of the time beyond 10 m), and an item is also forgotten when the player stands on its spot; the
forgotten item ids are reported through `removed_object_ids`, the list the executor already uses
(`agent.py`, `Sensors.forget_items`; `tests/test_pixel_memory.py`).

Rule, set before the replays with it: the change is kept if, for pix3 at 0.79 on the engine paths, it removes
at least half of the phantom and duplicate entries per frame while keeping at least 90 % of the real
monsters listed per frame.

### Short closed-loop gate (backlog item 2, pre-registered 2026-11-01)

Before any 12-game comparison, a model plays two short games with `planner_ext_safe` and
`--pixel-perception`, seed 54, skill 3: MAP01 for its usual 180 s, and MAP03 for 300 s
(`repeat_map.py --seconds 300`). First candidate: pix3 at threshold 0.79 (`results/vision/mobileclip_pix3`),
with the forgetting rules above if they pass their rule.

- **Passes** if MAP01 is finished, or MAP03 reaches a new region beyond the start area (the pix1 games never
  left it) with at least one kill, and neither game crashes or breaks the 5 % real-time rule.
- **Passes**, the full comparison follows: MAP01–03, seeds 54, 1, 2, 3, against the pix1 engine games
  (6/12), with pix1's reading (costs little: within 1 of 12 and deaths at most 1.5x; costs a lot: 3 or more
  fewer).
- **Fails**: no full games for that model; the next backlog item.

**Results** (`results/vision/runtime_replay_*.json`; per frame, means over the sequences; `_v1` = the grouping
used in the pix1 games). Entries listed = seen now plus remembered.

| | pix1 | pix1 + 1c | pix3 v1 | pix3 v2 | pix3 v2 + 1c |
|---|---|---|---|---|---|
| Engine MAP02: phantom / duplicate / real monsters listed | 14.3 / 23.1 / 3.9 | 12.5 / 19.8 / 3.7 | 6.5 / 15.9 / 3.3 | 5.2 / 10.2 / 3.3 | 4.4 / 8.6 / 3.1 |
| Engine MAP02: most entries at once; items remembered (most) | 254; 28 (100) | 250; 20 (65) | 175; 18 (75) | 106; 24 (119) | 102; 19 (80) |
| Engine MAP03: phantom / duplicate / real | 10.7 / 8.9 / 1.9 | 8.1 / 6.8 / 1.8 | 4.1 / 5.2 / 1.6 | 3.9 / 5.0 / 1.6 | 2.6 / 4.0 / 1.5 |
| Stuck MAP01 path: phantom / duplicate / real | 21.0 / 6.8 / 0.9 | 16.4 / 4.6 / 0.9 | 10.6 / 6.6 / 1.0 | 10.5 / 5.5 / 1.0 | 7.5 / 4.2 / 0.9 |
| Stuck MAP03 path: phantom / duplicate / real | 6.0 / 1.8 / 0.9 | 4.2 / 1.8 / 0.9 | 0.6 / 0 / 0.1 | 0.9 / 0 / 0.1 | 0.5 / 0 / 0.1 |

- **Grouping (v2), found while reading the most crowded frame:** in one MAP02 frame with 5 Demons within
  6 m the runtime reported 38 monster objects: cells were grouped by class (one Demon's cells were labelled
  Demon, Zombieman, Revenant, OtherMonster), and a near sprite came out in up to 8 pieces. Each piece takes
  its own track id. v2 groups touching cells by category (named by the most common class), merges pieces at
  one depth less than 32 units apart across the view, and splits at 0.1 m or 3 % of depth instead of 1.5 m
  or 15 % (a sprite lies at one depth in the buffer, whose step is 0.25 m; two Demons one step apart were
  being joined). That frame: 11 objects. Over the sequences, duplicates fall a third on MAP02 and little
  elsewhere, and more item entries are remembered.
- **1c fails its rule**: on the engine paths it removes 16 % (MAP02) and 25 % (MAP03) of phantom plus
  duplicate entries, not half; real monsters kept 95 % and 93 %.
- What remains is memory: phantoms in view are 0.3–0.5 per frame for pix3, the rest of the 10–15 extra
  entries are remembered ones, mostly the same monster or item under an earlier id (an id stays 70 tics
  after the tracker starts a new one), and most of them are outside the 40°/15 m region where 1c can see them.

**1d, re-identification (pre-registered 2026-11-01, before running it):** in pixel mode a remembered enemy
not seen now is dropped when a seen enemy lies within 128 units plus 25 % of its distance (twice the
tracker's gate): the seen one explains it. The same for items, among items of the same name. Kept, together
with 1c, if on the engine paths phantom plus duplicate entries per frame fall to at most half of pix3 v2
without forgetting (MAP02 ≤ 7.7, MAP03 ≤ 4.4), real monsters listed stay at ≥ 90 %, and the most items
remembered falls to at most half (MAP02 ≤ 60, MAP03 ≤ 42).

**1d result: fails** (`runtime_replay_mobileclip_pix3_ne.json` at this point, 1c + 1d; the 1c-only run is
`_ne_1c`). Engine paths, phantom plus duplicate entries per frame: MAP02 15.4 → 3.3, MAP03 8.9 → 2.2 (rule
≤ 7.7 / 4.4, met); most items remembered 119 → 45 and 84 → 31 (≤ 60 / 42, met); but real monsters listed
3.31 → 2.07 (63 %) and 1.64 → 0.92 (56 %), against ≥ 90 %. A remembered monster is dropped whenever another
monster is seen near it, and on MAP02 monsters come in groups.

**1e (pre-registered 2026-11-02, before running it):** as 1d, but an enemy is re-identified only by a seen
enemy of the same name (items already require it). Same rule as 1d, against pix3 v2 without forgetting.

**1e result: fails** (`runtime_replay_mobileclip_pix3_ne.json`, 1c + 1e). Phantom plus duplicate entries:
MAP02 15.4 → 4.3, MAP03 8.9 → 2.7 (met); most items 119 → 45, 84 → 31 (met); real monsters listed
3.31 → 2.36 (71 %) and 1.64 → 1.09 (66 %) (not met).

Diagnostic (MAP03 engine paths, `real_monsters_seen_per_frame`: distinct true monsters matched by an entry
seen on that tic): 0.44 with and without forgetting. Forgetting never removes a monster in view; all of the
loss is among remembered entries (1.20 → 0.65 per frame), and some of the 1.20 may be chance matches by the
junk entries, which the replay cannot separate.

**Gate amended before any gate game (2026-11-02).** The offline rules cannot settle the trade (70 % fewer
junk entries against about half of the remembered real monsters), so the gate plays both, one after the
other, from the same stack: **A** pix3 v2 with `--pixel-memory-plain` (no forgetting: memory window only,
items never forgotten), **B** pix3 v2 with 1c + 1e (failed its offline rule; recorded as such). Each passes
or fails by the gate rule above. If both pass, the full comparison uses the one with fewer deaths over the
two gate games, B on a tie. MAP03 "a new region beyond the start area" is made exact before any gate game:
at least 24 of `tools/check_navigation.py`'s regions (256 x 256 x 64 units) visited in the 300 s, twice
pix1's (the four pix1 pixel games: 11–12 regions and 0 kills in their first 300 s; the four engine games:
47–81 regions and 12–39 kills), and at least one kill.

**Gate results (2026-11-02)** (`results/doom/repeat_map0{1,3}_gate_pix3_{plain,forget}.json`, log
`results/vision/gate_pix3.log`; seed 54, all four valid and passing authority):

| | MAP01 (180 s) | MAP03 (300 s): regions, kills, deaths | Most objective options | Gate |
|---|---|---|---|---|
| A, pix3 v2, no forgetting | not finished; 2 kills, 5 regions | 10, 0, 0 | 33 (MAP01), 22 (MAP03) | fails |
| B, pix3 v2 + 1c + 1e | not finished; 16 kills, 61 regions | 28, 50 (8+15+14+13 over four lives), 3 | 19, 4 | **passes** |

- A plays like pix1: on MAP03 its last eight objectives were all pixel items (`item_10000xx`), and it never
  left the start. B on MAP03 chose switches and the blue key and reached 28 regions; on MAP01 it fought (16
  kills) but did not reach the exit in 180 s, choosing the same switch and pixel items in turn.
- So the closed loop settles what the offline rule could not: forgetting junk entries matters far more than
  the remembered real monsters it loses. B goes to the full comparison as pre-registered (pix3 v2 + 1c + 1e,
  MAP01–03, seeds 54, 1, 2, 3, against the pix1 engine games' 6/12).

**pix3 + 1c + 1e, full comparison (2026-11-02)** (`results/doom/repeat_map0{1,2,3}_pix3_forget.json`, log `results/vision/pix3_forget.log`; seeds 54, 1, 2, 3; all 12 valid and passing authority): **0/12 finished**, against 6/12 for the engine games of pix1 (MAP01 3/4, MAP02 1/4, MAP03 2/4). By pix1's pre-registered rule this **costs a lot** (6 fewer).

| | MAP01 (180 s) | MAP02 (1800 s) | MAP03 (1200 s) |
|---|---|---|---|
| Finished | 0/4 | 0/4 | 0/4 |
| Kills per game | 13, 20, 22, 15 | 182, 165, 213, 187 | 178, 190, 185, 190 |
| Deaths per game | 0, 1, 1, 0 | 11, 5, 10, 10 | 8, 8, 6, 5 |
| Median decision latency | 133–139 ms | 137–142 ms | 137–148 ms |

Speed is not the cause: on MAP01 pixel and engine games made the same number of decisions (351 vs 345), at 133–139 ms against 124–158 ms for engine games (pix1 pixel: 218–256 ms). Combat is engine-like (pix1 pixel: 2 kills on MAP01, 0 on MAP03).

Planner choices (`runs/repeat-planner_ext_safe-{pix1_engine,pix3_forget}-map0*-planner.jsonl`):

| | Mean objective options | Item choices / all choices | Of which pixel tracker ids | Exit offered in any choice |
|---|---|---|---|---|
| Engine MAP01 | 4.1–4.6 | 2/36 | – | every game; chosen in 2 (112 s, 160 s) |
| Pixel MAP01 | 6.3–9.6 | 17/41 | 15 | every game; never chosen |
| Engine MAP02 | 3.3–4.1 | 46/229 | – | 1 game (finished) |
| Pixel MAP02 | 5.6–5.9 | 112/238 | 54 | none |
| Engine MAP03 | 3.2–3.7 | 67/230 | – | 2 games |
| Pixel MAP03 | 3.3–4.3 | 70/201 | 34 | none |

On MAP01 the engine games finished when the switches had been tried and the menu had drained to exit and explore; in the pixel games the menu kept refilling with detected weapons "not yet owned" (three Chainguns and a PlasmaRifle in one game) under new tracker ids, each "Not tried yet", and Laya preferred them to the exit 20–40 m away. On MAP02 and MAP03 no pixel game reached a point where the exit was offered; item objectives took about half the choices, against a fifth (MAP02) and under a third (MAP03) with engine perception.

Next candidate (not yet pre-registered): a detected item where a chased item already failed inherits that spot's history, so the menu does not offer it as "Not tried yet"; then stricter confirmation for item classes.

**Pre-registration: MAP01 with 600 s (2026-11-02).** Same stack and flags as the full comparison (pix3 + 1c + 1e), seeds 54, 1, 2, 3, `--seconds 600`, suffix `pix3_forget_600`. Question: is pixel play only slow, or does it not converge? Prediction: at most 1/4 finished, because new detections keep arriving as "Not tried yet" options. If 3/4 or more finish, the 180 s failure was mainly time, and the item fix is judged on time to finish rather than on completion. Diagnostic only; nothing is tuned on it.

**Pre-registration: item spots (2026-11-02).** Change in `doom/planner.py`, active only with `--pixel-perception`:
1. An item objective that ends with the item forgotten but nothing collected (weapon still not owned, health not higher than when chosen) counts as failed, not done.
2. Every failed item objective (that, "objective failed", or "no progress") leaves its spot: item name, point, tick.
3. A detected item of the same name within `SPOT_UNITS = 128` of a failed spot is offered with that spot's history ("History: chosen n times, failed n times.", the existing wording) instead of "Not tried yet", and the existing objective rule applies to the spot: two failures set it aside for `OBJECTIVE_SET_ASIDE_SECONDS`.
Laya still makes every choice. The executor is unchanged; the menu reports outcomes and no longer hides history behind new tracker ids.
Gate: MAP01, 180 s, seeds 54, 1, 2, 3, otherwise the same flags as the full comparison (suffix `pix3_spots`). Passes if ≥ 2/4 finish with deaths no higher than in the full comparison's MAP01 games (2 in total). If it passes, the full 12-game comparison follows under pix1's rule. If it fails, check whether it fired (failed item objectives and inherited histories in the planner logs) before changing anything.

**MAP01 with 600 s: result** (`results/doom/repeat_map01_pix3_forget_600.json`, log `results/vision/pix3_forget_600.log`; all valid and passing authority): **0/4**, as predicted. Kills 16, 16, 26, 16; deaths 0, 0, 1, 0; 1168–1172 decisions, median latency 119–131 ms. More time did not converge.
Planner logs: 30–47 objective choices per game, 9–20 of them items (engine games: 7–11 choices, 0–1 items). New finding: Laya chose the exit five times (seed 1 at 229 s and 419 s, seed 3 at 400 s and 526 s, seed 54 at 537 s), and each time the objective was replaced 5 s later (once 25 s) by "keys, doors or mechanisms changed". In the same games switches 839 and 198 are chosen and "done" over and over, which suggests doors closing again and changing the signature. The two engine exit choices (112 s, 160 s) held 14–15 s until the level ended. This interruption is planner behaviour, not perception; it shows up because the pixel games last longer. Not acted on before the item-spots gate, which was pre-registered first.

**Item spots gate: result** (`results/doom/repeat_map01_pix3_spots.json`, log `results/vision/pix3_spots.log`; change kept as `results/vision/planner_item_spots.diff` and reverted from `doom/planner.py`): **0/4, fails** (rule ≥ 2/4). Kills 16, 13, 13, 12; no deaths; all valid and passing authority. It fired: failed item objectives now show "objective failed", inherited histories appear on 1–2 options at a time, and spots were set aside. The menu still held 4–16 options, because new detections mostly land more than 128 units from earlier spots.
Why the switches fail (seed 1, 30–61 s, switch_198 the objective, ended "no progress in 30 s"): the executor was on `attack` for 853 of 1085 tics, and 732 of those were aimed at **remembered** enemies, not visible ones, most 16–19 m away. They were listed under shifting classes (id 1000230 as DoomImp, HellKnight and ChaingunGuy) and included HellKnights. Engine game, seed 2, during its switch_198 objective (61–83 s): `use_switch` on 646 of 770 tics, a visible enemy on 32 tics. So in pixel games Laya spends the switch and exit trips attacking remembered phantom enemies. Of the three problems found today (phantom remembered enemies, phantom items, door closings re-triggering the plan), this one costs the most time. Next candidate: stricter admission of pixel enemies to the memory (seen on several of the last few tics, not only on the tic before), measured first with `doom/vision/runtime_replay.py` (remembered phantoms against real monsters kept), then gated on MAP01.

**Pre-registration: memory admission for pixel enemies (2026-11-03).** A pixel enemy that is not visible stays in the enemy memory only if its track has been seen on at least `REMEMBER_SEEN_TICKS` tics in total (`PixelPerception` counts sightings per track). Visible enemies are listed as before. Engine perception is unchanged. Swept offline with `doom/vision/runtime_replay.py --negative-evidence --remember-seen K`, K in {0 (off), 4, 8, 12, 18}, on the `seq_tic` sequences. A new metric, remembered phantom + duplicate entries per frame, is added, and the baseline (K = 0) is rerun with it.
Offline rule, on engine_MAP02 and engine_MAP03: remembered phantom + duplicate entries per frame at most half of K = 0, and real monsters listed per frame at least 90 % of K = 0. The smallest K that passes on both maps goes forward. If none passes, the largest K that keeps real monsters ≥ 90 % is gated anyway, and the notebook records the offline fail.
Gate: MAP01, 180 s, seeds 54, 1, 2, 3, same flags as the full comparison (suffix `pix3_admit`). Passes if ≥ 2/4 finish with no more than 2 deaths in total. Diagnostic either way: share of tics on `attack` aimed at remembered enemies during switch and exit objectives (was 732 of 1085 in the seed 1 example).
*Added after the pre-registration, as diagnostics only (no change to the rule):* `real_remembered_per_frame`, and at K = 0 the share of real remembered entries whose track was seen on fewer than 4, 8, 12 or 18 tics (what each K would drop: brief real threats such as a monster stepping out from cover). The replay now caches PixelPerception's output per model and sequence (`ext/vision-data/replay_cache`), since it does not depend on the memory; the first sweep took about two hours per K and was stopped after K = 0.

**Memory admission sweep: result** (`results/vision/runtime_replay_pix3_ne_admit{0,4,8,12,18}.json`; the replay had been reading each frame out of the compressed archive, decompressing the whole array every time, which is why a sweep point took two hours; fixed, output identical, about 2 s per sequence now):

| K | Remembered phantom + duplicate per frame, MAP02 / MAP03 (share of K = 0) | Real monsters listed per frame, MAP02 / MAP03 (share) |
|---|---|---|
| 0 | 2.07 / 1.71 | 2.36 / 1.09 |
| 4 | 1.70 (82 %) / 1.27 (74 %) | 2.28 (97 %) / 0.99 (91 %) |
| 8 | 1.10 (53 %) / 0.69 (40 %) | 2.05 (87 %) / 0.84 (77 %) |
| 12 | 0.72 (35 %) / 0.35 (20 %) | 1.81 (77 %) / 0.70 (64 %) |
| 18 | 0.47 (23 %) / 0.15 (9 %) | 1.64 (69 %) / 0.58 (53 %) |

**No K passes the offline rule.** K = 8 comes closest (MAP03 passes the halving, MAP02 misses at 53 %) but keeps only 87 % / 77 % of real monsters. Diagnostic: at K = 0, 11 / 22 % of the real remembered entries (MAP02 / MAP03) come from tracks seen on fewer than 4 tics, 37 / 54 % on fewer than 8, 58 / 71 % on fewer than 12. Real monsters are often seen briefly, so track length separates phantoms from real monsters poorly. As pre-registered, the largest K keeping ≥ 90 %, **K = 4**, is gated (`REMEMBER_SEEN_TICKS = 4`).

**K = 4 gate: result** (`results/doom/repeat_map01_pix3_admit.json`, log `results/vision/pix3_admit.log`; all valid and passing authority): **0/4, fails.** Kills 14, 11, 10, 11; no deaths. Tics on `attack` aimed at a remembered (not visible) enemy, whole game: 1844, 1863, 2110, 1134 of 6300 (18–33 %), against 1903–2378 (30–38 %) for the full comparison's MAP01 games and 317–895 of 4536–6300 (5–20 %) for the engine games. A small reduction, as the replay predicted. `REMEMBER_SEEN_TICKS` set back to 0 (the mechanism stays, off).
**Class switching per track (offline, not pre-registered; diagnostic only).** On the cached detections of all five `seq_tic` sequences, each visible pixel enemy sighting was labelled real or not (phantom or duplicate) with the replay's matching, and grouped by track (script kept in the session scratchpad as `track_classes.py`). One-class tracks: 1344 tracks, 29 % of sightings real. Two classes: 500, 35 %. Three or more: 632, 35 %. Tracks whose most common class is under 70 % of sightings: 33 %, the rest 33 %. Class switching does not separate real tracks from false ones either. More broadly, only about a third of *confirmed, visible* enemy sightings match a distinct real monster; the rest are phantoms or duplicates (fragments of one monster). Track-level filters on the memory (length, class consistency) cannot fix that; the lever is the detection itself (precision, and grouping of fragments).

**Duplicates split (2026-11-03).** The replay now reports visible duplicates separately. At K = 0, visible phantom / visible duplicate / remembered phantom / remembered duplicate per frame: MAP02 0.48 / 1.73 / 0.90 / 1.16; MAP03 0.28 / 0.74 / 0.90 / 0.82. With a maximum one-to-one matching instead of nearest-monster matching (two close monsters can make a greedy "duplicate"), visible extras are 1.33 (MAP02), 0.64 (MAP03), 1.03 (pix1stuck MAP01) per frame (scratchpad `dup_check.py`). An extra lies close in bearing to the entry that took its monster (median 8.2°, 4.3°, 2.5°) but **about 1.6 m away in depth** (median 1.68, 1.54, 1.63 m): sprites are split by depth, since `near()` joins pieces only within 0.1 m or 3 %, and cell depth (25th percentile of the block) picks up floor and wall at a sprite's edges.

**Pre-registration: depth-tolerant merge.** In `PixelPerception.detect`, after the existing merges, pieces of the same category whose column ranges overlap or lie less than MERGE_UNITS apart (at the nearer depth), and whose depths differ by at most `MERGE_DEPTH_M`, become one object at the nearer piece's depth. Swept offline, `MERGE_DEPTH_M` in {0 (off), 0.5, 1, 2, 3}, on segmentation labels cached per frame, through the full replay (tracker, memory with 1c + 1e, K = 0).
Offline rule, on engine_MAP02 and engine_MAP03, with one-to-one matching: visible extras per frame at most half of off; real monsters seen per frame at least 95 % of off; visible phantoms per frame not higher than off. The smallest value that passes goes to the gate. If none passes, no gate: the result is recorded, and the next step is detector precision.
Gate: MAP01, 180 s, seeds 54, 1, 2, 3, as before (suffix `pix3_merge`); passes with ≥ 2/4 finished and ≤ 2 deaths. Diagnostic: tics on `attack` at remembered enemies (now 18–38 % of tics).

**Depth-tolerant merge sweep: result** (`results/vision/runtime_replay_pix3_merge{0,0.5,1,2,3}.json`; segmentation cached per frame in `ext/vision-data/replay_cache/mobileclip_pix3/*_cells.npy`, about 3 min per sweep point; the baseline reproduces the earlier K = 0 replay exactly). One-to-one matching, visible entries; share of off in brackets:

| MERGE_DEPTH_M | MAP02 extras / real seen / phantoms | MAP03 extras / real seen / phantoms | Remembered false + dup, MAP02 / MAP03 |
|---|---|---|---|
| 0 | 1.33 / 1.61 / 0.48 | 0.64 / 0.55 / 0.28 | 2.07 / 1.71 |
| 0.5 | 0.66 (50 %) / 1.55 (96 %) / 0.44 | 0.32 (50 %) / 0.54 (98 %) / 0.26 | 1.61 / 1.21 |
| 1 | 0.29 (22 %) / 1.43 (89 %) / 0.37 | 0.15 (23 %) / 0.50 (91 %) / 0.22 | 1.22 / 0.92 |
| 2 | 0.10 (8 %) / 1.20 (75 %) / 0.28 | 0.06 (9 %) / 0.42 (76 %) / 0.18 | 0.81 / 0.76 |
| 3 | 0.05 (4 %) / 1.02 (63 %) / 0.25 | 0.03 (5 %) / 0.37 (67 %) / 0.16 | 0.73 / 0.68 |

Unrounded at 0.5: MAP02 extras 0.6639 against half of off 0.6644 (passes), real seen 96.4 %; **MAP03 extras 0.3215 against 0.3182 (fails by 0.003)**, real seen 97.4 %; phantoms down on both. At 1, extras pass easily but real seen is 89 % / 91 % (fails). **No value passes the pre-registered rule; as registered, no gate.** The trade-off is steep beyond 0.5: merging more removes real monsters standing behind one another. The mechanism stays in the code, off.

**Next step after the merge sweep (2026-11-04).** As registered, the merge goes no further, and 0.5 is not gated by amendment. Before turning to detector precision, which is a larger job, the third cause on the list (door closings re-triggering the plan) gets its own pre-registration. It is a planner change, independent of perception.

**Pre-registration: re-plan only on gains (2026-11-04).** Evidence (planner logs, `pix3_admit` seed 3 and `pix1_engine` MAP01 seed 3, the one engine game of four that did not finish MAP01): a door switch is used ("objective done"), the model then picks the switch beyond that door, and 4.6 s later the door closes again. The closing changes the signature (a remote door is closed again, and a repeatable switch drops out of the activated set), which re-plans with "keys, doors or mechanisms changed". The model picks the door switch again, and the two switches alternate until time runs out (seed 3: six times in 60 s).
Change, in `doom/planner.py`: a changed signature re-plans only if it **gains** something over the signature the objective was chosen under, meaning a key held, a switch activated, or a remote door open that was not before. Losses alone (a door closing again, a switch dropping out) no longer re-plan. The progress check after 30 s and the done and failed rules stay as they are. The model still chooses every objective; only when it is asked changes.
Gate 1 (pixel): MAP01, 180 s, seeds 54, 1, 2, 3, same flags as the full comparison, suffix `pix3_gain`. Passes with ≥ 2/4 finished and ≤ 2 deaths.
Gate 2 (engine, run whatever gate 1 shows, since the change also affects engine games): MAP01, 180 s, same seeds, suffix `pix1_gain`. Not worse than `pix1_engine` (3/4).
Diagnostic: count of "keys, doors or mechanisms changed" re-plans, and objective switches between two door switches.

**Re-plan on gains: result** (`results/doom/repeat_map01_pix3_gain.json`, `repeat_map01_pix1_gain.json`; logs `results/vision/pix{3,1}_gain.log`; all valid and passing authority, no deaths).
Gate 1, pixel: **1/4, fails** (rule ≥ 2/4). Seed 2 finished MAP01 in 158.7 s, the first pixel game to finish a map; seeds 54, 1 and 3 ran out of time. Kills 13, 12, 17, 16. "Keys, doors or mechanisms changed" re-plans: 0–2 per game, against 2–8 in `pix3_admit`.
Gate 2, engine: **4/4, passes** (was 3/4 in `pix1_engine`). Finished in 113, 101, 128 and 108 s (`pix1_engine`: 177, 152, 130, then seed 3 not finishing). Seed 3, the game the switch loop had stopped, now finishes in 108 s. Seed 2's verifier exited 1 on acceptance criteria only (stationary longer than 4 s, upgrade delay), not on validity or authority; it counts as finished, as in every earlier comparison.
As registered, gate 1 fails. The change stays in the working tree, uncommitted, pending a decision: it helps engine games and does no harm in pixel games, but it is not what limits pixel play. The pixel games that did not finish still spent their time on phantom item objectives and attacks on remembered enemies.
Decision (user, 2026-11-04): keep re-plan on gains. It is the planner baseline from here on, for pixel and engine games. Next: detector precision.

**Pre-registration: pix5, on-policy frames (backlog item 4, DAgger round 1, 2026-11-04).** pix3's training views come from random warps and scenery. The views the agent produces in its own games (standing in front of a screen, rooms full of distant sprites) are missing, and that is where its phantoms come from.
- **Games:** pixel games with pix3 at 0.79 (`planner_ext_safe`, re-plan on gains) on six training maps, MAP06, 08, 10, 12, 14 and 16, seeds 54 and 1, 300 s each (suffix `pix3_onpol`). These maps are already in pix3's training set; MAP01–04 stay out.
- **Frames:** `collect_sequences.py` along each game's telemetry, one window over the whole game, a frame every 24 tics (about 5,000 frames), into `ext/vision-data/onpolicy_d`. The labels come from the labels buffer, as for every other set.
- **pix5:** pix3's recipe (`train_seg.py --depth`, same data, epochs and loss weights) plus `onpolicy_d`. Monster threshold set by `operating_point.py` to pix3's matched recall (MAP04 78.3 %).
- **Offline rule:** runtime replay on `seq_tic` (engine_MAP02 and engine_MAP03; memory with 1c and 1e, K = 0, merge off; one-to-one matching). Visible phantoms plus extras per frame at most 70 % of pix3's on both maps, and real monsters seen per frame at least 95 % of pix3's. Diagnostics: walked-view false monsters (`operating_point.py`), and item false blobs on the validation frames.
- **Gate (only if the offline rule passes):** MAP01, 180 s, seeds 54, 1, 2, 3, suffix `pix5`. Passes with ≥ 2/4 finished and ≤ 2 deaths (pix3 with re-plan on gains: 1/4).
*On-policy games* (`results/vision/pix3_onpol.log`): 12 games, MAP06–16, none finished in 300 s (not the point; these maps are long). Deaths 1–9 per game. All passed authority. The four MAP10 and MAP12 games are marked invalid: "Game did not sustain real time" on those crowded maps (worth knowing for later pixel games there; the views and labels are unaffected). `collect_sequences.py` gave 876 frames per map, 5,256 in total (`ext/vision-data/onpolicy_d`). pix5 training started on frames_d + scenery_d + onpolicy_d.

**pix5: result. Fails the offline rule; no gate.** (`results/vision/mobileclip_pix5/`, `operating_point_pix5.json`, `runtime_replay_pix5.json`; 12,756 frames, 119 min on MPS; Core ML 2.2 ms on the Neural Engine.) Matched threshold 0.699 (MAP04 recall 78.3 %). Walked views: monster recall MAP01/02/03 81 / 79 / 92 % (pix3 74 / 74 / 93 %), views with a false monster 6.5 % (pix3 5.1 %). Runtime replay, visible, one-to-one, share of pix3 at 0.79: phantoms + extras **124 % (MAP02), 116 % (MAP03)** (rule ≤ 70 %); real seen 108 %, 103 %. At the MAP04-matched threshold, pix5 is more sensitive on the game paths, with both more real and more false monsters.
*Diagnostic, not part of the rule: equal sensitivity.* Replays at other thresholds (scratchpad model directories with only `monster_threshold` changed). Phantoms + extras / real seen, as shares of pix3 at 0.79:

| | MAP02 | MAP03 |
|---|---|---|
| pix3 0.86 | 77 % / 93 % | 80 % / 95 % |
| pix3 0.92 | 51 % / 85 % | 60 % / 88 % |
| pix5 0.82 | 94 % / 100 % | 91 % / 95 % |
| pix5 0.90 | 67 % / 91 % | 68 % / 90 % |
| pix3 0.79 + merge 0.5 m | **61 % / 96 %** | **63 % / 97 %** |

At equal real monsters seen, pix5 lies on pix3's curve: slightly below it on MAP02, above it on MAP03. One round of on-policy frames did not change the detector's precision. A threshold only trades real monsters for false ones along the same curve. The depth merge at 0.5 m lies well below that curve on both maps: it removes duplicates without losing monsters in the way a threshold does.

**Pre-registration: gate pix3 + merge 0.5 m, after the offline miss (2026-11-04).** Stated plainly: the merge missed its own offline rule by 0.003 extras per frame on MAP03, and as registered it got no gate. This gate is registered after that, on new evidence: the threshold curves above, which show that no threshold reaches the merge's trade-off. The game gate is new data, unaffected by that choice. It is read as follows:
Gate: `MERGE_DEPTH_M = 0.5`, pix3 at 0.79, re-plan on gains. MAP01, 180 s, seeds 54, 1, 2, 3, suffix `pix3_merge`. Passes with ≥ 2/4 finished and ≤ 2 deaths (re-plan on gains alone: 1/4). Diagnostic: tics on `attack` aimed at remembered enemies (18–38 % before).
**Merge 0.5 m gate: result** (`results/doom/repeat_map01_pix3_merge.json`, log `results/vision/pix3_merge.log`; all valid and passing authority): **0/4, fails.** Kills 13, 19, 16, 13; 1 death. Tics on `attack` aimed at a remembered enemy: 1837, 2675, 1592, 1236 of 6300 (20–42 %), no lower than without the merge (18–38 %). Halving duplicate sightings offline did not reduce the time spent attacking remembered enemies in play. `MERGE_DEPTH_M` set back to 0.
Where this leaves level 1 (2026-11-04): with re-plan on gains, pixel MAP01 went 1/4 once; every perception-side change since (memory admission, item spots, depth merge, pix5) has given 0/4. With 4 seeds, 1/4 against 0/4 is within noise. In every pixel game, a fifth to two fifths of all tics go on attacking enemies that are not in view, against 5–20 % in engine games.

**Diagnostic: attacks on enemies not in view (2026-11-04, pixel games with re-plan on gains, `pix3_gain`).** 84 % of attack tics (8,178 of 9,683) were aimed at an enemy not detected on that tic. Ticks since it was last seen: p10/50/90 = 3 / 22 / 58 (none past the 70-tic window). 35 % of those tics were aimed at classes that never appear on MAP01 in engine games (only Zombieman, DoomImp and ShotgunGuy do): ChaingunGuy, HellKnight, Arachnotron, Revenant, Cacodemon, Baron, Demon, Fatso. 242 targets, the top five 11 %: many brief targets, not one stuck one. `unseen_in_view` was 0 on 92 % of them. Where the targets lay: in the 40° cone at ≤ 15 m 14 %, at 15–20 m 28 %, beyond 20 m 20 %; outside the cone 39 %. So **48 % lie in view but beyond the 15 m where negative evidence counts**, and the memory keeps them for up to 70 tics while Laya fires at them.

**Pre-registration: negative-evidence range (2026-11-04).** `NEGATIVE_EVIDENCE_M` (enemies only) swept over {15 (now), 20, 25, 30} in the runtime replay (`--forget-range`; pix3 at 0.79, 1c + 1e, K = 0, merge off; seq_tic engine_MAP02 and engine_MAP03). Rule: remembered phantom + duplicate entries per frame at most 70 % of the value at 15, and real monsters listed per frame at least 90 % of the value at 15, on both maps. The smallest range that passes is gated: MAP01, 180 s, seeds 54, 1, 2, 3, re-plan on gains, suffix `pix3_range`, ≥ 2/4 finished and ≤ 2 deaths. Diagnostic: share of attack tics aimed at enemies not in view (84 % now).
**Negative-evidence range: result. No range passes; no gate** (`results/vision/runtime_replay_pix3_range{15,20,25,30}.json`). Remembered phantom + duplicate entries at 20, 25 and 30 m: 99 % (MAP02) and 100 % (MAP03) of the value at 15; real monsters listed 100 %. The reason: the test also requires the spot to be nearer than the wall along its bearing. On the seq_tic views, the wall at eye level (rows 40–60 %) is at a median of 15 depth units (3.8 m), p90 61 (15 m). So a remembered enemy 15–30 m away in the cone is almost always behind the wall at that bearing, and range is not the binding limit. Two lessons. (1) The 48 % "in view, beyond 15 m" from the game telemetry are mostly spots behind walls, so Laya is firing at remembered positions it cannot see. (2) The replay follows the engine games' paths, so it cannot show effects that come from where the pixel agent itself turns and walks.
`agent.Sensors.in_clear_view` now reads `NEGATIVE_EVIDENCE_M` when called, not when defined (same behaviour; the replay can set it), and `runtime_replay.py --forget-range` exists. `NEGATIVE_EVIDENCE_M` stays 15.

**Pre-registration: memory facts in the enemy question, A/B (2026-11-04).** Today a remembered pixel enemy's option in the enemy question begins "Not visible now; last seen." Laya picks it as an attack target on 84 % of attack tics, and most such spots lie behind a wall. New flag `--memory-facts`, pixel mode only (engine games are unchanged), adding facts to that prefix and nothing else. Every option stays, and Laya still chooses.
- **A, `age`:** "Not visible now; last seen 0.6 s ago."
- **B, `wall`:** A, plus where the remembered spot is now: "Its last position is behind a wall from here." (in the view cone, farther than the wall along that bearing), "Its last position is in clear view and nothing is there." (in the cone, nearer than the wall), or "Its last position is out of view." (outside the cone). The same depth test as negative evidence, at any range.
Each arm: MAP01, 180 s, seeds 54, 1, 2, 3, re-plan on gains, pix3 at 0.79 (suffixes `pix3_memage`, `pix3_memwall`). An arm passes with ≥ 2/4 finished and ≤ 2 deaths. Mechanism check, reported for each arm: share of attack tics aimed at an enemy not detected on that tic (84 % in `pix3_gain`). If both pass, B is preferred only if it also has the lower share. Wording is fixed now, before any game.
**Amendment before any game (2026-11-04): the memory-facts A/B is withdrawn, replaced by the pixel memory window.** While implementing it I found that the wording cannot reach the model. With `--enemy-sequences` the enemy question is rebuilt from a "Targets:" line, and the server's `--enemy-rank-facts` projects it again into the enemy head's trained format (`name; distance; bearing; visible|last seen; latest accepted target`), so any added fact is dropped. More to the point, the target choice is not where the time goes. Of the 8,178 attack tics at unseen enemies in `pix3_gain`, 90 % had **no** visible enemy listed at all. Of 663 attack decisions, the target was visible when chosen in 21 % and any enemy in 28 %, against 59 % / 68 % (`pix1_engine`) and 69 % / 74 % (`pix1_gain`) for engine games over 207–274 decisions. The command head picks "attack" on remembered pixel enemies the way it learned to on remembered engine enemies, which are few and real. Changing its input format would put it outside its training.
**Pre-registration: pixel memory window.** New flag `--pixel-enemy-memory-ticks N` (pixel mode only; default 70, as now). The state line already says "seen in the last N seconds" from this value, so the model is told the window. This is a perception parameter: what the screen reader reports as recently seen. Arms N = 35 (1 s) and N = 17 (0.5 s; the median age of attacked unseen targets was 22 tics). Each arm: MAP01, 180 s, seeds 54, 1, 2, 3, re-plan on gains, pix3 at 0.79 (suffixes `pix3_mem35`, `pix3_mem17`). An arm passes with ≥ 2/4 finished and ≤ 2 deaths. If both pass, the longer window is kept (it costs less memory of real monsters). Mechanism check for each arm: attack decisions with the target visible (21 % now) and attack tics at unseen enemies (84 % now). Offline, for the cost only (not a rule): runtime replay real monsters listed per frame at each N.
**Pixel memory window: result. Both arms fail** (`results/doom/repeat_map01_pix3_mem{35,17}.json`, logs `results/vision/pix3_mem{35,17}.log`, offline cost `results/vision/runtime_replay_pix3_mem{70,35,17}.json`; all valid and passing authority).
- N = 35: **1/4** (seed 54 finished at 179.7 s). Kills 16, 16, 23, 15; deaths 2.
- N = 17: **0/4.** Kills 18, 13, 14, 13; deaths 1.
- Mechanism (`pix3_gain` → 35 → 17): attack decisions 663 → 517 → 437; target visible when chosen 21 → 34 → 32 %; attack tics 9,683 → 6,493 → 4,567; at unseen enemies 84 → 74 → 68 %. Offline cost: real monsters listed 88 / 83 % (35) and 77 / 67 % (17) of N = 70.
- **Where the time went:** share of tics by executed action, MAP01:

| | use_switch | pickup | attack | open_door | exit |
|---|---|---|---|---|---|
| engine, `pix1_gain` (4/4) | 29 % | 26 % | 24 % | 12 % | 7 % |
| pixel, `pix3_gain` (1/4) | 5 % | 39 % | 49 % | 4 % | 1 % |
| pixel, `pix3_mem17` (0/4) | 11 % | 48 % | 31 % | 6 % | 0 % |

Attack time freed by the shorter memory went mostly to pickups (pixel item ids), not to the switches and doors the route needs. The command head treats pixel-reported enemies and items as it learned to treat engine ones, and removing one false source moves the time to the other. Flag kept, default 70 (unchanged).

**Pre-registration: command residual distilled from engine-perception choices (2026-11-06; user go-ahead "Go ahead").** Every perception-side gate has failed the same way: the command head treats pixel-reported enemies and items as it learned to treat engine ones. This adapts the head itself instead. It is an adapted checkpoint, reported separately as its own variant (`laya-pixel-command-distill-v1`).
- **Teacher:** Laya's own served command head (encoder + `laya-v031-command-key-retained-v1` residual), asked about the *same moment* built from engine perception. New agent flag `--shadow-engine-packet` (pixel mode, data collection only): at every decision the agent also builds the command question from the labels buffer, with its own enemy memory (70 tics) and its own item memory, the same controller facts, and the planner's current objective applied without re-planning. It is recorded with the decision and never sent during play. After the games each recorded engine question goes to the same server; the teacher target is the residual's answer (the `look_gate` base answer, without `look_back`). No hosted model; labels are Laya's own.
- **Games:** pixel games, pix3 at 0.79, `planner_ext_safe`, re-plan on gains, MAP02, MAP03, MAP04, seeds 11, 12, 13, 14, 180 s (suffix `pix3_shadow`). MAP01 stays out: it is the gate. MAP02–04 are out of the detector's training set, so their false detections are what an unseen map gives.
- **Student:** the parent residual's architecture and spec, initialised from the parent. Its inputs are the pixel packet's projected features and the pixel semantic logits, exactly what it gets in play. Loss: KL from the teacher's distribution, restricted to the actions offered in both packets and renormalised, plus 0.1 × the squared logit drift from the parent. Decisions where the two offer no common action, or only one, are dropped. Seed 14 games are validation; the epoch with the best validation agreement is kept. CPU, about a minute.
- **Offline rule:** on validation decisions, the student's choice equals the teacher's argmax at least 10 points more often than the parent's does. Otherwise stop, with no gate. Reported: agreement by teacher action, and how often the student picks attack or pickup when the teacher does not.
- **Gate (only if the offline rule passes):** server with `--command-residual` (the student replaces the command residual's weights; everything else unchanged). MAP01, 180 s, seeds 54, 1, 2, 3, pix3 at 0.79, re-plan on gains, suffix `pix3_distill`. Passes with ≥ 2/4 finished and ≤ 2 deaths (`pix3_gain` 1/4). Mechanism check: share of tics by action, against `pix3_gain` (attack 49 %, pickup 39 %, use_switch 5 %), and attack decisions with the target visible (21 %).
- Known limit: the data come from the parent's own pixel play. If the student moves the agent to states it has not seen, a second round on the student's own games is the DAgger remedy, registered separately.
*Collection* (`results/vision/pix3_shadow.log`, `results/doom/repeat_map0{2,3,4}_pix3_shadow.json`): 12 games, all valid and passing authority, none finished in 180 s (not the point). 4,244 decisions with a shadow packet, all answered by the teacher (`<run>/shadow-teacher.jsonl`). Both recorded sides replay exactly through the parent residual (pixel side → recorded choice, engine side → teacher choice).

**Distilled command residual: result. Fails the offline rule; no gate** (`results/doom_heads/laya-pixel-command-distill-v1/{meta,metrics}.json`; 3,182 training and 1,062 validation decisions). Agreement with the teacher's choice, validation: parent **73.0 %**, student **73.5 %** (+0.6 points; rule +10). Training: 68.0 → 69.0 %.
Where they disagree (parent, training): the teacher picks attack on 44 % of decisions, so engine Laya also fights a lot. The gap is 731 decisions where pixel Laya attacks and engine Laya would pick up (522 → only 171 matched use_switch, 1,168 → 646 pickup). The student changed almost none of them.
*Diagnostics, not part of the rule.*
- Without the drift term (600 epochs, lr 1e-3; scratchpad), training agreement reaches only 74 % and validation 76 %. The student cannot fit its own training data, so the limit is the input, not the regulariser.
- What the pixel input shows on the 943 decisions (all 12 games) where pixel attacks and the teacher does not: remembered enemies only, engine sees none, 676; a pixel enemy visible now, engine none, 204. Across all decisions with "remembered only" in the pixel packet (1,641), engine perception has a real enemy in 844 and none in 797. The command residual's inputs (count and nearest distance by visible / last seen) are the same for both halves, so no weights can tell them apart.
- Can what the agent knows about each remembered track tell them apart? On the 1,592 decisions with remembered pixel enemies and none visible (telemetry at the decision tic), the label being "engine has an enemy": AUC by age of the newest sighting 0.58, by most tics seen 0.59, by total tics seen 0.63, by nearest distance 0.51, by count 0.69. A small MLP on all five, trained on seeds 11–13 and tested on seed 14: 0.68. Track evidence carries little of the answer either.
Conclusion: the command head's mistakes in pixel games are decisions made on information that does not say whether the remembered enemy is real. Neither filtering the reports (the earlier gates) nor re-weighting the decision (this one) can recover what the detector's output does not contain. The next lever is the evidence itself: what a remembered pixel enemy's report carries, or a detector whose remembered tracks are real far more often than half the time.
`--shadow-engine-packet`, `doom/distill_command.py` (now with `--drift`, default the registered 0.1), the server's `--command-residual` and the `planner_ext_safe_distill` config stay for later rounds.

**Diagnostic (not pre-registered, no rule): does geometry tell real remembered pixel enemies from phantoms? (2026-11-07)** Question from the user: an internal model with the engine version's states, filled from pixels. Before building it, this checks whether the evidence such a model would add separates real tracks from phantoms. `--shadow-engine-packet` now also writes the engine-visible monsters to telemetry (`engine_enemies`, truth for offline checks only). 6 games, pix3 at 0.79, re-plan on gains, MAP02/03/04, seeds 21 and 22, 180 s (suffix `pix3_truth`, log `results/vision/pix3_truth.log`; all valid and passing authority). `doom/vision/track_evidence.py` → `results/vision/track_evidence.json`.
- **Unit:** each remembered (listed, not visible) pixel entry at a decision tic, 2,605 in all. Label: the track is real when most of its sightings so far had an engine monster within 96 units + 20 % of distance. 61 % are real (MAP02 80 %, MAP03 42 %, MAP04 59 %). This is a per-track label, so it is not directly comparable with the decision-level 0.68 of the distillation diagnostic.
- **Features**, only from what was known by that tic. Geometry comes from the WAD's lines, sides and sectors, never its things. AUC by feature:
  - tics seen 0.77; scatter of the sighting positions 0.67 (real tracks scatter more: monsters move, while a phantom stays fixed to its spot); share of sightings behind a one-sided wall 0.64 (phantom 40 %, real 18 %); distance 0.60; mean position in no open sector 0.59 (phantom 34 %, real 16 %); wall distance 0.59; clear-view misses 0.58; age 0.56; entries listed 0.53.
- **Small MLP**, trained on seed 21 and tested on seed 22 (1,275 entries): time and counts only 0.80, geometry only 0.78, **all 0.87**.
- **Operating points (held out):**

| | keep 95 % of real | keep 90 % | keep 80 % |
|---|---|---|---|
| time + counts: phantoms removed | 26 % | 38 % | 57 % |
| all features: phantoms removed | **52 %** | 63 % | 78 % |
| all: real share of remembered list (now 64 %) | 78 % | 82 % | 87 % |

Reading: geometry and motion double the phantoms removed at the same loss of real enemies. This is the evidence an internal state model would add. Caveats: one test seed, three maps; positions of real monsters also fall in walls or closed sectors often (16–18 %), because depth-based distance is noisy and doors open; static sector heights only. A track-admission threshold on tics seen alone was already gated (`pix3_admit`, 0/4), so the gain to test is the geometric part.

**Pre-registration: track belief for remembered pixel enemies (2026-11-07; user go-ahead "yes").** The internal-model step: each pixel enemy track gets a learned probability that it is real, from evidence known at that tic. Only remembered entries scoring at least the threshold are listed, so the packet keeps the engine version's format. Entries seen on this tic, and items, are unchanged.
- **Module:** `doomlib/track_belief.py`. One class serves both play and the offline scripts, so the features are identical in both:
  - WAD geometry: lines, sides and sectors, never things;
  - per-track history: sightings, position, player position, behind a one-sided wall;
  - clear-view misses.
  Features: age, tics seen, distance, wall distance, void, behind-wall share, scatter, clear misses. "Entries listed" is dropped: it would depend on the filter itself, and its AUC was 0.53. Scorer: an MLP (8 → 32 → 1) run in numpy, weights in JSON. New agent flag `--pixel-track-belief DIR`, pixel mode only.
- **Data:** pixel games, pix3 at 0.79, re-plan on gains, `--shadow-engine-packet`, MAP02/03/04, 180 s. Training: seeds 21, 22 (`pix3_truth`) and 23, 24 (new). Test: seeds 25, 26 (new, suffix `pix3_truth`), not read until the scorer and its threshold are frozen. The threshold is set on the training entries to keep 92 % of real ones.
- **Offline rule (test seeds, remembered entries at decision tics):** at least 90 % of real entries kept and at least 50 % of phantom entries removed. Otherwise stop, no gate.
- **Gate:** MAP01, 180 s, seeds 54, 1, 2, 3, pix3 at 0.79, re-plan on gains, `--pixel-track-belief`, suffix `pix3_belief`. Passes with ≥ 2/4 finished and ≤ 2 deaths (`pix3_gain` 1/4). Mechanism: share of attack tics aimed at an enemy not detected on that tic (84 %), and tic shares by action (attack 49 %, pickup 39 %, use_switch 5 %).
**Track belief: result. Fails the offline rule, narrowly; no gate** (`results/vision/track_belief_v1.json`, sha256 c08263c3…, and `track_belief_v1_test.json`; games `results/vision/pix3_truth2.log`).
- **Games:** the 12 new games (seeds 23–26) all passed authority. MAP03 seed 25 is invalid ("did not sustain real time", 202 s wall for 180 s): my trainer smoke test ran at the same time. Features and labels are per tic, so it stays in the test as registered, and the result without it is also given.
- **Training:** seeds 21–24, 5,629 remembered entries, 58.5 % real. Threshold 0.354 keeps 92 % of real training entries and removes 63 % of training phantoms.
- **Test:** seeds 25, 26, 3,313 entries, 56 % real. **Real kept 90.2 % (rule ≥ 90 %, met), phantoms removed 48.4 % (rule ≥ 50 %, missed).** Without MAP03 seed 25: 89.5 % / 47.5 %. Real share of the remembered list 56 → 69 %.
- **By map:** MAP02 87.5 % kept / 28 % removed; MAP03 94 % / 57 %; MAP04 91 % / 56 %.
- **Reading:** the diagnostic's 52 % at 95 % came from one train/test split of seeds 21/22. With twice the training data and fresh seeds, the scorer generalises less well, mostly on MAP02, where phantoms are fewer (35 %) and look more like real tracks. The threshold is not moved after the fact.
- `doomlib/track_belief.py` (with tests), `doom/vision/train_track_belief.py` and the telemetry truth stay. The agent flag `--pixel-track-belief` was prepared but not applied, since there is no gate to run.

**Pre-registration (2026-11-07): oracle upper bound for the pixel memory (diagnostic, not a level-1 result).** Question: if the internal model filled from pixels were perfect at telling real tracks from phantoms, would MAP01 be finished? New agent flag `--pixel-oracle-filter` (pixel mode only, diagnostic): each pixel sighting is matched against the labels buffer on that tic (an engine object of the same kind, monster or item, within 96 units + 0.2·d, the track-belief label rule); a track is real while most of its sightings matched. Remembered pixel enemies on phantom tracks are dropped from the list; pixel items on phantom tracks are dropped before they reach the item memory. Seen pixel enemies are not filtered (detector errors, a separate question). The engine is used here as truth in play, so this is an upper bound only and is never reported as level 1.
- **Run:** MAP01, 180 s, seeds 54, 1, 2, 3, pix3 at 0.79, re-plan on gains (as `pix3_gain`), plus `--pixel-oracle-filter`, suffix `pix3_oracle`.
- **Reading, fixed now:** ≥ 2/4 finished with ≤ 2 deaths → the phantom memory is the bottleneck, and a better real/phantom model (enemies and items) is worth pursuing. ≤ 1/4 → it is not the main bottleneck; more filtering will not finish MAP01, and the next look is elsewhere (unseen real enemies, seen phantoms, Laya's play from pixel views). Mechanism reported either way: share of attack tics aimed at an enemy not detected on that tic (84 % in `pix3_gain`), and tic shares by action.
**Oracle upper bound: result. 1/4 finished, so by the reading fixed beforehand the phantom memory is not the main bottleneck** (`results/doom/repeat_map01_pix3_oracle.json`, log `results/vision/pix3_oracle.log`; all 4 valid and passing authority).
- Seed 54 finished (160 s, 0 deaths); seeds 1, 2, 3 timed out at 180 s (0, 1, 1 deaths). `pix3_gain` was also 1/4, on seed 2.
- **Mechanism:** attack tics 11,951 → 8,095 over the 4 games; attack share of tics 49 % → 33 %, pickup 39 % → 40 %, use_switch 5 % → 14 %, open_door 4 % → 9 %. Attack tics on a remembered (unseen) target 8,178 → 4,178; the share aimed at an enemy not detected on that tic 87 % → 81 % (my recount of `pix3_gain` with this script gives 87 %, against the 84 % noted earlier). Remembered targets left after the filter are real tracks, so Laya still spends most attack time on enemies it cannot see.
- Unfinished games ended 1, 18 and 21 m from the exit (`pix3_gain`: 23, 32, 40 m), so the filter moved play forward, but not enough. Seed 1 ended 1 m short.
- **Reading:** a perfect real/phantom memory does not by itself finish MAP01 from pixels. What remains: attacking real enemies that are out of sight (the command head's habit from engine games, where remembered enemies are few), and 40 % of tics on pickups. No fix is taken from this run.

**Diagnostics (not pre-registered, no rule, 2026-11-07): blind attacks and pickup time in MAP01 pixel games.** Both offline on recorded games: `pix1_engine` (engine perception), `pix3_gain` and `pix3_oracle` (4 games each). No fix is taken from the held-out maps; these are development-map recordings.
- **Blind attacks** (`doom/vision/attack_counterfactual.py` → `results/vision/attack_counterfactual.json`): command decisions answered attack while enemies were listed but none visible. Share of attack decisions: engine 20 % (55/274), `pix3_gain` 67 % (445/663), `pix3_oracle` 48 % (217/451). Replaying the recorded questions through the served stack gives attack on all 662 pixel ones (the recordings replay). With the remembered enemies removed, as the agent builds a question with no enemies listed, Laya picks **pickup** in 89 % (`pix3_gain`) and 83 % (`pix3_oracle`), use_switch 8 % / 12 %. Within 3 s of a blind attack: an enemy was in view 91 % (engine), 85 %, 73 %; health was lost 24 % (engine), 9 % (`pix3_gain`), 27 % (`pix3_oracle`). Once phantoms are gone, blind attacks are about as often followed by a real threat as in engine games. They are more frequent because pixel detection drops in and out.
- **Pickup time** (`doom/vision/pickup_time.py` → `results/vision/pickup_time.json`; an episode is consecutive pickup tics on one target, across re-decisions): pickup share of tics 22 % (engine), 39 %, 40 %. Episodes that came within 40 units of the item: **65 % engine, 12 % `pix3_gain`, 8 % `pix3_oracle`**. Of pixel pickup tics, **61–62 % go to item classes never listed in the engine games of MAP01**: BlueCard 30–31 %, Chaingun 10–16 %, SuperShotgun 3–5 %, RocketLauncher. The oracle did not remove them, because it matches a sighting to any engine item by position, not by class: these are mostly real objects given the wrong class. Laya then chases a "key" or a "weapon" that is not there as such, and the target is lost ("no longer observable") or replaced before it is reached.
- **Reading:** the command head is not the main fault. Hidden from remembered enemies, it would move the time to pickups, and pickups are where the pixel games lose most: item class errors (keys and weapons that are not on the map) draw Laya's highest-value choices. The next look is the detector's item classes (confusion of BlueCard, Chaingun, SuperShotgun against what is really there), not the memory or the command head.

**Diagnostic (not pre-registered, 2026-11-07): what the impossible item classes are.** `doom/vision/item_confusion.py` (cells of each predicted item class against the engine instance covering them). On the validation frames (`frames_d`, `scenery_d`, MAP01–04, 1,000 frames) and on engine-path views (`walk_d`, MAP01–03) the item classes look sound, e.g. BlueCard cells 41 % / 67 % on a real BlueCard (`results/vision/item_confusion_pix3.json`, `item_confusion_pix3_walk.json`). At the poses where the pixel games listed a BlueCard, Chaingun, SuperShotgun or RocketLauncher on MAP01 (241 views collected there with `collect_frames.py --from-runs`, map restarted), the cells called **BlueCard lie on nothing in 87 %**, Chaingun on nothing 61 % and on an ExplosiveBarrel 28 %, RocketLauncher on nothing 89 %. They are scenery read as items, not real items misnamed. The oracle kept them because its match radius (96 units + 0.2·d) usually holds some real item on this map. Item tracks seen in the 8 pixel games: about 6,000; the share of impossible classes falls with the tics a track was seen (32 % at 2 tics, 7 % beyond 40), so confirmation by time separates them only weakly.

**Pre-registration (2026-11-07): item operating point (pix3, items only).** Change: `classes.json` gains `item_threshold`; in `PixelPerception.cells`, a cell whose class is an item keeps it only when its summed item-class probability (Health, Armor, Ammo, Weapon, Key, Powerup) is at least the threshold, else it becomes none. Monsters are untouched (`monster_threshold` stays 0.7883). Same weights, same Core ML model.
- **Threshold** (`doom/vision/item_operating_point.py`), set before the rule is measured: the highest value on a grid of 0.50–0.99 (step 0.01) at which item object recall on MAP04 (`frames_d` and `scenery_d`; a validation map, not a game map) is at least 90 % of the recall at the plain argmax. An item instance is found when any cell it covers is predicted as an item; a false item object is a 4-connected group of item cells touching no item instance.
- **Offline rule** (`walk_d` MAP02 and MAP03, engine-path views, not used for the threshold): false item objects per frame at most 50 % of the argmax's, and item recall at least 85 % of the argmax's, on both maps. Otherwise stop, no gate.
- **Gate:** MAP01, 180 s, seeds 54, 1, 2, 3, re-plan on gains, pix3 with the item threshold (model directory `results/vision/mobileclip_pix3_items`: the same compiled model, `classes.json` with `item_threshold` added), no oracle; suffix `pix3_items`. Passes with ≥ 2/4 finished and ≤ 2 deaths (`pix3_gain` 1/4). Mechanism: pickup share of tics (39 %), pickup episodes reaching the item (12 %), pickup tics on classes never listed in MAP01 engine games (61 %), from `pickup_time.py`.
*Amendment before any measurement:* `scenery_d` holds training maps only, so the threshold is set on MAP04 `frames_d` alone (250 frames). Nothing else changes.
**Item operating point: result. Fails the offline rule; no gate** (`results/vision/item_operating_point_pix3.json`; the suspect-pose confusion of the diagnostic above is `results/vision/item_confusion_pix3_suspect_poses.json`).
- **Threshold:** 0.77. MAP04 item recall at the argmax 75.7 % (465 instances); at 0.77, 90.6 % of it, with false item objects per frame at 51 % of the argmax's.
- **Check (walk_d):** MAP02 false item objects 69 % of the argmax's (0.51 → 0.35 per frame), recall 93 %; MAP03 57 % (0.31 → 0.18), recall 91 %. **Rule: false ≤ 50 % on both maps, missed** (recall ≥ 85 %, met).
- *Not part of the rule:* on MAP04, halving false items costs about 10 % of recall, and getting to a third costs 15 % (0.85: recall 85 %, false 34 %). Item confidence alone separates real items from scenery only moderately; the walked views are harder than MAP04's.
- `PixelPerception` keeps the `item_threshold` option (off unless `classes.json` sets it; 2 tests). No model directory with it was made.
