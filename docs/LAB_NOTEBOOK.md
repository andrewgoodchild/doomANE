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
