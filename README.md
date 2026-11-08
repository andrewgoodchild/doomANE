# doomANE

**A 421M decision model plays Freedoom in real time, with every forward pass on the Apple Neural Engine.**
It works as a fast decision-maker you can audit, but what it learned did not carry over to new maps. On
the maps the stack was developed on, the model's choices are what make it play: it finishes the first three
and, at skill 1, the fourth, while the same code with random choices finishes none. On six maps it has never
seen it finishes nothing, even at skill 1, where it barely dies. The model chooses every command, target,
weapon, movement and objective, but from menus that code has already built with knowledge of the whole map,
from the engine's internal state rather than pixels. A second phase (2026-10-31 to 11-07) put a detector in
front of it that reads enemies and items from the screen and the depth buffer: from pixels the bot finishes
MAP01 in 1 of 4 games against 4 of 4 from engine state, and about two dozen pre-registered fixes tried on top
of the detector have not moved that ([Pixel perception](#pixel-perception-level-1)). This repository has the
code, the data, and an honest account of what helped and what did not.

![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-Neural%20Engine-555?logo=apple)
![Python 3.13](https://img.shields.io/badge/python-3.13-3776AB?logo=python&logoColor=white)
![Core ML](https://img.shields.io/badge/Core%20ML-coremltools%209-0A84FF)
![Licence](https://img.shields.io/badge/licence-Apache--2.0-green)
![Status](https://img.shields.io/badge/status-research%2C%20pixel%20perception%20in%20progress-lightgrey)

![Laya fighting through Freedoom MAP03 on the Neural Engine](assets/hero-map03.gif)

*Freedoom MAP03, skill 3, real time: a win with no deaths in 319 s. The panel shows the command the model
chose, its probability for every command, the weapon and movement it picked, and the decision latency.
Nothing on the panel is chosen by code; code decides which options are on the menu ([Rules we kept](#rules-we-kept)).*

## Results in short

![Completion rates: MAP01 10/10, MAP02 8/10, MAP03 4/10, MAP04 3/3 at skill 1 but 0/79 at skill 3, held-out maps 0/18](assets/completions-light.svg#gh-light-mode-only)
![Completion rates: MAP01 10/10, MAP02 8/10, MAP03 4/10, MAP04 3/3 at skill 1 but 0/79 at skill 3, held-out maps 0/18](assets/completions-dark.svg#gh-dark-mode-only)

| | |
|---|---|
| **Speed** | Encoder and question heads on the ANE: a median of about 110–140 ms per decision in play (about 420 ms with only the encoder there). Real time at 35 tics/s. |
| **MAP01–03** | Skill 3, real time, a death restarts the map from a pistol start (final 10-game batch per map): 10/10, 8/10, 4/10 (95 % intervals 72–100 %, 49–94 %, 17–69 %). MAP03 is noisy: 16/51 over the later stacks (20–45 %). |
| **MAP04** | A development map since 2026-10-23 (its failures drove the route fixes). Generic route rules find its whole route: **3/3 at skill 1** (44–100 %; two of the three on GPU servers), and once more inside a campaign. At skill 3, **0/79** (0–5 %): the route takes 500–700 s, and one death after the keys ends the run. |
| **Held-out maps** | 0/18 (0–18 %) on six held-out maps, like the doomLaya v0.3.1 we started from. With the route rules, from a frozen snapshot on new seeds: 0/18 again, with no more keys or doors. At skill 1, where it barely dies on most of them: 0/18, with twice the keys. MAP11 and MAP15 lost real time in both, so those games are invalid. |
| **Random baseline** | The same stack with every choice replaced by a random option: 0/12 (0–24 %) on MAP01–03, against 7/12 (32–81 %) for Laya in the same session, with 370 deaths against 45. |
| **Campaigns** | One hour from MAP01. Pistol start on every map: skill 3 reached MAP03 and MAP04, skill 1 MAP03 and MAP05. With the inventory carried over as in vanilla Doom: skill 3 MAP03 and MAP02, skill 1 MAP05 twice. Within the noise of 2 seeds. |
| **Where it stopped** | Fighting. Every lever on the decision side was tested and came up flat. |
| **From pixels (level 1)** | 2026-10-31 to 11-07: enemies and items from a detector on the screen and depth buffer, everything else as before. MAP01, 180 s, 4 seeds: best **1/4** (engine 4/4); MAP01–03 full comparison 0/12 (engine 6/12). Phantom enemies and scenery read as keys and weapons take the time; fixes on the memory, the planner, the decision head and the detector have not moved it. |

The full story is in [docs/WRITEUP.md](docs/WRITEUP.md), and every batch, table and dead end is in the
[lab notebook](docs/LAB_NOTEBOOK.md).

## How it works

```mermaid
flowchart LR
    G["ViZDoom<br/>Freedoom 2, 35 tics/s"] -->|engine state| T["Code: state as text<br/>+ typed questions"]
    P["System 1 planner<br/>objectives + what-ifs"] -->|objective| T
    T -->|"~2 decisions/s"| M["Laya 421M<br/>encoder + 8 question heads<br/><b>Apple Neural Engine</b>"]
    M -->|"command, target,<br/>weapon, movement"| E["Executor<br/>route, aim, buttons"]
    E -->|buttons| G
    M -.->|"objective choice"| P
```

- **The model chooses; code executes.** Code turns the game into text and asks typed questions (which
  command, enemy, item, weapon, movement). The executor routes, aims and presses buttons for the chosen
  command, and never picks a target, a weapon or an exit itself. `tools/check_authority.py` audits every
  run, and a completion counts only if the run passes.
- **Laya** is the doomLaya v0.3.1 fine-tune of Laya (ModernBERT-large, 421M): one shared encoder, eight
  question heads and small numeric residuals, all as Core ML on the ANE (the encoder as one multifunction
  package of seven sequence lengths, 0.76 GB).
- **Route knowledge is generic:** a table of Doom 2's linedef specials, lifts, crushers and headroom. No
  map name, line number or coordinate is used in the code's logic.

![Milliseconds per decision: 612 with the encoder on the ANE, 276 with the heads too, 206 when one-option questions are skipped](assets/latency-light.svg#gh-light-mode-only)
![Milliseconds per decision: 612 with the encoder on the ANE, 276 with the heads too, 206 when one-option questions are skipped](assets/latency-dark.svg#gh-dark-mode-only)

*An offline benchmark used to compare the optimisation steps: 40 recorded MAP03 packets replayed in one
process. The 110–140 ms in the table above is the server's median measured in real games. The two are
measured differently, so compare each with its own kind.*

## System 1 planning

Laya answers every half second from one snapshot of the game, with no plan and no memory; on its own it
shuttled between three switches on MAP03 for 15 minutes with the red key about 5 m away. The planner
(`doom/planner.py`) splits the work: **code does the search, the model does the choosing.** Code lists the
reachable objectives (keys, switches, lifts, the exit, exploring), with what-ifs of what each would open
two steps ahead. Laya picks one through an ordinary `choice` question, only on events (an objective done or
failed, a key, door or floor change, a respawn), and the objective then narrows the 2 Hz route options.

The first objective question of a winning MAP04 game (skill 1, seed 101):

| Option | Laya's probability |
|---|---|
| Use floor switch #883, 22m away; reachable now. Opens: red key (which opens lift #463). On the way to: red key. Not tried yet. | **0.69** |
| Collect the Shotgun (#1), 4m away; reachable now. A weapon not yet owned. Not tried yet. | 0.31 |
| Collect the SuperShotgun (#weapon_27), 25m away; reachable now. A weapon not yet owned. Not tried yet. | 0.005 |
| Use door switch #879, 30m away; reachable now. Opens: nothing new found. Not tried yet. | 0.00 |
| No fixed objective: explore accessible areas for another route. Not tried yet. | 0.00 |

Zero-shot, Laya's first objective was the first step of the real route on MAP01–04. On the same four
seeds of MAP01–03, the planner raised completions from 7–8 of 12 to 10 of 12, and the gain came from the
facts and filters code gives it, not from the question. A local LLM in its place (Qwen3 8B and 27B, MLX)
chose no better. Details: [docs/research/System 1 planning/notes.md](docs/research/System%201%20planning/notes.md).

## Pixel perception (level 1)

Every number above comes from the engine's object list: only what is on screen, but with exact names, ids and
positions. From 2026-10-31 to 11-07 the enemy and item lists came instead from a detector reading the screen
and the depth buffer (agent flag `--pixel-perception`, `doomlib/pixel_perception.py` in the patch, tooling in
`doom/vision/`). Level 1 means that the map, doors, switches, keys, counters and projectiles still come from the
WAD and the engine, so these games measure the cost of perception alone, not a human-fair game. The dated record
is in the [lab notebook](docs/LAB_NOTEBOOK.md) from the pix1 entry on, the fuller account in
[docs/WRITEUP.md](docs/WRITEUP.md), and the reading list and backlog in [docs/PIXEL_BACKLOG.md](docs/PIXEL_BACKLOG.md).

- **Detector.** MobileCLIP-S2's image trunk, fine-tuned, with a small CNN on the depth buffer, names every 4x4
  cell of the 192x320 view (48x80 cells) with one of 49 classes: monsters and items by name, "Other" per
  category, none. Ground truth is ViZDoom's labels buffer on 13 development maps; MAP01–04 are validation maps
  and never in training. 2.2 ms per frame on the Neural Engine, run every tic. Touching cells of one category at
  one depth form an object; distance comes from the depth buffer along the bearing; ids come from a position
  tracker; an object is reported after 2 consecutive tics and remembered for 70 after its last sighting.
- **Six detectors** (pix1–pix6: RGB only; empty cells weighted up; depth as a fourth channel; distractor classes
  for corpses, barrels and decorations; on-policy frames from the agent's own games; item hard negatives). At
  the same recall, pix3 with depth is clearly the best: 5.1 % of views along the agent's paths show a false
  monster, against 13.3 % for pix1. The later retrains sat on the same recall-against-phantoms curve or below it.

| MAP01, skill 3, 180 s, seeds 54, 1, 2, 3 | Finished |
|---|---|
| Engine perception, same stack and session | 3/4; 4/4 with re-plan on gains |
| pix1 | 0/4 (MAP01–03: 0 of 9, against 6 of 12) |
| pix3 with forgetting on negative evidence | 0/4 (MAP01–03: 0 of 12, against 6 of 12; 0/4 with 600 s) |
| pix3, forgetting and re-plan on gains | **1/4**, the first pixel game to finish a map |
| Six further gates: item spots, memory admission, depth merge, memory windows of 1 s and 0.5 s, oracle memory | 0/4, 0/4, 0/4, 1/4, 0/4, 1/4 |

What went wrong, in the order it was found:

1. **Phantoms that never go away.** pix1 read wall screens at the MAP03 start as monsters, and Laya attacked them
   in 2,129 of 2,335 decisions until the ammo was gone. The depth channel (pix3) cut phantoms on bare walls from
   59 to 25 on the test views; what is left are sprites that are not monsters, which stand out in depth just as
   monsters do.
2. **Memory that never empties.** In pixel mode a remembered enemy stayed 70 tics with nothing else removing it,
   duplicates of one monster piled up under new tracker ids, and items were never forgotten (one planner
   question had 129 "Collect the Shotgun" options and crashed the server). Forgetting a remembered object whose
   spot is in clear view and empty, and dropping a remembered enemy that a seen one explains, got the bot moving
   on MAP03 (the short gate passed), but 0 of 12 full games finished.
3. **Where the time goes.** In pixel games 84 % of attack tics were aimed at an enemy not in view (engine games:
   5–20 %), and 39–40 % of all tics went to pickups, 61 % of that to item classes not on the map: scenery at a
   few poses read as a blue key, a chaingun or a rocket launcher, the options the planner values most.
4. **Filters on the memory do not fix it.** Admission by track length, a depth-tolerant merge, a longer
   negative-evidence range, a shorter window, and real-or-phantom scorers from WAD geometry (AUC 0.87) each
   missed their offline rule or their gate; real monsters are often seen briefly, so phantoms and real tracks
   look alike. An oracle that drops phantom tracks with the engine's labels (a diagnostic, never a result) still
   finished only 1 of 4: a perfect memory filter would not be enough.
5. **The decision cannot be re-weighted.** A command residual distilled from Laya's own answers to the same
   moments built from engine perception agreed 0.6 points more with the teacher (rule: 10). The residual's
   inputs are identical for a real and a phantom remembered enemy, so no weights can tell them apart.
6. **Three item fixes missed** (an item confidence threshold, an item belief from geometry, pix6's hard
   negatives), and one round of on-policy frames (pix5) did not move the detector's precision.

What this phase taught: offline detector scores do not predict play, and the measures that do are phantoms that
persist along the agent's own paths and what its memory accumulates (`doom/vision/runtime_replay.py` replays the
stack's perception and memory tic by tic in seconds); nothing downstream can recover what the detector's output
does not contain; and the one change that stuck was on the planner, not perception: re-planning only when
something is gained, which took engine MAP01 from 3/4 to 4/4. Level 1 is open, with the detector's precision on
items and on sprites that are not monsters as the next lever.

## What we learned

The method, a small fast model choosing among options that code lists, works as an on-device chooser whose
every decision can be audited. It does not make the bot good at Doom: it is only as good as the options it
is offered and the actions the executor can carry out.

1. **Speed is the easy part.** A game asks about two decisions a second, so an ANE server is busy only
   about a quarter of the time.
2. **It chooses well, but cannot judge.** Given concrete options it picks sensibly, even zero-shot. Yes/no
   judgements ("making progress?", "pull back?") were at chance (AUROC 0.36–0.52), with yes 70–95 % of the
   time whatever the facts.
3. **Wording is a weapon.** Option names, wording and order move choices by tens of percent, and names
   weigh more than the conditions in them: "Ammo: N" against "Shots left: N" moved choices that cannot hit
   from 0 to 35 in 200 replayed questions. Test a new wording offline before playing it.
4. **You cannot tell it things, and the numbers often decide.** Free-text hints changed no command in 463
   replays. A small numeric residual on top of Laya changes the command in 21–50 % of decisions.
5. **Its competence is partly borrowed.** Most gains came from code: executor fixes, the planner's option
   lists, the route rules. That code was shaped on the development maps and did not transfer to held-out
   ones. But the code alone does nothing: with random choices it finished 0 of 12 games on MAP01–03, against
   7 for Laya, and died eight times as often. The model's choices carry the play the code makes possible.
6. **The executor's actions set the ceiling.** Without a "fall back while firing" action, choosing to avoid
   a fight could not be carried out safely. Skill 1 shows the route is solved on the development maps; skill 3 shows survival is not. On held-out
   maps at skill 1 Laya survives and still finds no exit, so there the route is the limit.
7. **The decision layer is not where fights are lost.** A movement head trained on rollout outcomes from
   7,056 states beat the original offline but not in games; forcing each alternative command found none
   better than Laya's beyond the noise; Qwen 8B/27B as planner and untrained open decision models did no
   better. Cheap outcome labels teach short, local choices; which fight to take is a longer-horizon one.
8. **Single runs mislead.** One different decision sends a game down another path: the same seed gave 0,
   6 and 4 deaths on repeats. Use several seeds, paired offline replays, held-out maps played only from a
   frozen snapshot, and decide what each result means before the batch.

Going further needs richer executor actions, or something that learns values over longer horizons (a real
System 2, or far more game experience), not more questions or new wording.

## Compared with game AI

The design matches what shooters use for their enemies: the executor is a
behaviour tree, System 1 a utility AI that scores the options, and the planner a GOAP/HTN-style planner
(F.E.A.R., Killzone 2). Only the scorer is learned. The job is harder, though: enemies are built to lose,
never solve levels, and get hand-marked cover points, where this agent has to win at skill 3, find keys in
unknown maps, and has no fall-back action. Agents that do play as the player lean on authored help
(deathmatch bots' waypoints) or very large amounts of RL (the ViZDoom competition agents, GT Sophy).

## Quick start

**You need:** an Apple Silicon Mac, Xcode command line tools, and
`brew install uv ffmpeg cmake boost sdl2 openal-soft`. About 20 GB of disk.

```bash
git clone https://github.com/andrewgoodchild/doomANE.git && cd doomANE
scripts/setup.sh        # doomLaya v0.3.1 + our patch, venv, patched ViZDoom, checkpoints, Core ML export, tests
cd ext/doomLaya-dev
.venv/bin/python ../../doom/repeat_map.py --maps MAP01 --seeds 54 --configs planner_ext_safe --suffix demo
```

This starts an ANE model server, plays, verifies the run and audits authority. The run directory
(`runs/<time>_repeat-...`) holds `video.mp4`, `decisions.jsonl` (every question, answer and probability),
`events.jsonl`, `telemetry.jsonl` and `summary.json`; the result row goes to
`results/doom/repeat_map01_demo.json`.

| Task | Command (from `ext/doomLaya-dev`) |
|---|---|
| Several seeds and games at once | `.venv/bin/python ../../doom/repeat_map.py --maps MAP03 --seeds 1 2 3 4 --configs planner_ext_safe --parallel 4 --ane-servers 2` |
| Scenarios from saved fight states | `.venv/bin/python ../../doom/scenarios.py --runs runs/<run> --configs planner_ext_safe --seconds 15 --out out.jsonl` |
| Movement rollouts for training labels | `.venv/bin/python ../../doom/fight_rollouts.py runs/<run> --before-death 10 --out labels.jsonl` |
| Static route check | `.venv/bin/python ../../doom/route_solver.py MAP04 --extended --diagnose` |
| Freeze the stack (from the repository root) | `doom/snapshot_stack.sh <name>` |
| A campaign (from `ext/snap/<name>/ext/doomLaya`) | `.venv/bin/python ../../doom/repeat_map.py --maps MAP01 --campaign 3600 --seeds 301 --configs planner_ext_safe --evaluation --extra-flags='--carry-inventory'` |
| A game from pixels (level 1; needs a trained detector, see below) | `.venv/bin/python ../../doom/repeat_map.py --maps MAP01 --seeds 54 --configs planner_ext_safe --extra-flags="--pixel-perception $(pwd)/../../results/vision/mobileclip_pix3" --suffix pixdemo` |

- Always pass `--configs`; the `system2*` configs need a local MLX server.
- Core ML holds the ANE models of two processes at most. With `--parallel` above 2, the other servers run
  the same model on the GPU (identical answers on 40 of 40 checks). Game results are from ANE servers
  unless marked; scenarios and rollouts used GPU servers.
- Campaigns reach the held-out maps, so they run only from a snapshot with `--evaluation`.
  `--carry-inventory` carries weapons, ammo, backpack, health and armour to the next map as vanilla Doom
  does (ViZDoom would otherwise start every map from a pistol start), and stops the run if any field is not
  restored exactly.
- **Reproducing exact numbers:** `scripts/setup.sh` builds the final stack (2026-10-29). Earlier batches ran
  on earlier code (the oldest with `doom/patches/doomLaya-executor-fixes.patch`); each run directory keeps
  the exact source it ran. The run directories and videos are not in this repository.
- **The pixel detector is rebuilt, not downloaded:** its weights and Core ML packages (`results/vision/*/model.pt`,
  `*.mlpackage`) and the frame sets (`ext/vision-data/`) are not in the repository; the reports, operating points
  and replays are. Collect frames on the development maps with `doom/vision/collect_frames.py` (random warps,
  `--scenery`, and `--from-runs` for views along recorded games), train with `train_seg.py --depth` (about an hour
  on the GPU in `ext/vision-venv`), export with `export_seg.py`, and set the monster threshold with
  `operating_point.py`.

## Repository layout

| Path | What |
|---|---|
| `doom/ane_doomlaya.py`, `doom/export_doom_*.py`, `doom/merge_doom_encoder.py` | The ANE model server and the Core ML export |
| `doom/planner.py`, `doom/system2.py` | The System 1 objective planner used for every headline number; the optional local LLM planner |
| `doom/planner_v6.py` | An earlier planner, frozen only for A/B comparisons (config `planner6`) |
| `doom/repeat_map.py`, `doom/scenarios.py`, `doom/fight_rollouts.py` | Game batches, scenarios from saved states, outcome rollouts |
| `doom/route_solver.py`, `doom/map_split.json` | Static route checks; the development / held-out split |
| `doom/patches/doomLaya-dev.patch` | Everything we changed in doomLaya v0.3.1 |
| `doom/vision/` | Pixel perception (level 1): frame and sequence collection, detector training and Core ML export, matched operating points, the runtime replay of perception plus memory, and the offline diagnostics |
| `doom/distill_command.py` | The command residual distilled from Laya's own engine-perception choices (missed its rule) |
| `coreml_cache.py` | Compile each Core ML package once (`MLModel(path)` leaks a compiled copy per load) |
| `results/doom/`, `results/doom_heads/`, `results/vision/` | Every batch's results; outcome labels and the trained movement residual; detector reports, operating points, runtime replays and belief scorers |
| `docs/WRITEUP.md`, `docs/LAB_NOTEBOOK.md`, [`docs/research/`](docs/research/README.md), `docs/PIXEL_BACKLOG.md` | Summary, dated record, research notes, the pixel reading list and backlog |

Most of the game-side code is in [`doom/patches/doomLaya-dev.patch`](doom/patches/doomLaya-dev.patch), against
doomLaya v0.3.1 (`scripts/setup.sh` applies it). The files worth reading in it:

| File in the patch | What |
|---|---|
| `doomlib/executor.py` | Carries out the model's commands; no combat, item or weapon preferences of its own |
| `doomlib/policy.py`, `serve_doom_laya.py` | Turns the game state into the model's questions and options |
| `doomlib/specials.py`, `doomlib/mission.py`, `doomlib/navigation.py` | Doom 2's linedef specials as one table, level exits and routes |
| `doomlib/combat_facts.py` | What the player can see of a fight (projectiles, threat types), for the movement question |
| `doomlib/carry.py`, `assets/carry_inventory.acs` | Inventory carried to the next map in campaigns |
| `doomlib/pixel_perception.py`, `doomlib/track_belief.py`, `tests/test_pixel_*.py` | Enemies and items from the screen (cells, objects, tracker, confirmation) and the real-or-phantom scorers (trained, not wired in) |
| `tools/check_authority.py`, `tests/test_authority.py` | The audit that every executed choice is the model's own |

## Rules we kept

- **Authority:** the model chooses the command, target, weapon and movement; the executor carries them out.
  An API error is never replaced by a heuristic.
- **Real time:** wall and game time agree within 5 %.
- **Held-out maps** (odd Freedoom maps from MAP05, MAP31–32, an Obsidian set) are played only from a frozen
  snapshot, and their results are never used for fixes.
- **No hosted model** is in the decision loop or produces training data (the `memory` config's command
  residual was trained on a hand-written rule).

Where code still decides, or we broke a rule:

- **Crusher safety** (in `planner_ext_safe`): the executor moves the player out from under a closing
  crusher whatever movement the model chose.
- **Rocket gate:** no rockets at targets closer than 5.5 m (4 m in the earlier batches).
- **Item options:** the planner decides which items are worth offering (keys, new weapons within 15 m,
  health below 50, items within 6 m); the model chooses among them.
- **Commitment** (configs `commit`, `resume*`, 2026-10-19 only): the server could keep Laya's previous
  answer over a new one, invisibly to the audit. The headline batches do not use these configs.
- **MAP04** was held out until 2026-10-22, then used for fixes.
- **Pixel mode (level 1):** perception-side rules decide what is reported, never what is chosen. An object is
  reported after 2 consecutive tics; monster cells need the matched threshold (0.79 for pix3); unnamed weapons
  and keys and the distractor classes are dropped; a remembered enemy or item whose spot is in clear view and
  empty on 3 tics is forgotten; a remembered enemy is dropped when a seen enemy of the same name explains it;
  enemies stay listed for 70 tics after the last sighting. `--pixel-oracle-filter` and `--shadow-engine-packet`
  read engine labels and are diagnostics only; nothing played with them is a level-1 result.

## Limitations

- The headline numbers use engine internals, not pixels, and the route rules know the whole map. In pixel mode
  only enemies and items come from the screen; the map, doors, switches, keys, counters and projectiles still
  come from the WAD and the engine.
- Three held-out evaluations (two at skill 3, one at skill 1) found no completions; most results are from development maps.
- The later evaluations ran on one ANE server, and the stack lost real time on MAP11 and MAP15.
- No jumping or crouching; vertical aim is automatic. Every number is from one Apple Silicon Mac.

## Related work

- [azalio/doomLaya](https://github.com/azalio/doomLaya): the Laya fine-tune and controller this builds on.
- [dexmac221/doomGemma](https://github.com/dexmac221/doomGemma) and
  [dexmac221/system-switch](https://github.com/dexmac221/system-switch): Laya against Gemma, and when a fast
  model should hand over to a reasoner.
- [dylanbstorey/laya-doom](https://github.com/dylanbstorey/laya-doom) (Laya in real time on the Apple GPU)
  and [mizorewww/laya-coreml](https://github.com/mizorewww/laya-coreml) (Laya on Core ML and the ANE).
- TypeSafe's Jev × Doom demo, which started the genre.
- Pixel agents: Lample & Chaplot's Arnold, the ViZDoom 2018 Track 1 winner (a YOLOv3 detector, depth and
  hierarchical RL on unseen maps), Sample Factory, and the VLM screenshot agents; what carries over is in
  [docs/PIXEL_BACKLOG.md](docs/PIXEL_BACKLOG.md).

## Credits and licence

Apache-2.0 (see [LICENSE](LICENSE) and [NOTICE](NOTICE)). Built on doomLaya and Laya (Apache-2.0),
ModernBERT (Apache-2.0), ViZDoom (MIT) and Freedoom (BSD-3-Clause). None of their code or weights is
bundled; the animation shows Freedoom graphics (see [LICENSES/Freedoom.txt](LICENSES/Freedoom.txt)). This
is an independent experiment, not an official Laya, doomLaya or Jev release. To cite it, see
[CITATION.cff](CITATION.cff).
