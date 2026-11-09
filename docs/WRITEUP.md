# Laya plays Doom from the Neural Engine: write-up (2026-10-30; pixel perception added 2026-11-08)

A summary of the Doom work in this repo, from 2026-10-18 to 2026-10-25, plus the evaluations after the
park (two more held-out evaluations, one-hour campaigns and a random-choice baseline, 2026-10-29 to 30), and the
level-1 pixel perception phase (2026-10-31 to 11-07, its own section below). The lab notebook with every
batch, table and dead end is `docs/LAB_NOTEBOOK.md`; this file says what was built, what it achieved, what we learned
and where it stopped.

## Goal and setup

Play Freedoom 2 in real time with a small typed decision model ("System 1") running on the Apple
Neural Engine, Jev-style: code turns the game into text and typed questions, the model answers, code
executes. The model must choose; code may not.

- **Model:** Laya (421M), as fine-tuned for Freedoom by `azalio/doomLaya` v0.3.1: one shared encoder,
  eight question heads (command, enemy, item, switch, movement, weapon, combat, look) and small numeric
  residuals.
- **On the Neural Engine:** the encoder (one multifunction Core ML package, 128–640 tokens) and the
  2-layer heads. Server median about 110–140 ms per decision, against about 420 ms with the encoder alone
  on the ANE and about 270–310 ms for doomLaya's published MPS runs.
- **Around the model:** doomLaya's navigator, aim and executor; our patches (`doom/patches/`); a
  System 1 objective planner (`doom/planner.py`); an optional local System 2 (Qwen3 through MLX).
- **Authority boundary:** the model picks the command, target, weapon and movement; the executor only
  carries it out. `tools/check_authority.py` audits every run, and only valid, audited completions count.
  The exceptions where code still decides (crusher safety, the rocket gate, which items the planner
  offers, and the early commitment configs) are listed in the README under "Rules we kept".
- **Rules of play:** skill 3, real time (wall and game time within 5 %), a time budget of 3 minutes for
  MAP01, 30 for MAP02 and 20 for the others, and a death restarts the map from a pistol start. No hosted
  model is in the decision loop or produces training data.

## Results

| Map | Result | Notes |
|---|---|---|
| MAP01 | 10/10 (exec5) | reliable since the planner (v6) |
| MAP02 | 8/10 (exec5); 3/3 (melee1) | sensitive to weapon wording; 0/2 on two seeds with `--combat-facts` (fight1, the last batch) |
| MAP03 | 4/10 (exec5); about 30–50 % | 13/25 pooled for planner v6; 16/51 over the later stacks; 1/3 in melee1 |
| MAP04 (held out until 2026-10-22, development since 2026-10-23) | **0/79 at skill 3; 3/3 at skill 1** | route works end to end; deaths end the run; 2 of the 3 skill-1 games ran on GPU servers |
| Held-out MAP05–15 (heldout1) | 0/18, as is clean doomLaya v0.3.1 (0/18) | nobody gets past the first key or two; played before the generic route rules |
| Held-out MAP05–15 (heldout2, 2026-10-29) | 0/18 (0/12 valid) | with the route rules; 7 keys and 509 doors against heldout1's 8 and 496; MAP11 and MAP15 lost real time (invalid) |
| Held-out MAP05–15 at skill 1 (heldout3, 2026-10-30) | 0/18 (0/11 valid) | 14 keys and 547 doors; 0–3 deaths per game on MAP05–09 but 7–10 on MAP13; MAP11, MAP15 and one MAP05 game lost real time |
| Random-choice baseline, MAP01–03 (eval4) | 0/12, against 7/12 for Laya in the same session | same stack, every choice a uniform random option; 370 deaths against 45 |
| Campaign from MAP01, 1 h (2026-10-29) | skill 3: reached MAP03, MAP04; skill 1: MAP05, MAP03 | pistol start on every map; at skill 1 MAP01–04 with no deaths |
| Campaign with inventory carry-over (eval3, same seeds) | skill 3: reached MAP03, MAP02; skill 1: MAP05, MAP05 | a death still restarts from a pistol start; within the noise of 2 seeds |

Route knowledge, measured statically with `doom/route_solver.py` (keys, generic linedef specials, lifts
both ways, no monsters): 12 of 17 Freedoom development maps and 8 of 10 Obsidian development maps solve
from the start.

**Speed was solved on day one; play was not.** The Neural Engine is fast enough for real time at
about 2 decisions per second with two subrequests each; a server is busy about a quarter of the time.

## What worked

1. **Moving everything on the hot path to the ANE.** The encoder plus heads on the ANE, a merged
   multifunction package (5.3 → 0.76 GB) and skipping one-option questions brought a decision from
   612 ms to about 206 ms in-process. `coreml_cache.py` stopped a leak of temp model copies (about 300 GB
   in three days).
2. **Fixing executor and model traps, not the model.** Most failures on MAP01–03 were the executor and
   Laya trapping each other: door/lift ping-pong, pressing a switch from behind, a no-progress
   watchdog. Together with the planner's facts and filters (v3 → v6), MAP01–03 went from about 7/12 to
   10/12 on the same four seeds.
3. **A System 1 planner over code-listed objectives.** Code lists the reachable objectives, with
   what-ifs of what each opens two steps ahead; Laya picks one on events. Zero-shot it picks the first
   step of the real route on MAP01–03 and MAP04.
4. **Generic route knowledge instead of per-map fixes.** A table of vanilla linedef specials, headroom,
   crushers, lifts called from either floor, touch-range pickups and prerequisite chaining. With these
   the whole MAP04 route connects, and the bot completes it at skill 1. MAP04's own failures shaped
   these rules (it was a development map by then), and they have not been played on a held-out map.
5. **Rewording options, checked offline first.** Laya follows an option's name and wording more than its
   conditions. Replaying logged questions with a new wording before any game caught regressions that
   whole-game batches had misattributed (the weapon text: 161/200 choices that cannot hit before a fix,
   2/200 after).
6. **Labels from the game itself.** Saved engine states plus deterministic rollouts give outcome labels
   for any choice, with no human data and no hosted model. A movement head trained on the outcomes of
   7,056 states (5,694 training, 1,362 validation, four movements each) beat the original on validation
   regret: 1.92 at its best epoch, chosen on that same set, against 2.24 for doomLaya's head and 2.15 for
   the movements actually played. Its last epoch was 2.42, so the gain is optimistic.
7. **Faster experiments.** Parallel games (2 ANE servers plus identical GPU servers), short scenarios
   from saved states, stall cut-offs and pre-registered decision rules turned day-long batches into
   hours.

## What did not work

- **Zero-shot yes/no judgement questions.** Making progress? Pull back to heal? Keep going? Laya answered
  at chance (AUROC 0.36–0.52) and said yes 70–95 % of the time regardless of the facts.
- **Free-text hints to the command head.** "Hidden areas opened nearby…" changed no command decision in
  463 replays: the head reads its own projection of the state and a numeric residual.
- **Avoidance without a safe executor.** A fight/back-off/heal question worked as a decision, but
  `retreat` walks backwards without firing, and runs with back-off died more.
- **Forced holds, healing detours, extra planner versions (v7, v8), always-run, combat facts in the
  movement question.** None of these beat the baseline in games, and several were worse.
- **An outcome-trained movement head in games.** In scenarios it cut deaths 18 → 13 on 44 pre-death
  states (9 vs 4 differing states, p ≈ 0.27, from the runs its epoch was chosen on). In five full MAP04 games deaths were unchanged (28, as before). The scenarios start inside
  fights that are already being lost; which fights to take, and when to break off, is a command choice.

## Lessons

1. **Single runs mislead.** Trajectories are chaotic: one different decision, or a different decision
   time, sends a run down another path. The same seed gave 0, 6 and 4 deaths on repeats. Variants that
   differ in 2 of 2,206 decisions swapped wins and timeouts. Rank variants on several seeds in the same
   session, or on paired offline replays and scenarios.
2. **The numeric residuals often decide, not the language model.** On MAP03 a command residual turned 347
   of Laya's `pickup` answers for a reachable red key into `use_switch`, and looped for ten minutes. It
   changes the command in 21–50 % of decisions. Changes to text only reach the decisions the residual
   leaves alone.
3. **Wording is a weapon.** Small changes to option text moved choices by tens of percent: "Ammo: N" vs
   "Shots left: N"; melee options listing ammo; listing melee first. The heads were trained on one
   wording; changing it is a distribution shift, so test it offline before playing.
4. **Generic route rules work, combat survival is the wall.** With generic rules the bot finds MAP04's
   route without anything map-specific in the code's logic. They did not transfer: on six held-out maps
   they gave no more keys or doors than before (heldout2, 0/18), and at skill 1, where the bot mostly
   survives, still no exit (heldout3, 0/18). At skill 3, a MAP04 game needs about zero deaths:
   the route takes 500–700 s, and each death restarts it.
5. **Development gains did not transfer to held-out maps.** Our stack and clean doomLaya v0.3.1 were
   level on six held-out maps (0/18 each). The code is still not the whole player: with random choices
   in place of Laya's, the same stack finished none of 12 development-map games. Keep a held-out set from the start and play it only with a
   frozen stack.
6. **Offline first, then decide before the batch.** The fastest progress came from replays and rollouts
   with a stated rule for what continues and what stops an experiment.

## Where it stopped

On 2026-10-25 MAP04 at skill 3 had no completion in 79 runs. The park rule in `docs/LAB_NOTEBOOK.md` (no skill-3
MAP04 completion by 2026-10-28) was met early. The last lever tested, command-level rollouts (next section), found no command that beats Laya's beyond the noise, so the project is parked here.

### Command rollouts: no command beats Laya's beyond the noise

From 209 fight states up to 25 s before a death (MAP02–04, 7 runs, frozen exp16, GPU servers), the first
command was forced to each offered option and the unchanged stack then played 15 s
(`doom/scenarios.py --commands`, `agent.py --force-first-command`; 1,395 rollouts in about an hour,
`results/doom/command-rollouts-v1.jsonl`). These runs fail the authority audit by design; they were
meant as labels for the command head, never as results. Loss is health lost plus 100 per death.

| First command | States | Deaths | Mean loss | Survives where Laya's dies / reverse | Sign test p |
|---|---|---|---|---|---|
| Laya's own choice | 209 | 71 | 56.9 | – | – |
| wait | 209 | 57 | 49.1 | 34 / 20 | 0.08 |
| explore | 209 | 66 | 53.2 | 32 / 27 | 0.60 |
| attack | 157 | 53 | 56.2 | 21 / 16 | 0.51 |
| pickup | 144 | 49 | 56.0 | 23 / 18 | 0.53 |
| use_switch | 101 | 29 | 52.7 | 12 / 12 | 1.00 |
| retreat | 157 | 62 | 62.6 | 22 / 26 | 0.67 |
| look_back | 209 | 79 | 61.4 | 24 / 32 | 0.35 |

- **The outcome is mostly noise at this horizon.** Replaying Laya's own command as a forced command
  changed whether the player died in 43 of 207 states (21 %). "Another command survived" in 54 of
  Laya's 71 deaths, but with seven noisy alternatives that is what chance gives; best-of-seven loss
  (15.4) is not reachable.
- **No alternative is clearly better.** Only `wait` comes close (14 fewer deaths, p = 0.08), and in
  the same direction on every map (MAP02 24 → 20, MAP03 22 → 17, MAP04 25 → 20). Retreating and looking
  back are worse, as the executor's `retreat` walks back without firing.
- **So there is little to learn here.** Labels from single rollouts would teach the command head
  mostly noise, and even a real 20 % cut in these windows did not carry into games for movement
  (scenarios 18 → 13 deaths, games unchanged). With the deterministic movement rollouts the signal was
  clear; at the command level, 15 s later, it is not.

The approach — a typed System 1 choosing among code-listed commands, with a scripted executor — is at
its limit at skill 3 on MAP04. Doing better needs a better executor in fights (fall back while firing,
use cover) or a policy trained on far more game experience, not another question or wording.

### Open competitors zero-shot

OpenDecider-nano (400M encoder) and OpenDecider-small (4B decoder), both Apache-2.0, answered the same
Doom questions with no Doom training (`doom/opendecider_check.py`). On 1,362 outcome-labelled fight
states their movement regret was 2.74 and 2.63, against 2.24 for doomLaya's Doom-trained Laya head, 1.92
for our outcome-trained head and 3.49 for random. On 300 logged weapon questions they chose a weapon
that cannot hit 95 and 78 times, against Laya's 6. A better general System 1 is not a substitute for
Doom training, so swapping models is not a way past the limit either.

## After the park: held-out maps again, and campaigns

Two questions were left open: do the generic route rules transfer, and how far does the final stack get
in a campaign? Both were pre-registered and played from frozen snapshots on one ANE server.

- **heldout2** (the six held-out maps, new seeds): 0/18, with 7 keys and 509 doors against 8 and 496
  before the route rules. The rules shaped on MAP04 did not transfer. MAP11 and MAP15 lost real time in
  all six games (the agent loop, not the model server, is slower on those maps), so they have no result.
- **Campaigns** (one hour from MAP01, `repeat_map.py --campaign`): with a pistol start on every map
  (eval2), skill 3 reached MAP03 and MAP04, and skill 1 MAP05 and MAP03. With inventory carried to the
  next map as in vanilla Doom (`--carry-inventory`, eval3, same seeds), skill 3 reached MAP03 and MAP02,
  and skill 1 MAP05 twice, with no deaths on MAP01–04. Carry-over cannot help where the bot dies: a death
  restarts the map from a pistol start. On 2 seeds per skill the difference is within the noise.
- Still no MAP04 completion at skill 3: none in 79 single-map runs and 4 campaigns.
- **Random baseline** (eval4): the same stack with every choice answer replaced by a uniform random
  option finished 0 of 12 games on MAP01–03 and died 370 times; Laya, in the same session, finished 7 and
  died 45 times. The audit flags every replaced choice, as it should. The model matters: the code around
  it makes good choices playable but does not make them.
- **heldout3** (the six held-out maps at skill 1, heldout2's seeds): 0/18. On MAP05, MAP07 and MAP09 the
  bot survived (0–3 deaths per game), collected keys in 8 of 9 games and still found no exit; on MAP13 it
  died 7–10 times per game even at skill 1. With survival mostly taken out, the route is what fails on
  unseen maps. MAP11 and MAP15 lost real time again and have no result.

## Level 1 pixel perception (2026-10-31 to 2026-11-07)

Every result above takes enemies and items from ViZDoom's labels buffer: only what is on screen, but with exact
names, ids, positions and outlines. The question of this phase: what does it cost to find them from the screen
instead? Level 1 replaces the enemy and item lists with a detector that sees pixels and the depth buffer (agent
flag `--pixel-perception DIR`, `doomlib/pixel_perception.py` in the patch, tooling in `doom/vision/`). The map,
doors, switches and keys known from the WAD and the object list, the counters and the projectile facts stay as
they were, so this measures perception alone and is not a human-fair game. Every experiment was pre-registered
in `docs/LAB_NOTEBOOK.md` with its reading fixed beforehand; the prior work and the backlog are in
`docs/PIXEL_BACKLOG.md`.

### Setup

- **Data** (`doom/vision/collect_frames.py`, `collect_sequences.py`): frames from the 17 development maps with
  the labels buffer as ground truth, warped to random points near monsters and items, plus plain scenery, plus
  views along the paths of recorded games (300 per map on MAP01–03, and tic-by-tic sequences for the runtime
  replay). MAP01–04 are validation maps and never in the detector's training; held-out maps are never touched.
  No hosted model.
- **Model** (`train_seg.py`, `export_seg.py`): MobileCLIP-S2's image trunk, fine-tuned, joined with a small CNN
  on the depth buffer, names every 4x4 cell of the 192x320 view above the status bar (48x80 cells) with one of
  49 classes. Core ML, 2.1–2.2 ms per frame on the Neural Engine, 99.9 % cell agreement with torch; 60–200
  minutes of training on the GPU per model.
- **Runtime:** touching cells of one category at one depth form an object named by its most common class;
  bearing from the column, distance from the depth buffer along that ray (times 1.07, the median true-to-ray
  ratio on development maps); ids from a position tracker; reported after 2 consecutive tics; remembered for
  70 tics after the last sighting.
- **Offline tests that stood in for play:** views with a false monster along the agent's paths (walked views),
  phantom tracks that persist over consecutive frames, and the runtime replay (`runtime_replay.py`): the stack's
  own perception and memory fed tic by tic and compared with the engine's monsters (phantoms, duplicates and
  real monsters listed, per frame). Models are compared at a matched operating point: each monster threshold is
  set to the same recall on MAP04.
- **Gate for every candidate:** MAP01, skill 3, 180 s, seeds 54, 1, 2, 3, `planner_ext_safe` with the pixel
  flags; passes with 2 or more finished and at most 2 deaths. Engine perception on the same stack: 3/4, then
  4/4 with re-plan on gains. Before a gate, an offline rule; a miss there meant no gate.

### The detectors

| Model | Change | Walked views with a false monster, matched recall | Monster object recall, validation frames |
|---|---|---|---|
| pix1 | MobileCLIP-S2 trunk, RGB | 13.3 % | 71.6 % |
| pix2 | empty cells weighted like objects | 17.8 % | 50.5 % |
| pix3 | the depth buffer as a fourth channel | **5.1 %** | 77.3 % |
| pix4 | classes for corpses, barrels and decorations | 9.7 % | 80.7 % |
| pix5 | pix3 plus on-policy frames from the agent's own games (DAgger, one round) | 6.5 % | on pix3's curve |
| pix6 | pix5 plus item hard negatives | 7.2 % | fewer false items on MAP03 only |

pix3 is the detector of every game after pix1. Depth did what it was for: phantoms on bare walls and floors fell
from 59 to 25 on the walked views. What remains are sprites that are not monsters (barrels, corpses, armour
bonuses, lamps), which stand out in depth as monsters do, and at a few poses scenery read as high-value items
(on MAP01 the cells called BlueCard lie on nothing in 87 % of the suspect views).

### Results

| Date | Candidate | Offline rule | Gate (MAP01, 4 seeds) |
|---|---|---|---|
| 10-31 | pix1, RGB detector | chosen on recall | 0/4; MAP01–03 0/9 against engine 6/12, one server crash on option length |
| 10-31 | pix2, pix3, pix4 | all miss (walked-view phantoms above 3.3 %) | none |
| 11-01 | matched operating point, phantom persistence | pix3 best at matched recall | measurement |
| 11-01 | forgetting on negative evidence (1c) | misses (16–25 % removed; rule 50 %) | |
| 11-01/02 | re-identification by a seen enemy (1d, 1e) | miss (real kept 63 %, 71 %; rule 90 %) | |
| 11-02 | short gate, pix3 plain against pix3 + 1c + 1e | | plain fails; forgetting passes (MAP03: 28 regions, 50 kills) |
| 11-02 | pix3 + 1c + 1e, full comparison | | MAP01–03 0/12 against 6/12; MAP01 0/4 with 600 s |
| 11-02 | item spots (planner: a failed item's spot keeps its history) | | 0/4 |
| 11-03 | memory admission by track length (K = 4) | no K passes; K = 4 gated as registered | 0/4 |
| 11-03 | depth-tolerant merge of sprite pieces | 0.5 m misses by 0.003 extras per frame | 0/4 (gated later on new evidence) |
| 11-04 | re-plan only on gains (planner) | | pixel 1/4; engine 4/4, kept |
| 11-04 | pix5, on-policy frames | misses (phantoms 124 % / 116 % of pix3's; rule 70 %) | |
| 11-04 | negative-evidence range 20–30 m | no change: the spots are behind walls | |
| 11-04 | memory facts in the enemy question | withdrawn: the server's projection drops added facts | |
| 11-04 | memory window 35 / 17 tics | | 1/4; 0/4 |
| 11-06 | command residual distilled from engine-perception choices | misses (+0.6 points; rule 10) | |
| 11-07 | track belief from WAD geometry (MLP, AUC 0.87) | misses narrowly (48.4 % phantoms removed; rule 50 %) | |
| 11-07 | oracle memory filter from engine labels (diagnostic) | | 1/4: the memory is not the main bottleneck |
| 11-07 | item confidence threshold | misses (false items 69 % / 57 % of argmax; rule 50 %) | |
| 11-07 | item belief from geometry; pix5's items | miss (45 % removed; rule 60 %) and (92 % / 86 %; rule 50 %) | |
| 11-07 | pix6, item hard negatives | misses (93 % / 63 %; false monsters 7.2 % against 6.1 %) | |

About 110 pixel games were played. All passed the authority audit; six lost real time and have no result (one
pix1 MAP03 game, the four on-policy games on MAP10 and MAP12, one MAP03 data-collection game).

### What worked

1. **Depth as an input.** The one detector change that moved precision: pix3 at matched recall has a third of
   pix1's walked-view phantoms and almost none on the MAP03 path where pix1 stalled.
2. **Testing along the agent's own paths, and through its own memory.** Per-frame false-object rates over random
   views (pix1's 0.33 per frame) said nothing about play; persistent phantoms on walked paths, and the runtime
   replay of perception plus memory, explained each failure before the next game. The replay runs a sequence in
   about 2 s once the detections are cached, so parameter sweeps take minutes.
3. **Forgetting on negative evidence and re-identification.** They missed their offline rule and passed the
   short closed-loop gate: forgetting junk entries matters more than the remembered real monsters it loses. They
   are on in pixel mode.
4. **Re-plan only on gains.** Found through the pixel games (they last longer, so the door-closing loop showed),
   it took engine MAP01 from 3/4 to 4/4 and is the planner baseline for both modes.
5. **Diagnostics that localised the loss:** blind attacks (84 % of attack tics at enemies not in view), pickup
   time (61 % on item classes not on the map), item confusion at the suspect poses (scenery, not misnamed items).
   Each took a few hours offline on recorded games.

### What did not work

- **Retraining the detector** on more scenery (pix2), with distractor classes (pix4), on the agent's own views
  (pix5) or with item hard negatives (pix6). Each moved along one recall-against-phantoms curve or below it.
  Textures a new map brings have never been seen, and MAP01's phantom keys are of that kind.
- **Filtering the memory.** Track length, class consistency, a depth merge, a longer range for negative evidence,
  a shorter window, and learned real-or-phantom scorers from geometry: real monsters are often seen for only a
  few tics, so every filter that removed phantoms removed them too, and the oracle shows that even a perfect one
  gives 1/4.
- **Adapting the decision.** The distilled command residual could not fit its own training data (74 %): the pixel
  packet carries the same counts and distances for a real remembered enemy and a phantom.
- **Planner-side item history** (item spots): new detections land more than 128 units from earlier spots, so the
  menu refilled with "Not tried yet" weapons and keys.

### Lessons

1. **Phantoms cost more than misses, and persistent ones most of all.** A missed monster costs a shot; one phantom
   in a fixed view holds a goal-directed agent in a fight with a wall for the whole game.
2. **Score perception through the agent's memory on its own paths.** Frame metrics on random views misled twice
   (pix1's 0.33 false objects per frame; pix4's best recall with the worst phantoms).
3. **A decision head trained on perfect perception trusts what it is told.** The command head attacks remembered
   pixel enemies as it learned to attack remembered engine enemies, which were few and real, and it chases a
   phantom blue key because keys are what the planner values most. Nothing downstream can recover what the
   detector's output does not carry.
4. **Removing one false source moves the time to the other.** A shorter memory cut blind attacks and the time went
   to pixel pickups, not to switches and doors.
5. **Gates of four seeds separate 0 from 4, not 1 from 0.** 1/4 against 0/4 recurred and is noise.

### Mechanisms left in the code, and their defaults

| Switch | Where | Default | Why |
|---|---|---|---|
| Confirmation on 2 consecutive tics; unnamed weapons and keys not reported | `pixel_perception.py` | on | pix1's false-object rate, before any game |
| Monster threshold at matched MAP04 recall (0.7883 for pix3) | `classes.json` | on | matched operating point |
| Item threshold | `classes.json` | unset | item operating point missed |
| Forgetting on negative evidence and re-identification (1c, 1e); `--pixel-memory-plain` turns them off | `agent.py` | on | short gate |
| `REMEMBER_SEEN_TICKS` | `agent.py` | 0 | K = 4 gate 0/4 |
| `MERGE_DEPTH_M` | `pixel_perception.py` | 0 | gate 0/4 |
| `NEGATIVE_EVIDENCE_M` | `agent.py` | 15 | no range changed the replay |
| `--pixel-enemy-memory-ticks` | `agent.py` | 70 | 35 and 17 failed |
| `--pixel-oracle-filter`, `--shadow-engine-packet` | `agent.py` | off | diagnostics and data collection that read engine labels |
| `--command-residual`, config `planner_ext_safe_distill` | server, `repeat_map.py` | off | distillation missed |
| `doomlib/track_belief.py` (enemy and item belief, with tests) | patch | not wired to a flag | both scorers missed |
| Re-plan on gains | `doom/planner.py` | on, both modes | engine 4/4 |

### Where it stands

Level 1 is open. From pixels MAP01 finishes in 1 of 4 games at best, and MAP02 and MAP03 in none. The chain of
evidence points at the detector: about a third of confirmed, visible enemy sightings match a distinct real
monster, the rest are phantoms or fragments, and at a few poses scenery is read as the items the planner values
most. Filters, planner changes and re-weighting the decision have each been tried and cannot recover that. The
next lever is the detector's output itself: its precision on sprites that are not monsters and on item classes,
and what each report carries about its own reliability.

## Open directions

- **Train the command head on game outcomes** (expert iteration) only with repeated rollouts per
  command (single rollouts are 21 % noise in the death outcome); `wait` is the one candidate.
- **A fall-back-while-firing executor action**, so that an avoid decision can be carried out safely.
- **Enemy facts in the option text** (hit chance by distance, melee-only monsters, hitscan shooters),
  tested offline first.
- **An outcome-prediction head** ("will this fight be won?"), trained on the same labels, to drive
  engage/avoid.
- **Find why the agent loop loses real time on some maps** (MAP11 and MAP15 in heldout2 and heldout3), on development
  maps only.
- **A stricter "only what has been seen" map mode**, since the current route knowledge is like a full
  automap that also says what every line does.
- **A calibrated zero-shot baseline for the judgement questions** (added 2026-11-08): WaterSheep, a
  ModernBERT-base decision model with temperature-calibrated yes/no answers that serves the Jev request format
  locally (Apache-2.0, github.com/SamratDuttaOfficial/WaterSheep). Ask it the same "pull back?" question on the
  held-out states where Laya's noul head scored 0.53 and the trained judgement head 0.76 (poc/judgement/), and
  report its AUROC and calibration beside them. Not Doom-trained, so a baseline, not a candidate.
- **A Doom-tuned small decision model with calibrated nouls, if phase 3 shows judgement matters** (added
  2026-11-08): the LLM-as-Jev recipe (Li and Wagle, Microsoft, arXiv 2610.02076) scores a frozen causal LLM over
  a trie of candidate suffixes and fine-tunes compact models with a listwise loss and KL anchors; its finding
  that fine-tuning pays off for small models in specialised domains matches our OpenDecider and Qwen results.
  A Qwen3-0.6B tuned this way on our outcome labels would keep calibrated yes/no answers and could take
  multimodal input. Needs an MLX or Core ML path; the released code is CUDA-only.

For level 1:

- **Bound the gap before the next fix.** Engine perception gives 4/4 and plain pixels 1/4. Step down from engine one
  error source at a time (item classes with a class-matched, tight radius; visible enemies; positions) to find where
  the completions are lost. The item oracle so far matched by position only and let scenery near real items pass.
- **Detector precision rather than more filters:** motion input, a larger input or a box detector (backlog items
  5–7), and texture diversity from generated development maps (`doom/oblige_maps.py`) rather than more frames of
  the same 13 maps, since the phantom keys on MAP01 are textures the detector never saw.
- **Shot feedback as a fact** (backlog item 9), for the model to weigh.
- **Real time on crowded maps:** the detector runs every tic at 2.2–3.2 ms, 8–11 % of the tic budget, and MAP10
  and MAP12 lost real time in pixel games.
- **Gate power:** eight seeds, or a pre-registered continuous metric (distance to the exit at timeout, share of tics
  on route actions), since 1/4 against 0/4 cannot be read.

## Reproducing

- Stack: `ext/doomLaya-dev` (doomLaya v0.3.1 plus `doom/patches/doomLaya-dev.patch`), as of 2026-10-29;
  frozen copies with `doom/snapshot_stack.sh`. Earlier batches ran on earlier code (see the README,
  "Reproducing exact numbers").
- Games: `doom/repeat_map.py` (results in `results/doom/repeat_<map>_<suffix>.json`; campaigns with
  `--campaign SECONDS`, inventory carry-over with `--extra-flags='--carry-inventory'`); scenarios:
  `doom/scenarios.py`; movement rollouts and training: `doom/fight_rollouts.py`,
  `training/finetune_combat_movement.py`.
- Held-out maps: `doom/map_split.json`; played only from `ext/snap/` with `--evaluation`.
- Pixel perception (level 1): `doom/vision/collect_frames.py` (random warps, `--scenery`, `--from-runs`) and
  `collect_sequences.py`; `train_seg.py --depth`; `export_seg.py`; `operating_point.py` for the monster threshold;
  games with `repeat_map.py ... --extra-flags='--pixel-perception <model dir>'`; `runtime_replay.py` for the
  offline replay of perception plus memory. The detector weights and Core ML packages (`results/vision/*/model.pt`,
  `*.mlpackage`) and the frame sets (`ext/vision-data/`) are not in the repository; reports, operating points,
  replays and belief scorers are.
