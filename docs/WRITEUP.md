# Laya plays Doom from the Neural Engine: write-up (2026-10-30)

A summary of the Doom work in this repo, from 2026-10-18 to 2026-10-25, plus the evaluations after the
park (two more held-out evaluations, one-hour campaigns and a random-choice baseline, 2026-10-29 to 30). The lab notebook with every
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

## Reproducing

- Stack: `ext/doomLaya-dev` (doomLaya v0.3.1 plus `doom/patches/doomLaya-dev.patch`), as of 2026-10-29;
  frozen copies with `doom/snapshot_stack.sh`. Earlier batches ran on earlier code (see the README,
  "Reproducing exact numbers").
- Games: `doom/repeat_map.py` (results in `results/doom/repeat_<map>_<suffix>.json`; campaigns with
  `--campaign SECONDS`, inventory carry-over with `--extra-flags='--carry-inventory'`); scenarios:
  `doom/scenarios.py`; movement rollouts and training: `doom/fight_rollouts.py`,
  `training/finetune_combat_movement.py`.
- Held-out maps: `doom/map_split.json`; played only from `ext/snap/` with `--evaluation`.
