# RL Doom agent repos: second-pass code review (2026-10-24)

Scope: combat/survival and navigation ideas our bot (doomLaya-dev + Laya typed choices) missed.
Source read with grep/cat from shallow clones in
a scratch directory. Nothing was run or installed.

## Licences

| Repo | Licence | Notes |
|---|---|---|
| isl-org/DirectFuturePrediction | MIT (Intel 2017) | TF1 code; IntelAct competition agent itself is not in this repo |
| glample/Arnold | **none** (no LICENSE file) | read for ideas only; do not copy code |
| akolishchak/doom-net-pytorch | MIT | |
| alex-petrenko/sample-factory | MIT | |
| TTomilin/LevDoom, HASARD, COOM | MIT (each) | scenario WADs run on freedoom2 (no IWAD path set in `envs/base.py`) |
| RohanNaga/Doom | **none** | GameNGen-style world model; records Arnold play; not a control agent |

## Finding in our own code first

`ext/doomLaya-dev/agent.py:55` BUTTONS has no `SPEED`, `agent.py:378` adds no `+cl_run 1`, and
`_vizdoom.ini:240` is `cl_run=false`. `doomlib/executor.py:184-186` and `doom/fight_rollouts.py:77`
press binary MOVE_LEFT/RIGHT/BACKWARD. **The bot walks: every strafe, backpedal and retreat runs at walking
speed.** Doom's `g_game.c` tables: forwardmove {25, 50}, sidemove {24, 40} (walk, run), so running strafes
about 1.7x faster and backpedals 2x faster. All three combat repos that dodge use run:
Arnold `run.sh` `--speed "on"` for both competition tracks; sample-factory `action_space.py`
`doom_action_space_full_discretized()` has a "sprint" dimension; LevDoom
`envs/dodge_projectiles/scenario.py` `get_available_actions()` is exactly {none, left, right} x {walk, run};
HASARD `utils/action_space.py` `doom_turn_move_jump_speed()`.
The 42-tic fight labels were all collected at walking speed too, so they describe a slower player.

## 1. DirectFuturePrediction (DFP)

Files: `examples/D3_battle/run_exp.py`, `DFP/future_target_maker.py` `FutureTargetMaker.make_targets`,
`DFP/agent.py` `prepare_controls_and_actions`, `DFP/future_predictor_agent_advantage.py` `act_net`,
`maps/D3_battle.cfg`.

- Measurements: `available_game_variables = {AMMO2 HEALTH USER2}` (ammo, health, kills from ACS).
  Only three numbers, no position, no enemy list.
- Targets: **differences** meas(t+k) - meas(t) for k in `future_steps = [1,2,4,8,16,32]` decisions,
  `frame_skip = 4`, i.e. 4, 8, 16, 32, 64, 128 tics. Six horizons for each measurement, 18 outputs per action.
- Objective: `objective_coeffs_temporal = [0,0,0,0.5,0.5,1]` (only 32, 64, 128 tics count, longest double),
  `objective_coeffs_meas = [0.5, 0.5, 1]` (ammo, health, kills). Scales for normalisation `[7.5, 30, 1]`
  (ammo/7.5, health/30, kills/1). Action = argmax over actions of sum(coeff x predicted change).
- Death: `invalid_targets_replacement='nan'`: targets past the end of an episode are NaN and ignored
  in `mse_ignore_nans`. There is **no explicit death penalty**; health falling is what it learns to avoid.
- Test-time goal change: `D3_battle_randobj_pos[_neg]` set `random_objective_coeffs=True`
  (`agent.py` `reset_objective_coeffs`, uniform positive or +/-), the goal vector is a network input,
  and any goal can be plugged in at test time without retraining. This is the key DFP idea for us:
  **store a vector outcome, choose the weights later.**
- Action space: `discrete_controls` = all 2^8 combinations of MOVE_FORWARD, MOVE_BACKWARD, MOVE_RIGHT,
  MOVE_LEFT, TURN_LEFT, TURN_RIGHT, ATTACK, SPEED (`opposite_button_pairs` empty in D3/D4, so 256 actions,
  including forward+strafe+turn+run together).
- Health/ammo pickups: nothing special. The health and ammo terms in the objective make walking over
  medkits/ammo positive; D2_navigation uses HEALTH as the only measurement and learns pickups from it.
- Generalisation: `D4-tx_battle2_99maps` trains over 89 random-texture maps (`switch_maps=True`).

Compatible with our rules: yes for the label side. We already save the state and roll out each option; DFP
says record the measurement trajectory, not one scalar.

## 2. Arnold (Lample and Chaplot)

Files: `run.sh` (track1/track2 configs), `src/doom/actions.py` `create_action_set`, `src/doom/game.py`
`make_action`, `update_statistics_and_reward`, `WEAPONS_PREFERENCES`, `src/doom/reward.py`,
`src/doom/utils.py` (game features), `src/doom/labels.py`.

- Action space, track2: `"move_fb+move_lr;turn_lr;attack"`: one of {none, F, B, L, R} x {none, turn L,
  turn R} x {fire, no fire} = 29 actions; track1: `"attack+move_lr;turn_lr;move_fb"`. Movement and turn are
  always chosen together (no diagonals in track2). `--speed on` (always run), `--frame_skip 3` or 4
  (decision every 3-4 tics, ~10 Hz), LSTM with `hist_size 4`.
- Game features: `--game_features "target,enemy"`: two binary auxiliary targets, "enemy under crosshair"
  and "enemy visible", read from an ACS bitmask in `GameVariable.USER1` (`utils.py`). Only used as an
  auxiliary loss for the DQN; we already give enemy visibility as a fact.
- Separate navigation/action networks: **not in the released code**. The scenarios run one network;
  the paper's switch (action net when an enemy is visible, else navigation net) is absent.
- Rewards (`reward.py` defaults): KILL +5, DEATH -5, SUICIDE -5, MEDIKIT +1, ARMOR +1, INJURED -1
  (per event, not per HP), WEAPON +1, AMMO +1, USE_AMMO -0.2.
- Code makes choices (would break our rule): `make_action` forces the favourite weapon
  (`WEAPONS_PREFERENCES`: BFG(>=40 cells) > shotgun > chaingun > plasma > pistol > rocket); anti-stuck
  `manual_control`: after 30 decisions without forward or 60 without a turn, it runs TURN_RIGHT+SPEED
  (+FORWARD) for 40 tics. Both could be options for the model instead.
- No dodge logic and no respawn logic beyond the 10 s respawn constant.

## 3. doom-net-pytorch (CIG 2018 single-player runner-up)

Files: `src/bt_doom.py` (behaviour tree), `src/bt_doom_actions.py`, `src/planner.py`,
`src/doom_instance_oblige.py`, `src/mcts*.py`, `src/simulator.py`.

- Behaviour tree goals (`goal_defs`): attack only if `enough_health` (HP > 30 or no health item in view)
  and `enough_ammo`; otherwise `pick_health` first. Health is picked whenever visible and HP < 100.
  This is the "low HP: get health before pressing on" rule our bot lacks as an explicit option/fact.
- `Attack.run`: shoot at the angle of the enemy, then a random 100-unit lateral step left or right
  each time (unpredictable sidestep).
- `Planner.get_round_view`: every 20 steps turn 4 x 90 degrees to build a full panorama (look around),
  then head for the exit sign / farthest depth column.
- `doom_instance_oblige.py` `step_normalized`: reward = decrease in path distance to exit, -1 if no
  progress, +10 finish, -10 death. Trained on Oblige maps.
- MCTS/simulator (`simulator.py`) is a stub (`get_next_state`, `rollout` pass); no usable planning code.

## 4. sample-factory ViZDoom

Files: `sf_examples/vizdoom/doom/wrappers/reward_shaping.py`, `action_space.py`,
`wrappers/additional_input.py`, `wrappers/exploration.py`, `doom_utils.py` (env specs).

- `REWARD_SHAPING_DEATHMATCH_V0` (per unit change): FRAGCOUNT +1/-1.5, DEATHCOUNT -0.75,
  DAMAGECOUNT +0.003, HEALTH **+0.005 gain / -0.003 loss**, ARMOR +0.005/-0.001, weapons and ammo by
  `WEAPON_PREFERENCE`, small reward for keeping one weapon selected (stops weapon thrash).
  `reward_delta_limits = dict(DAMAGECOUNT=200, HITCOUNT=5)`: damage dealt is capped per step so
  splash/BFG does not dominate. In V1 HEALTH is +/-0.01. Health is worth about the same as dealt damage.
- doom_battle / battle2: `doom_action_space_discretized_no_weap()` = {none,F,B} x {none,R,L} x attack x
  sprint x 11 turn bins (-10..10); duel/deathmatch: 21 turn bins +/-12.5 plus weapon select and USE.
  Every combat action is factored move x strafe x turn x run.
- Measurements input (`additional_input.py`): weapon, ammo/15, health/30, armor/30, kills, ATTACK_READY,
  player count, per-weapon owned and ammo. ATTACK_READY (weapon can fire now) is a cheap fact we lack.
- `ExplorationWrapper`: +0.1 for reaching a pose more than 75 units (angle counts /2) from all of the
  last 200 landmarks.

## 5. LevDoom / HASARD / COOM (Tomilin)

- LevDoom `levdoom/envs/dodge_projectiles/scenario.py`: lateral-only dodging of lined-up projectile
  monsters; actions {none, L, R} x {walk, run}; reward +0.01 per frame survived or -0.1 per HP lost.
  Levels change monster type: imps (default), barons, revenants (homing), cacodemons, mancubus, arachnotron,
  flaming skulls, plus resized agent and city textures (`maps/*.wad`, 20 WADs).
  `seek_and_slay` adds a 0.001 x distance-travelled bonus. Generalisation is tested by held-out
  visual/monster modifications; training on more variations was what helped.
- HASARD `hasard/envs/*/env.py` `calculate_cost`: health loss is a separate **cost** with a per-episode
  safety budget (`_default_safety_budget`), optimised with PPO-Lagrangian/PID-Lagrangian
  (`sample_factory/algo/learning/ppo_lag_learner.py` `_update_lagrange`). Lava/precipice/volcanic scenarios
  are hazards as HP-cost floors.
- COOM `COOM/env/scenarios/hide_and_seek/hide_and_seek.py`: survive melee chasers by moving away and
  grabbing medkits at low health; dense reward +5 per health gained, -5 per health lost, +movement.
- No repo exposes a damage direction. None tracks unseen attackers.

## 6. RohanNaga/Doom

DoomDiT world model trained on Arnold recordings (`record_arnold.py` reproduces Arnold's weapon and
anti-stuck preamble, `third_party/arnold-torch2.patch` ports Arnold to torch 2). Useful only as a way to
run Arnold as a baseline or data source; no licence.

## Unseen attackers and damage direction

Nothing in these repos. Arnold and DFP rely on recurrence and on turning being part of every action.
What we have: `doomlib/look_questions.py` `DamageHistory` + a fixed 180-degree `look_back` command.
Ways to give direction honestly, listing facts only:
1. Last-known bearing/distance of every monster seen before that is not visible now (memory), with
   the age of the sighting.
2. Engine objects list (already used for projectile facts): awake monsters outside the view cone that
   have line of sight to the player (our navigator geometry can check LOS), as "possible attacker,
   bearing B, distance D, type T, hitscanner yes/no". This is the same kind of engine fact as
   "incoming projectile, closest approach".
3. ViZDoom audio buffer (stereo) gives left/right loudness of gunfire; ambiguous front/back.
Then offer turn options: face candidate k, turn 90 left, turn 90 right, look_back, and label them by
rollouts like the movement question (score after 42-70 tics).

## Ranked ideas (expected impact on deaths per 20 min)

1. **Run while dodging and retreating** (SPEED held or `+cl_run 1`). About 1.7x strafe speed and 2x
   backpedal. It goes straight at the ~30% projectile damage and the retreats. Code-side execution
   detail (allowed), or offer run/walk as a model option. Re-collect the fight labels with run on.
   Check it first: from 20 saved states, measure 42-tic displacement walk vs run.
   Cost: one button. Data: relabel the existing states.
2. **Command-level outcome labels at low HP** (the red-key to blue-key deaths). At command questions
   with HP < 50, save state and roll out the top 2-4 commands (pickup nearest health/armor, retreat,
   explore toward key, wait) for 10-20 s with the frozen model playing on, then score survival + progress.
   doom-net's `enough_health` gate (HP > 30 or no health in view) is the rule to learn, but from labels,
   not code. Give facts: nearest known health/armor item, distance and amount. Hundreds of states enough
   because the decision is low-dimensional.
3. **Unseen-attacker facts + directed turn options** (13-31% of damage). Facts as in the section above;
   options face-candidate / 90L / 90R / 180 / ignore; rollout labels on states where HP dropped with no
   visible attacker or projectile. Extends `look_back` from a blind 180 to a bearing.
4. **DFP-style vector outcomes and multi-horizon scoring**. In the same rollouts record health, armor,
   dealt damage, alive at 8, 16, 32, 64, 128 tics (and continue past 42 tics with the frozen policy).
   Pick weights and horizon by validation regret, DFP's [0,0,0,0.5,0.5,1] over (32, 64, 128 tics) as a
   starting point, and make the HP weight depend on current HP (HASARD budget idea; e.g. lost x 100/HP,
   or death probability over 128 tics). No relabelling needed to change the objective. Cap dealt damage
   per rollout (sample-factory caps DAMAGECOUNT at 200 per step).
5. **Bigger movement set, factored like Arnold/DFP/sample-factory**: add forward, forward-left,
   forward-right, back-left, back-right (diagonals reach cover and break line of sight), all with run.
   6-9 options x 42 tics is still cheap to roll out. Keep the aim turn in code (it already turns while
   moving, which is the move+turn combination these repos use).
6. **Curriculum/extra data from LevDoom dodge_projectiles and COOM hide_and_seek WADs** (MIT, freedoom2):
   dense projectile and melee-chase states for movement labels across monster types (revenant homing,
   mancubus spread) without map-specific code. Use as extra training states only; keep MAP04 and
   held-out maps for evaluation. Also add ATTACK_READY as a fact.

Not compatible with our rules as written: Arnold's forced weapon preference and anti-stuck macro,
doom-net's hard-coded behaviour tree (code decides). Both are fine as **options** the model can pick.
Not useful at our data budget: end-to-end DQN/APPO/PBT training (10^7-10^9 frames).
