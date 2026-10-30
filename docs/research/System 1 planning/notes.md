# Planning with a fast learned policy ("System 1") for doomLaya (2026-10-25)

Question: how to put Laya (typed choice heads, ~100 ms per call on the ANE) inside a planning loop so that
combat survival improves, while the model keeps choosing the command, target and weapon, and labels come
only from game outcomes.

What our own logs already say (docs/LAB_NOTEBOOK.md, read before writing this), because it constrains every design:
- Deaths are mostly fights the executor loses, not bad objective choices. MAP03: median health 45 five
  seconds before death; a third had no enemy in view 1 s before death. 40 % of MAP03 deaths are one trap
  (yellow-key closets); survivors entered with median 74 HP, the dead with 54.
- More healing did not help: v8 healed after 81/99 heal-first questions, only 1/37 deaths happened while
  heading for health, and the detours cost completions.
- The fight question with "back off" made it worse (MAP03 0/4): `retreat` is "walk backwards without
  firing" (`doomlib/executor.py`: `a[1]=1`, no path, no fire). 17/22 matched deaths followed a back-off.
  So the missing thing is not the *choice* to retreat but an executable retreat that is worth choosing.
- Run-to-run variance is huge (same seed: 0, 6, 4 deaths). Any in-play A/B needs ~10+ seeds per arm, or
  a paired design on saved states.
- We already have the core machinery: `agent.py --fight-states` (ZDoom save, ~6 ms, 60 KB),
  `doom/fight_rollouts.py` (deterministic per-movement rollouts), `training/finetune_combat_movement.py`
  (residual trained on rollout regret). Everything below extends this rather than starting fresh.

---

## 1. Planning with a learned policy prior / value

**Rollout policy improvement (Tesauro & Galperin 1996).** Estimate each candidate action's value by
simulating forward with the base policy, take the best; cut base-player error rates by up to 5x in
backgammon, from random up to TD-Gammon base policies. This is the simplest "System 1 inside planning":
one step of lookahead, the policy plays on after the branch.
https://proceedings.neurips.cc/paper/1996/hash/996009f2374006606f4c0b0fda878af1-Abstract.html
(Bertsekas' later "rollout algorithm" literature generalises this; the guarantee is that one-step rollout
is never worse than the base policy in expectation.)

**Expert Iteration / "Thinking Fast and Slow" (Anthony, Tian, Barber 2017).** Explicit System 1 / System 2
framing: tree search (System 2) is the expert, the network (System 1) is the apprentice trained to imitate
the search's choices, and the improved network then guides the next search. Beat REINFORCE and MoHex in Hex.
https://arxiv.org/abs/1705.08439

**UCT-to-classification for Atari (Guo et al. 2014).** Slow offline UCT planner with emulator access
generates labels; a CNN learns to play in real time; beats DQN. Exactly our situation (savestates, fast
student) and the clearest precedent that "slow planner offline, distilled fast policy online" works.
https://proceedings.neurips.cc/paper/2014/hash/8bb88f80d334b1869781beb89f7b73be-Abstract.html

**AlphaZero / MuZero.** PUCT search with policy prior and value head; training targets are visit counts
and outcomes. https://arxiv.org/abs/1712.01815 , https://arxiv.org/abs/1911.08265 . They need hundreds of
simulations per move; at our cost per simulation (seconds of game time) that is out of reach online.

**Gumbel AlphaZero/MuZero (Danihelka et al., ICLR 2022).** Sample actions *without replacement* from the
prior (Gumbel-top-k), allocate simulations by sequential halving, and use a policy-improvement target that
is guaranteed to improve even when not every root action is visited. Strong at 2-16 simulations, where
plain AlphaZero fails to improve its policy. This is the right recipe for our tiny budgets: pick the top
k=2-4 of Laya's own options, roll each out, halve, and train on the completed-Q target.
https://openreview.net/forum?id=bERaNdoegnO (ICLR page: https://iclr.cc/virtual/2022/poster/6418)

**Sampled MuZero (Hubert et al., ICML 2021).** Plan over a sampled subset of actions with a corrected
improvement target; matters for us because our "action" is a product (command x target x weapon x
movement), which should be sampled from Laya's heads, not enumerated. https://arxiv.org/abs/2104.06303

**Policy-guided width-based planning, π-IW (Junyent, Jonsson, Gómez, ICAPS 2019).** Rollout-IW on the
Atari simulator, with the learned policy guiding which branches to expand, and the policy trained by
supervised learning on targets extracted from the planned trajectories. Another expert-iteration loop that
works with a simulator and small budgets. https://arxiv.org/abs/1904.07091 , code https://github.com/mjunyentb/pi-IW

**AggreVaTe (Ross & Bagnell 2014).** Run the learner, stop at a random time, try one action, then let a
policy finish the episode to get a cost-to-go Q(s,a); learn to minimise cost-to-go (not 0/1 imitation).
This is exactly what `fight_rollouts.py` does, with the frozen Laya (or the executor) as the "expert"
continuation. Gives the no-regret framing for repeated rounds of data collection on the learner's own state
distribution (DAgger-style). https://arxiv.org/abs/1406.5979

Takeaway: for a classifier-style model, every successful method above reduces to **"branch on the model's
own top candidates, score by simulation, train the heads on the outcome-ranked target, repeat"**. Online
search is optional; the gain usually comes from distillation.

## 2. Hierarchical / recursive decomposition with the same model at several levels

**Options framework (Sutton, Precup, Singh 1999).** Temporally extended actions with initiation set,
policy and termination. Our planner objectives are already options; a "tactic" level would be options
of seconds-scale. https://doi.org/10.1016/S0004-3702(99)00052-1 ; option-critic learns them end to end
(https://arxiv.org/abs/1609.05140) but needs RL-scale data.

**FeUdal Networks (https://arxiv.org/abs/1703.01161), HIRO (https://arxiv.org/abs/1805.08296), Director
(Hafner et al., NeurIPS 2022, https://arxiv.org/abs/2206.04114).** Manager emits goals (latent or state
offsets), worker reaches them; trained with RL from millions of frames (Director: inside a learned world
model). Not usable with our labelling budget, but the *shape* — manager picks a goal, worker is rewarded for
reaching it, manager re-decides every k steps — is what our planner + 2 Hz loop already is.

**ADaPT (Prasad et al., NAACL Findings 2024).** Recursive "as-needed" decomposition: try to execute; only
when the executor fails, decompose that sub-task and recurse. +28/27/33 points on ALFWorld/WebShop/TextCraft.
https://arxiv.org/abs/2311.05772 . The *control flow* transfers to a classifier: the decomposition options
can be code-generated text options, and the same model chooses at each level. The trigger "decompose only
on failure" maps to our "re-plan on failure / no progress", and suggests inserting a tactic level **only
when a fight starts going badly** rather than always.

**SayCan (Ahn/Ichter et al. 2022).** Score = LLM usefulness x learned affordance (value function: will this
skill succeed from here?). https://arxiv.org/abs/2204.01691 . The best fit for us of all the LLM-agent work:
both factors are *scores over a fixed option list*, which is what Laya's heads produce. An affordance /
outcome head on Laya's encoder ("if I take this option, do I survive the next N s?"), trained from rollouts,
is a SayCan value function that stays inside the model.

**Plan-and-Act (https://arxiv.org/abs/2503.09572), DEPS (https://arxiv.org/abs/2302.01560), Voyager
(https://arxiv.org/abs/2305.16291).** Planner/executor split, re-planning from failure descriptions, and a
skill library of verified programs. These rely on generation (plans, code, explanations). Transferable
pieces: (a) the failure description fed back into the next choice (we already exclude failed commands);
(b) Voyager's idea that skills are only added once verified in the environment — for us, a tactic option is
only offered if its executor has been shown to work.

**Tree of Thoughts (https://arxiv.org/abs/2305.10601), RAP (https://arxiv.org/abs/2305.14992), LATS
(https://arxiv.org/abs/2310.04406).** The LM proposes, evaluates and (RAP) simulates states. With Laya, the
proposal step must be code (option generation) and the evaluation must be either a learned value head or a
real rollout; LATS is the closest since it uses real environment feedback instead of a world model, and it
uses the LM as policy prior inside MCTS — i.e. AlphaZero with an LM prior. Nothing in these needs a
generative model *if* options and simulator are external, which they are for us.

Which work with a classifier-style model: options/feudal shape (yes, already), ADaPT control flow (yes),
SayCan affordance scoring (yes, best fit), LATS/MCTS with policy prior (yes, with real rollouts, budget
permitting). Not transferable: anything where the model writes the plan, the sub-goal text, code or a
self-critique (ToT/RAP proposals, Voyager code, DEPS explanations).

## 3. Search in games with savestates; rollouts as labels

- **Go-Explore (Ecoffet et al., Nature 2021).** Archive of states, *restore the emulator* to a promising
  one, explore from there; then a robustification phase distils trajectories into a policy. Solved
  Montezuma/Pitfall. The transferable trick: **return to the hard state many times** instead of replaying
  the whole map — for us, the 5-20 s before each death. https://arxiv.org/abs/2004.12919
- **Rolling Horizon Evolution (RHEA) vs MCTS in GVGAI.** Both need a forward model; RHEA evolves action
  sequences and is competitive with MCTS in real-time games at low budgets; MCTS is more robust to opponent
  model error. https://arxiv.org/abs/1802.10363 , https://arxiv.org/abs/1607.01730 . Over raw 0.5 s actions
  in Doom both need far more simulations than we can afford; over a handful of macro options they collapse
  to the "try each option, keep the best" scheme.
- **Combat-specific tactical search.** RTS work (Uriarte & Ontañón, "Combat Models for RTS Games",
  https://arxiv.org/abs/1605.05305 ; Soemers' StarCraft tactical MCTS) shows search pays off at the
  *tactic* level (which group to engage, where to fall back) with cheap abstract combat models, not at the
  frame level. Doom analogue: macro options like "fight from here / fall back to doorway X / clear room".
- **Distillation.** Guo 2014, ExIt, π-IW, AggreVaTe (above) all distil rollouts into a fast policy. In
  shooters the evidence is thin: no published ViZDoom MCTS-with-savestate agent was found (search turned
  up nothing); most ViZDoom work is model-free RL. So this is plausible but unproven for Doom combat.

Practical points for our deterministic savestates:
- Deterministic rollouts are **one sample** of a stochastic fight (monster AI, hitscan spread). A label
  from one RNG stream overfits that stream. Decorrelate by inserting 0-3 random no-op tics or a tiny view
  jitter before branching, and run 2-3 seeds per branch for high-stakes states (common random numbers
  across branches within one seed reduce variance of the *difference*, which is what the label needs).
- The continuation policy matters (AggreVaTe vs SEARN): continuing with frozen Laya gives the value of
  "this option, then my current self", which is what one round of policy improvement needs.

## 4. Doom-specific tactical layers

- **Arnold (Lample & Chaplot, AAAI 2017).** Two networks, navigation and action, switched by a learned
  "enemy visible" detector; strong in deathmatch. The tactic layer is just a mode switch. https://arxiv.org/abs/1609.05521
- **DFP / IntelAct (Dosovitskiy & Koltun, ICLR 2017).** Predict future *measurements* (health, ammo, frags)
  at several horizons for each action; act by weighting predictions with a goal vector that can change at
  test time; won the 2016 ViZDoom full-deathmatch track. Engage/retreat/heal emerges from predicted health
  vs frags with no explicit tactic code. https://arxiv.org/abs/1611.01779 . This is the most directly
  useful Doom result for us: a multi-horizon outcome head is a value function we can train from rollouts.
- **ViZDoom competitions (Wydmuch, Kempka, Jaśkowski 2018).** Entries are model-free RL from pixels; none
  plans with savestates; bots still lose to humans. https://arxiv.org/abs/1809.03470
- **LLM/VLM agents.** VideoGameBench (2025): frontier VLMs barely progress in Doom II; latency is the
  stated problem in FPS. https://arxiv.org/abs/2505.18134 . Our earlier surveys (docs/research/Doom AI
  research survey, Doom repos second pass) cover doom-mcp, lucas-moont etc.: their gains came from facts
  (attacker bearing, death briefings), not from tactical planning.

Nothing published puts an explicit engage/retreat/health tactic layer between route objectives and frame
actions in Doom with planning; DFP's goal-weighted measurement prediction is the closest learned version.

---

## Design proposals for our stack (ranked)

Common pieces:
- *Branch point*: a saved engine state where Laya answered a question.
- *Continuation*: after the branch's forced first choice is held for its commitment time, frozen Laya plays
  on with the normal planner and executor.
- *Score*: as in `fight_rollouts.py`, extended: −HP/armor lost + 0.3·dealt − 100·death + progress term
  (planner's progress measure toward the current objective), at horizons 2/5/10/20 s (DFP-style).
- *Authority*: in play, only Laya's heads choose. Rollouts are only used offline as labels, or (P4) as
  facts. Rollouts never pick a choice in play.
- Cost unit: a rollout of H game-seconds costs about H/7 s of engine time headless **plus** Laya calls at
  2 Hz (≈2·H decision rounds x ~100 ms per call x questions per round). At 7x engine speed the model
  calls, not the engine, are the bottleneck; run several headless ViZDoom processes against one ANE server
  and batch questions across them.

### P1 (rank 1). Death-anchored command-level rollouts, distilled into the command/enemy/weapon heads

*What it is:* one round of expert iteration (Tesauro rollout + AggreVaTe + Go-Explore's "return to the hard
state") at the level Laya already decides: the 2 Hz command question and its target/weapon.
- Collect: normal play with `--fight-states` extended to save at every *command* question (not only
  movement). Keep states from the 25 s before each death plus a matched random sample of fights survived.
- Branch: at each kept state, take Laya's top k=3-4 joint options (command x target x weapon), sampled
  Gumbel-top-k without replacement from its own heads (Sampled/Gumbel MuZero). Always include Laya's argmax.
  Force each for its commitment window (≈1-2 s), then let Laya play on for H=10 s.
- Label: completed-Q target from Gumbel MuZero (softmax of prior logits + scaled Q) or simply regret-weighted
  cross-entropy toward the best branch, as in `finetune_combat_movement.py`. Drop states where all branches
  are within noise.
- Laya at which level: same 2 Hz heads, as rollout prior, continuation policy and student.
- Cost: 1 state ≈ 4 branches x 10 s ≈ 40 game-s ≈ 6 s engine + ~80 decision rounds of Laya calls. Death
  sample: ~6 deaths/run x 25 s window / 0.5 s ≈ 300 states/run; subsample to ~60 states/run (every 2 s).
  2,000 states ≈ 80k game-s ≈ 3-4 h wall with 4 parallel engines. Zero extra model calls in play.
- Authority: labels are game outcomes; the student is Laya; in play nothing changes except head weights.
- Cheapest experiment (1 day): 300 states from MAP03 trap + MAP04 red→blue deaths; compute **offline regret
  of Laya's argmax** (how often a different top-4 branch survives where argmax dies). If argmax is within
  noise of the best branch in >80 % of states, command choice is not the lever — stop and go to P2. If
  not, fine-tune the command residual, then re-run the same saved states (paired, deterministic: deaths
  avoided per 100 states) before any 10-seed in-play run.

### P2 (rank 2). A "tactic" question between objective and 2 Hz, with executable maneuvers (recursive System 1)

*What it is:* the user's "recursive set of calls of System 1", done ADaPT-style: the same Laya is asked a
third, seconds-scale question **only when needed** (a fight starts, HP drops by >20 in 3 s, or a hit comes
from out of view). Code generates 3-5 tactic options from facts, each with a *working* executor:
- "Hold and fight here" (current behaviour),
- "Fall back while firing to <nearest cell out of the attackers' line of sight, 6-15 m back along the
  explored route>" (new executor: path-follow backwards while the aim/fire executor keeps shooting the
  chosen target — fixes the walk-backwards-no-fire retreat),
- "Push to <cover point / doorway> nearer the enemies" (for the closet trap: fight at the closets),
- "Get <health item>, N m, then return" (only if one is reachable and not past the enemies),
- "Leave toward the objective" (break contact).
Options are generic: cover cells come from the navigation graph + line-of-sight to visible monsters; no map
ids. Laya picks one in a "tactic" head (reusing the objective head's architecture: score text options);
then the planner narrows the 2 Hz questions to that tactic for 3-6 s or until it terminates (option
framework: initiation, policy, termination).
- Labels: same rollout machinery as P1, but branching over tactic options with H=15 s. Before any labels
  exist, zero-shot Laya on the option text (it followed outcome sentences 16/30 and 24/30 in the fight
  question), but expect the earlier back-off failure to repeat until the fall-back executor is fixed.
- Cost: ~1 extra Laya call per tactic decision (a few per fight, i.e. negligible); labelling ≈ P1 at 1.5x
  horizon. Executor work: one new path-following-while-firing maneuver, plus cover-cell facts.
- Authority: code describes facts (where cover is, which item, distance) and executes; Laya chooses the
  tactic. Fall-back target cell is a fact ("out of sight of the visible shooters"), not a priority ranking.
- Cheapest experiment: before training anything, run *forced-tactic* rollouts on the 300 P1 death states:
  for each, does "fall back while firing" or "push to closets" survive where "hold" dies? If no maneuver
  beats hold in a meaningful fraction (say >20 %) of death states, the tactic layer has nothing to choose
  between and the fix is lower (aim/movement/facts). If it does, the same rollouts are the training set.

### P3 (rank 3). Outcome (affordance) head on Laya's encoder: SayCan x DFP

*What it is:* add a small head that predicts, for each option text in any question, future measurements at
2/5/10 s (HP lost, death, damage dealt, objective progress). Trained by regression on all branches from P1/P2
(every branch is a labelled (state, option) → outcome example, not only the winner), plus on-policy logs.
Use: (a) a combined score = policy logit + λ·goal-weighted predicted outcome, all inside the model (SayCan
product, DFP goal vector); (b) a cheap "value" for truncating rollouts after 2-5 s instead of 10-20 s
(AlphaZero-style bootstrap), cutting P1/P2 cost 2-4x.
- Laya level: every level (objective, tactic, command) shares the encoder; one outcome head per horizon.
- Cost: no extra calls (the head runs with the existing ones); training data is a free by-product of P1/P2.
- Authority: the goal vector is the same fixed reward already used in `fight_rollouts.py`; the outcome
  estimate is a model output, so the model is still choosing. Keep λ and the goal vector fixed and generic
  (not per map, not per enemy type).
- Cheapest experiment: train on P1 branches, measure held-out ranking accuracy (does predicted death
  separate branches that died from those that survived, AUC on other maps' states). AUC < 0.7 → not useful.

### P4 (rank 4). Online small-budget lookahead at objective level only, results shown as facts

*What it is:* when the planner re-plans with no enemy in view (safe moments, e.g. right after the red key on
MAP04), fork a headless engine from the current save and play each of the top 2-3 objectives (Laya's prior)
for 20-30 s with Laya playing on; turn each result into a fact in the option text ("Tried from here:
reached in 22 s, health 61 → 15" / "died after 14 s"); Laya chooses. Gumbel sequential halving if budget is
3+ options.
- Cost: 3 x 25 s / 7 ≈ 11 s engine + ~150 decision rounds of calls ≈ 15-30 s wall per re-plan, **while the
  real game keeps running** (real-time rule). Only viable at moments with nothing in view, and the result
  is stale by the time it arrives. Use the same data for offline objective-head training instead if this
  proves too slow.
- Authority: borderline. The outcome sentence is a game fact, but Laya follows outcome sentences strongly,
  so in effect the rollout chooses. Treat as an experiment and report it as such; the clean version is
  offline distillation into the objective head.
- Cheapest experiment: offline on MAP04 post-red-key saves: does any objective (e.g. health first, a
  different route) give higher blue-key success than Laya's choice over 3 seeds each? If not, skip.

### P5 (rank 5). Expert-iteration loop (repeat P1-P3 for 2-3 rounds)

Once P1 helps, repeat collection with the new heads (DAgger/AggreVaTe: the state distribution must come from
the current model) and anneal toward longer horizons. Only after a round shows a paired improvement on saved
states. Cost per round ≈ P1. This is ExIt with a 4-way Gumbel search at the root and the policy as
continuation; no tree below the root.

### Order of work

1. P1 offline regret study on ~300 death-window states (needs only `--fight-states` at command questions
   and a branch loop around `fight_rollouts.py`). This answers "is the choice the problem?" cheaply.
2. In parallel, forced-tactic rollouts for P2 with a minimal fall-back-while-firing executor. This answers
   "is there a better maneuver available at all?".
3. Whichever shows headroom gets distilled; P3's head is trained from the union of both branch sets.
4. Paired evaluation on saved death states first (deterministic, low variance); in-play 10-seed A/B only
   after that, on development maps; held-out maps only by snapshot.

## What is unlikely to work at our scale

- **Online MCTS at 2 Hz.** Even one rollout of 5 s per option costs ~1-3 s wall with model calls; a
  0.5 s decision cycle allows effectively zero simulations. Search must be offline or at rare safe moments.
- **MuZero-style learned dynamics.** Learning a Doom world model for planning needs data and model
  capacity we do not have; we already own a perfect simulator (savestates), which is strictly better.
- **End-to-end hierarchical RL (FuN, HIRO, Director, option-critic).** Millions of frames; our label budget
  is thousands of rollouts. Use the hand-written planner/tactic option generators instead.
- **ToT / RAP / Voyager / DEPS-style generation.** Laya cannot propose plans, write code or self-critique;
  only the scoring half of these methods transfers.
- **Using System 2 (Qwen 8B/27B) for tactics.** Seconds per call; fights are decided in 2-4 s (85 → 10 HP in
  4 s in the trap). Fine for objectives, too slow for combat.
- **RHEA / sequence search over raw 0.5 s actions.** Branching 9 commands x targets x weapons x 4 moves per
  step; needs thousands of evaluations per decision.
- **Rollout labels from one deterministic RNG stream.** Without decorrelation they encode luck of that stream.
- **Expecting command-level relabelling to fix executor-level losses.** If the death-window regret study
  shows Laya's choice is already the best of its top-4 in most states, the remaining deaths are in aim,
  movement, unseen attackers or missing maneuvers, and no amount of search over the same options helps.
