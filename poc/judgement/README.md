# POC 1: a calibrated judgement head from the repo's own labels

**Question.** From what Laya already sees at a fight decision, can a small model trained on the project's own
rollout outcomes give a calibrated yes/no judgement that is clearly better than chance? Laya's own judgement
questions were at chance (AUROC 0.37 to 0.52, `results/doom/question_eval_noul.json`), and they said yes 70 to
95 % of the time whatever the facts.

**Pass criterion, fixed before any result:** on held-out runs, label A, AUROC >= 0.70 and expected calibration
error (10 equal-width bins) <= 0.05 after temperature scaling.

**Verdict: pass.** Label A: AUROC 0.755 (95 % bootstrap 0.729 to 0.780), ECE 0.009, from leave-one-run-out
over 8 runs on development maps. Details and caveats below; the honest reading is in "What the POC shows".

## Data and labels

All from the repository, development maps only (MAP02, MAP03, MAP04; checked against `doom/map_split.json`).

| Label | Source | Rows | Positives | Meaning |
|---|---|---|---|---|
| A | `results/doom_heads/fight-rollouts-v1.jsonl`, 8 runs | 7,056 | 408 (5.8 %) | the chosen movement's deterministic rollout lost >= 20 health or died |
| died | same | 7,056 | 96 (1.4 %) | the chosen movement's rollout died |
| B | `results/doom/command-rollouts-v1.jsonl`, rows of Laya's own command, 7 runs | 209 | 71 (34 %) | died within the 15 s rollout |

`death-rollouts-3s.jsonl` (560 states) is a subset of the fight rollouts and is not counted twice.

## Features

Parsed from the text Laya had at the decision, never from the outcome (`poc/judgement/features.py`):

- the movement question's state: known and visible enemies, nearest threat distance, the 12 m flag, current
  movement, body clearance left, right and back;
- the full packet state: HP, armour, inventory ammo per weapon, enemy range and health categories, open space,
  current command and its target distance, the enemy list (visible and remembered counts, nearest distances,
  counts within 5, 10 and 20 m, hitscan, melee and projectile types, a 1/distance threat sum), nearest health item;
- the combat facts in the question (incoming projectiles, seconds, miss, side, melee nearest, hitscan visible);
- Laya's chosen movement (ablated: "without chosen").

Label B uses the same kind of features read from the run's telemetry row at the state's tick
(`poc/judgement/label_b.py`).

## Model and evaluation

Logistic regression and a 2x64 MLP (dropout 0.1, weight decay 1e-3), full-batch Adam on plain cross-entropy, in
torch on the CPU (`poc/judgement/calib.py`). **Split: leave one run out.** Inside each outer fold, an inner
leave-one-run-out over the training runs gives out-of-fold logits at every epoch; these choose the epoch count and
fit the temperature. The final model is trained on all training runs and its test logits are divided by that
temperature. Nothing is tuned on the test fold. Metrics are on the pooled out-of-fold predictions.

Run:

```bash
ext/vision-venv/bin/python poc/judgement/train.py      # labels A and died, about 4 minutes on the CPU
ext/vision-venv/bin/python poc/judgement/label_b.py    # label B, seconds
# optional, needs a model server on the Neural Engine (see laya_compare.py's docstring)
ext/vision-venv/bin/python poc/judgement/laya_compare.py --endpoint http://127.0.0.1:8017/predict
```

## Results (held out by run)

| Label A: lost >= 20 or died | AUROC | 95 % CI | ECE | Brier |
|---|---|---|---|---|
| constant (prevalence) | 0.500 | | 0.000 | 0.0545 |
| rule: nearest threat distance | 0.733 | | | |
| rule: nearest visible enemy | 0.722 | | | |
| rule: 1/distance threat sum | 0.717 | | | |
| rule: HP | 0.578 | | | |
| logistic regression | 0.738 | 0.711 to 0.764 | 0.007 | 0.0520 |
| **MLP 2x64** | **0.755** | **0.729 to 0.780** | **0.009** | **0.0504** |
| MLP 2x64 without the chosen movement | 0.753 | 0.727 to 0.779 | 0.006 | 0.0506 |

Per map (MLP): MAP02 0.789, MAP03 0.712, MAP04 0.706. Per fold the AUROC ranges from 0.66 (one MAP04 run) to 0.84.

Reliability of the MLP, label A (`results/poc/judgement/reliability_A.svg`):

| Predicted | n | mean p | fraction positive |
|---|---|---|---|
| 0.0 to 0.1 | 5,709 | 0.035 | 0.032 |
| 0.1 to 0.2 | 896 | 0.141 | 0.109 |
| 0.2 to 0.3 | 346 | 0.241 | 0.272 |
| 0.3 to 0.4 | 87 | 0.343 | 0.276 |
| 0.4 to 0.5 | 18 | 0.414 | 0.444 |

| Died label | AUROC | 95 % CI | ECE | Brier |
|---|---|---|---|---|
| rule: HP | 0.913 | | | |
| rule: nearest visible enemy | 0.790 | | | |
| **logistic regression** | **0.943** | **0.919 to 0.962** | **0.003** | **0.0124** |
| MLP 2x64 | 0.918 | 0.887 to 0.947 | 0.004 | 0.0120 |

| Label B: Laya's command, died within 15 s (209 states) | AUROC | 95 % CI | ECE |
|---|---|---|---|
| rule: HP | 0.689 | | |
| logistic regression | 0.440 | 0.360 to 0.523 | 0.159 |
| MLP 2x64 | 0.480 | 0.392 to 0.563 | 0.087 |

### Laya's own yes/no head on the same states

A doomLaya model server on the Neural Engine was asked two noul questions about each of 2,000 sampled fight states
(all 408 label-A positives plus 1,592 random negatives; the full packet text as the state): "The player will
lose a lot of health or die in the next few seconds if the fight continues as it is." and "Pull back now to heal
or take cover instead of continuing the fight." The POC model's out-of-fold probabilities on the same rows are
shown next to them (`poc/judgement/laya_compare.py`, 5.5 minutes).

| Same 2,000 states | Label A AUROC (95 % CI) | Died AUROC (95 % CI) |
|---|---|---|
| Laya noul, "about to lose a lot of health or die" | 0.547 (0.514 to 0.578) | 0.586 (0.533 to 0.635) |
| Laya noul, "pull back now" | 0.530 (0.496 to 0.559) | 0.560 (0.510 to 0.614) |
| POC model (MLP for A, logistic for died) | **0.759** (0.731 to 0.788) | **0.943** (0.920 to 0.963) |

Laya's yes-probability barely moves with the facts: mean 0.54 when the fight is about to go badly and 0.47 when
it is not. ECE is not comparable on this sample, because it holds 20 % positives against 5.8 % in play; the POC
model's calibration is the one measured on the natural prevalence above.


## What the POC shows, and what it does not

- **The pass is real but narrow.** Laya's own head is at chance on the same states (0.53 to 0.59), so the gap to it is large and consistent with the phase-1 result. The MLP beats the criterion, and its probabilities are calibrated: the ECE is
  below 0.01 before and after temperature scaling. But the single rule "nearest threat distance" already gives
  0.733. The model adds about 0.02 of AUROC over that rule; what it adds beyond the rule is mainly a calibrated
  probability rather than a better ranking. The same facts were in Laya's state text when it answered at chance.
- **Deaths within a few seconds are easy to see coming** (AUROC 0.94; HP alone 0.91). This is the short-horizon
  signal the movement rollouts already carried.
- **Label B is at chance,** as the phase-1 command rollouts predicted: 15 s outcomes of a whole command are
  mostly noise, and 209 states across 7 runs cannot support a model. Logistic regression's inner selection picked
  one epoch on most folds, which says nothing generalised across runs.
- **Not shown:** that a calibrated judgement changes play. The label is the outcome of the next few seconds under
  the chosen movement, a short, local quantity. Whether a probability of "this is about to go badly" helps the
  agent needs the fall-back-while-firing executor action and a MAP04 game gate, as the project description says.
- **Not shown:** generalisation to maps outside MAP02 to MAP04. The held-out unit is the run, not the map; every
  fold has training data from the test fold's map (MAP04 folds train on three other MAP04 runs).
- **Not done:** the head on the Neural Engine. It is a 77-feature MLP; export is routine but was not needed to
  answer the question.

Outputs: `results/poc/judgement/result.json` (every number above, per fold and per map), `reliability_A.svg`,
`reliability_died.svg`, `oof_A_mlp64.npy`, `oof_died_logistic.npy` (out-of-fold probabilities, row order of the
label file).
