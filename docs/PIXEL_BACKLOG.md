# Pixel perception: prior work and experiment backlog

Written 2026-10-31, during pix4 (see `docs/LAB_NOTEBOOK.md`, pix1–pix4). Level 1 means monsters and items
come from the screen (and, from pix3, the depth buffer); the map, doors, keys and counters still come from
the WAD and the engine.

## What others have done

| Work | Input | Task | What carries over |
|---|---|---|---|
| Lample & Chaplot, *Playing FPS Games with Deep RL* (AAAI 2017); Arnold, winner of ViZDoom 2017 | pixels | deathmatch | The same trick as pix1: predict "enemy / item on screen" from the labels buffer during training, play from pixels. Separate navigation and action networks. |
| Dosovitskiy & Koltun, *Learning to Act by Predicting the Future* (ICLR 2017), winner of ViZDoom 2016 full deathmatch | pixels plus health, ammo, frags | deathmatch on unseen maps | Measurements as inputs; no object detector at all. |
| Tsinghua (TSAIL), winner of ViZDoom 2018 Track 1 (Jiayi Weng's slides) | pixels plus depth | **finish 10 unseen single-player maps**, fastest | The closest system to ours: YOLOv3 detector for enemies and obstacles, depth into navigation, hierarchical RL (navigation, attack, tools, resources). 25.3 min total over 10 maps. Second was DoomNet (pixels plus game variables). The slides do not say how false detections were handled. |
| Sample Factory (Petrenko et al., ICML 2020) | pixels | battle, duel, deathmatch | End-to-end RL at billions of frames; no levels with keys and exits. |
| de Wynter, *Will GPT-4 Run DOOM?* (2024) | screenshots described by GPT-4V | E1M1 | Doors, fights and pathing work; reached the last room once. Fails on memory and **hallucinated explanations**. A hosted model, so not usable here. |
| VideoGameBench (2025) | screenshots, VLMs | ten 1990s games | Best models complete 0.48 %; they stall at the start of each game, and latency is a large part of it. |
| *See, Symbolize, Act* (2026), Atari, ViZDoom, AI2-THOR | frames plus extracted symbols | gameplay | Symbols help "only when symbol extraction is reliable"; perception quality is the bottleneck. This is pix1's lesson in another setting. |
| *Playing DOOM with 1.3M Parameters* (2026) | ASCII frame plus depth | defend_the_center | Depth as an input to a tiny model, 31 ms per decision; 178 frags against 13 for the LLMs. |
| Video object detection (tracking-based suppression, TCM and others) | video | detection | Drop detections that do not persist or that do not agree with neighbouring frames. Hard-negative mining is standard. |

What nobody reports: a pixel agent finishing full campaign maps in real time with a planner on top, or how
a **persistent** false detection affects a goal-directed agent. In deathmatch a phantom costs a shot; for us
one phantom that stays in view holds the agent in a fight with a wall (pix1, MAP03).

## Backlog

Each item gets a pre-registered rule in the notebook before it runs. Development maps only; MAP01–04 stay out
of training while they are the game maps. The order is the proposed order.

### Measurement first

1. **Sequence phantom test (gate for everything below).** The walked-view test samples single positions.
   Play instead contiguous stretches of the pix1 engine games' telemetry (for example 10 s at every 60 s)
   and count *persistent* false monsters: present on at least 2, 10 and 35 consecutive tics (the runtime's
   confirmation, a short fight, one second). Check it against pix1: the MAP03 start must show persistent
   phantoms. Cost: an hour of code, minutes to run. Why: the tracker literature drops short-lived
   detections, and the runtime already does; what hurts is what persists.
2. **Short closed-loop gate.** Before the 3.5-hour comparison, play MAP01 (180 s) and the first 300 s of
   MAP03 with seed 54. A model that does not leave the MAP03 start in 300 s does not get the full games.
   Cost: 10 minutes per model.

1a. **Done 2026-11-01** (items 1 and 8): pix3 at threshold 0.79 is the best model at matched recall; see the
   notebook. Found on the way: pix1's MAP01 failure came from the runtime's memory and tracker, not from
   lasting phantoms.
1b. **Runtime sequence test.** Run `PixelPerception` plus the agent's enemy memory over the sequence sets and
   compare with the engine objects tic by tic: enemies listed that do not exist, ids per real monster,
   position error. This is what the stack actually receives.
1c. **Forget on negative evidence.** A remembered enemy or item whose position is in view, unoccluded by the
   depth buffer, and not detected is dropped, as a player who looks back at a spot and sees nothing would.
   Standard track deletion on missed detections. Perception only; the model still chooses.

### Perception (the model)

3. **pix4: distractor classes** (training now). Corpses, barrels and decorations named instead of called
   "nothing".
4. **On-policy frames (DAgger).** Play pixel-perception games on the training maps and save every view where
   pixels and labels disagree (both phantoms and misses), with the labels as ground truth; add them to
   training and repeat once or twice. This is the standard cure for a model that is trained on one
   distribution of views and acts in another (Ross et al., 2011). The views come from the agent's own
   behaviour, so they include standing in front of a screen for a minute. Cost: a few hours of games on
   training maps per round, plus training.
5. **Motion input.** Give the model the previous frame (or the difference after compensating for the turn) as
   extra channels. Monsters animate and move; wall screens and decorations do not. Pairs with the
   tracker-based approaches above. Cost: collector change (two frames per sample), training.
6. **Larger input for range.** 10–20 m is where items are found 24–41 % of the time. Try the full 400×640 view
   or a second crop of the centre at full resolution. Cost: about 4× compute on the Neural Engine
   (8–10 ms), still under budget.
7. **Box detector instead of cells.** A YOLO-style head (as the 2018 winner used) gives one score per object
   and natural non-maximum suppression; cell grouping in the runtime goes away. Compare at equal latency.
8. **Confidence output.** Report each object's softmax confidence and track its running mean; test whether a
   threshold on it separates phantoms (offline only, at first).

### Facts the stack could be given (the model still chooses)

9. **Shot feedback as a fact.** A player who shoots a wall screen sees sparks, not blood, and stops. Report for
   each tracked enemy how many shots were fired at it and whether it reacted (blood puffs, pain frames or
   movement seen in pixels; level 1 also has the hit counter). Laya decides what to do with it; the executor
   does not drop the target. Needs a check against `tools/check_authority.py`.
10. **Option length.** The planner's objective options grow with the enemy list and hit the server's
    `head_max_len=384` (pix1 crash). Cap or summarise the enemy facts in options so many detections cannot
    crash a game. Applies to engine perception too.

### Beyond level 1

11. **Projectiles from pixels.** Level 1 still takes projectiles from the labels buffer. Add them as classes
    (they are in the data as "Explosive": imp balls, plasma, rockets).
12. **Level 2.** No WAD and no object list: keys, doors, switches and the exit from pixels, and a map built from
    depth and position as a human builds one. Only after level 1 plays within the pix1 rule.
13. **Continual training loop** (the user's first idea). Once items 1, 2 and 4 exist, run them as a loop on
    the training maps: play, collect disagreements, retrain, gate, keep the model only if it passes.

## Sources

- [Lample & Chaplot, arXiv 1609.05521](https://arxiv.org/abs/1609.05521v2); [Arnold](https://github.com/puzhao8/arnold)
- [Dosovitskiy & Koltun, arXiv 1611.01779](https://arxiv.org/abs/1611.01779v2)
- [Jiayi Weng, ViZDoom 2018 slides](https://trinkle23897.github.io/pdf/Interview_Slides.pdf); [Track 1 re-evaluation](https://groups.google.com/g/vizdoom/c/UQXrW6rJupM); [DoomNet](https://github.com/akolishchak/doom-net-pytorch)
- [Wydmuch et al., ViZDoom Competitions, arXiv 1809.03470](https://huggingface.co/papers/1809.03470)
- [Sample Factory, arXiv 2006.11751](https://arxiv.org/abs/2006.11751)
- [de Wynter, arXiv 2403.05468](https://arxiv.org/abs/2403.05468)
- [VideoGameBench, arXiv 2505.18134](https://arxiv.org/abs/2505.18134)
- [See, Symbolize, Act, arXiv 2603.11601](https://arxiv.org/abs/2603.11601)
- [Playing DOOM with 1.3M Parameters, arXiv 2604.07385](https://arxiv.org/abs/2604.07385)
- [Ensembling detectors with tracking, arXiv 2102.04798](https://arxiv.org/pdf/2102.04798); [TCM](https://ntnuopen.ntnu.no/ntnu-xmlui/handle/11250/2776365)
