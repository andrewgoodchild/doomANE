# RL and IL agents that play Doom (ViZDoom), 2016 to October 2026

Notes compiled 2026-10-24. Items marked **[prior knowledge, not re-fetched]** come from the researcher's training knowledge, with the canonical URL given but not opened in this session. Treat them as likely correct but unverified.

## 1. ViZDoom competitions 2016-2018: winners, methods, findings

### Takeaway
The 2016-2017 Visual Doom AI Competition (VDAIC) winners were all deep RL agents with hand-coded crutches: F1 (A3C plus curriculum), IntelAct (DFP), Marvin (A3C pretrained on human replays) and Arnold/Arnold4 (a modular DQN-navigation plus DRQN-aiming split). The best results came from modular perception/aim/navigation splits and hard-coded anti-stuck rules. Every bot stayed far below a competent human, and their weaknesses were navigation, exploration, memory, dodging and wall-shooting suicides. Those are the same weaknesses our bot has.

### Cited Findings
**2016, Track 1 (limited deathmatch, known map):** F1 559 frags (1st), Arnold 413, Clyde 393, TUHO 312. Arnold had the best frags-per-death ratio (F/D 1.90 vs F1 1.35) — [Wydmuch et al. 2018, ViZDoom Competitions](https://ar5iv.labs.arxiv.org/html/1809.03470)

**2016, Track 2 (full deathmatch, unknown maps):** IntelAct 256 frags (F/D 3.08), Arnold 164 (F/D 32.8: 167 kills, only 5 deaths), TUHO 51 — [Wydmuch et al.](https://ar5iv.labs.arxiv.org/html/1809.03470)

Methods:
- **F1** (Wu & Tian, Facebook): an A3C variant with curriculum learning, from weak opponents on small maps to stronger opponents on bigger maps. Some behaviours were hard-coded, e.g. moving faster when not firing — [Wydmuch et al.](https://ar5iv.labs.arxiv.org/html/1809.03470). The paper is "Training Agent for First-Person Shooter Game with Actor-Critic Curriculum Learning", ICLR 2017 — [OpenReview](https://openreview.net/forum?id=Hk3mPK5gg) **[prior knowledge, not re-fetched]**.
- **IntelAct** (Dosovitskiy & Koltun, Intel): Direct Future Prediction (DFP). It predicts future measurements (health, ammo, frags) for each action, and the goal is a linear combination of measurements that can be changed at test time without retraining. Only weapon selection was hard-coded. It was trained against built-in bots on several maps — [Wydmuch et al.](https://ar5iv.labs.arxiv.org/html/1809.03470). Paper: "Learning to Act by Predicting the Future", ICLR 2017 — [arXiv 1611.01779](https://arxiv.org/abs/1611.01779) **[prior knowledge, not re-fetched]**.
- **Arnold** (Lample & Chaplot, CMU): a DQN navigation network and a DRQN aiming network, with a binary "enemy visible" classifier choosing which one acts. Navigation was rewarded for speed and penalised for lava. Aiming was rewarded for kills and pickups and penalised for health or ammo loss. Hard-coded crouching gave it a large edge — [Wydmuch et al.](https://ar5iv.labs.arxiv.org/html/1809.03470). Paper: "Playing FPS Games with Deep Reinforcement Learning", AAAI 2017 — [arXiv 1609.05521](https://arxiv.org/abs/1609.05521) **[prior knowledge, not re-fetched]**. In that paper the game-feature auxiliary head (enemy-in-view supervised from engine labels) gave a large boost **[prior knowledge]**.
- **Clyde**: plain A3C, rewarded for kills and items, penalised for suicides. **TUHO**: a Dueling DQN navigator plus a Haar-cascade enemy detector that turned and fired on detection — [Wydmuch et al.](https://ar5iv.labs.arxiv.org/html/1809.03470)

The organisers' 2016 conclusions: no agent came close to a human. Bots were decent at tactical aiming but weak strategically (defence, escaping, navigation, exploration). On Track 2 they often circled one spot waiting for targets. None aimed vertically, strafing was rare, none dodged missiles, and none showed memory (they did not chase fleeing enemies) — [Wydmuch et al.](https://ar5iv.labs.arxiv.org/html/1809.03470)

**2017, Track 1:** Marvin 248 frags, Arnold2 245, Axon 215, TBoy 198, F1 164, YanShi 158, DoomNet 139 — [Wydmuch et al.](https://ar5iv.labs.arxiv.org/html/1809.03470)

**2017, Track 2 (five unknown community maps):** Arnold4 275 (0 suicides), YanShi 273 (best F/D 1.47), IntelAct 221, Marvin 193 — [Wydmuch et al.](https://ar5iv.labs.arxiv.org/html/1809.03470)

2017 methods:
- **Marvin**: A3C pretrained on replays of human games, then trained against built-in bots. It had separate movement and aiming outputs (including vertical aim), and a hand-crafted aiming policy overrode the network when the network was unsure — [Wydmuch et al.](https://ar5iv.labs.arxiv.org/html/1809.03470)
- **Arnold2/Arnold4**: dropped the separate navigation DQN and added strafing. Crouching was banned in 2017. They were **hard-coded to run straight after about 1 s standing still**, to fix low mobility — [Wydmuch et al.](https://ar5iv.labs.arxiv.org/html/1809.03470)
- **Axon**: A3C pretrained on a small human dataset, then trained on skill tasks (navigation, aiming), then against F1. The input was five image scales, including a zoomed crosshair crop — [Wydmuch et al.](https://ar5iv.labs.arxiv.org/html/1809.03470)
- **YanShi**: separated perception from planning. A Region Proposal Network detected enemies and resources, combined with **SLAM + MCTS + hand-written rules** (fast scanning, dodging, escaping corners). An RPN aimbot handled vertical aim, and the 2016 F1/IntelAct agents were reused for low-level control — [Wydmuch et al.](https://ar5iv.labs.arxiv.org/html/1809.03470)
- Turmio: A3C plus a Haar detector. TBoy: A3C with random grouped curriculum. DoomNet: AAC. AlphaDoom: A3C — [Wydmuch et al.](https://ar5iv.labs.arxiv.org/html/1809.03470)
- Track 1 stats: Marvin picked up the most medkits (354) and took the most hits (1266). Arnold2 had the best shooting precision (32.8%) and detection precision (64.3%) — [Wydmuch et al.](https://ar5iv.labs.arxiv.org/html/1809.03470)

The organisers' 2017 conclusions: bots remained weaker than humans, especially on unknown maps. Suicides stayed high because of poor shooting precision and shooting walls with splash weapons. Detection precision fell on unknown maps, and some textures made bots fire wildly. Bots had little memory and passed enemies without pursuing them. Strafing and vertical aim did improve. An author "can consistently defeat all of them by a considerable margin" — [Wydmuch et al.](https://ar5iv.labs.arxiv.org/html/1809.03470)

**2018:** the competition ran on AICrowd, with a single-player track and a multiplayer track, closing on 12 Aug 2018 with prizes of $500/$300/$200 — [AICrowd Track 1](https://www.aicrowd.com/challenges/visual-doom-ai-competition-2018-singleplayer-track-1), [AICrowd Track 2](https://www.aicrowd.com/challenges/visual-doom-ai-competition-2018-multiplayer-track-2).

Code: Arnold is at [glample/Arnold](https://github.com/glample/Arnold), with about 537 stars, last push 2022-01-20 and **no licence file** (GitHub API, checked 2026-10-24). DFP is at [isl-org/DirectFuturePrediction](https://github.com/isl-org/DirectFuturePrediction): archived, licence "NOASSERTION" (a custom Intel licence), about 151 stars. DoomNet's PyTorch code is at [akolishchak/doom-net-pytorch](https://github.com/akolishchak/doom-net-pytorch), MIT, last push 2022-03. The F1 code repo (yuxinwu/F1) was not found (404).

### Inferences
- Arnold is the closest prior art to our design: a classifier gates between a navigation policy and a combat policy. Its two big wins were an auxiliary "enemy visible" head trained from engine labels and simple hard-coded reflexes (crouch, anti-stuck run). That supports keeping our hand-written executor and adding reflex rules, rather than learning everything end to end.
- DFP's measurement-prediction framing (predict Δhealth, Δammo and Δkills for each candidate action, then score with goal weights) maps naturally onto a typed command picker. Laya could predict the future Δhealth and Δprogress of each command and choose by weighted sum, and the weights could be switched at runtime (e.g. "survive" vs "push").
- YanShi shows that a mostly classical stack (detector + SLAM + search + rules) can place 2nd on unknown maps, which is the same architecture class as our bot.
- Even the 2017 winners died about 200-290 times per 10 matches against bots. Combat survival was never solved in this era, so we should not expect to copy a solution from it.

### Gaps
- Official 2018 VDAIC winners and methods were not found (the AICrowd pages list no results, and the Wydmuch paper covers only 2016-2017).
- F1 code availability is unconfirmed. Exact network sizes and input resolutions for the competition bots are not given in the summary paper.

## 2. Sample Factory and its Doom results

### Takeaway
Sample Factory (Petrenko et al., ICML 2020; MIT; still actively maintained, last push 2026-10-19) is the strongest public pixel-based Doom combat agent. Its 8-agent PBT runs beat the hardest built-in bots in 100% of deathmatch and duel matches, at about 146k FPS on one 36-core + 2080 Ti machine. Pretrained checkpoints for battle, battle2, deathmatch, duel-bots and self-play duel are on HuggingFace.

### Cited Findings
- Throughput: about 146,551 env FPS peak on a 36-core dual Xeon Gold 6154 with one RTX 2080 Ti. PBT Battle reached about 146-154k FPS — [Sample Factory paper](https://arxiv.org/html/2006.11751v2)
- Deathmatch vs built-in bots: mean 80.5 frags vs the bots' 12.6. Duel: 34.7 vs 3.6. Wins 100% of matches against the highest-difficulty bots — [paper](https://arxiv.org/html/2006.11751v2)
- In 100 matches, the self-play duel agent beat the bot-trained agent 78 times, lost 3 and drew 19 — [paper](https://arxiv.org/html/2006.11751v2)
- Each of the 8 agents trained for 2.5 B frames on one 36-core, 4-GPU server, about 18 years of simulated experience for the population. The authors say the agents are "not yet at the level of expert human players". They give no human win rate — [paper](https://arxiv.org/html/2006.11751v2)
- Architecture: 3 conv layers (32/64/128), FC 512, GRU 512, plus a 128-unit branch for measurements (health, armour, ammo). It has **7 independent action heads** (move, strafe, attack, sprint, use, weapon, horizontal aim), about 12,096 joint actions. Agents learn to open doors and call lifts (Fig. 9 shows a secret door being opened). The stated limitations are no audio and privileged bots — [paper](https://arxiv.org/html/2006.11751v2)
- Environments and recipes: `doom_battle`, `doom_battle2`, `doom_deathmatch_bots`, `doom_duel_bots`, `doom_duel`, `doom_deathmatch_full`, plus the basic scenarios such as `doom_my_way_home`. Battle runs use 4 B env steps with APPO and an RNN — [Sample Factory docs](https://www.samplefactory.dev/09-environment-integrations/vizdoom/)
- HuggingFace checkpoints (mean of 10 runs): `andrewzhang505/sample-factory-2-doom-battle` 59.37±3.93, `…-doom-battle2` 36.40±4.20, `andrewzhang505/doom_deathmatch_bots` 85.66±28.53, `andrewzhang505/doom_duel_bots_pbt` 55.39±17.13, plus `andrewzhang505/doom-duel-selfplay` — [docs](https://www.samplefactory.dev/09-environment-integrations/vizdoom/). The checkpoints are about 35-41 MB pickled `.pth` files. Community re-trains include `execbat/sample-factory-doom-deathmatch-bots` and `hishamcse/doom_deathmatch_bots` — [HF](https://huggingface.co/andrewzhang505/doom_deathmatch_bots/tree/main/checkpoint_p0), [HF execbat](https://huggingface.co/execbat/sample-factory-doom-deathmatch-bots/tree/main/checkpoint_p0)
- Repo: [alex-petrenko/sample-factory](https://github.com/alex-petrenko/sample-factory), MIT, about 1,028 stars, last push 2026-10-19 (GitHub API)

### Inferences
- The SF deathmatch checkpoint is a small conv+GRU policy (about 35 MB in fp32) that could plausibly be converted to Core ML and run on the ANE as a **combat sub-policy**. Our executor would hand control to it when Laya emits `attack` or when enemies are visible, Arnold-style. The input format must match: SF Doom envs use a low-res RGB frame (128x72 by default, prior knowledge) plus a measurements vector. It was trained on deathmatch maps against bots, so transfer to Freedoom single-player monsters is unproven. Fine-tuning in a custom ViZDoom scenario built from Freedoom monster rooms is a realistic route at Apple-Silicon scale only if env throughput is high; one Mac will be far below 146k FPS.
- The 7-head factorised action space is a good template if we ever learn low-level control.

### Gaps
- No published SF result on single-player campaign maps, or against Freedoom monsters, was found.
- The checkpoints' exact observation preprocessing was not re-verified here.

## 3. RL agents that complete full Doom/Freedoom campaign levels

### Takeaway
No peer-reviewed RL agent with reported key, door and exit success rates on full Doom/Freedoom campaign levels was found. The only evidence is a hobby repo, lucas-moont/doom-player (Oct 2026). Its PPO with distance-to-exit shaping clears E1M1, which has no keys, in 15 of 15 runs. Those E1M1 policies score 0 of 5 on E1M2, which needs a key and lifts. ViZDoom 1.3 (2026) now ships original/Freedoom levels as Gymnasium environments, so the benchmark exists but has no published agents.

### Cited Findings
- lucas-moont/doom-player PR #31 "RL Clears E1M1: 15/15 from screen pixels": PPO trained on pixels with a progress-shaping reward (movement toward the exit) and a death penalty clears 15 of 15 runs across 3 seeds after about 5 M steps each. A version trained only on the exit reward reached 8.5% progress after 1 M steps, below random play (about 20%). Self-reported, not peer-reviewed — [PR #31](https://github.com/lucas-moont/doom-player/pull/31)
- PR #32 (merged 2026-10-07): adds a key-aware "KeyedRoute" progress meter for E1M2 (9,177 units: 2,222 to the red key, then 6,955 to the exit) and treats lifts and switch-opened doors (linedef specials 62, 88, 103, 46) as passable. E1M1-trained PPO scores 0 of 5 clears, 0.6-4.2% progress (random 2.3%), and **no contender picked up the red key**. ViZDoom has no key game variable, so held keys are inferred from the objects list — [PR #32](https://github.com/lucas-moont/doom-player/pull/32). Repo: no licence, 0 stars, last push 2026-10-10 (GitHub API).
- GameNGen (Valevski et al., Google, Aug 2024): a PPO agent (Stable-Baselines3, CPU, Nature-CNN on 160x120 frames plus automap, last 32 actions) was trained 50 M steps with dense event rewards (+1000 kill, -5000 death, exploration bonuses, secrets). It has **no level-completion reward**, and the authors say it "still does not explore all of the game's locations and interactions". It was used to generate 70 M frames for the diffusion world model — [GameNGen](https://arxiv.org/html/2408.14837)
- ViZDoom ships Gymnasium environments for original levels named `Vizdoom<Game><Map>-S<X>-v0` (Freedoom, Freedoom2, Doom, Doom2; skill 1-5). Commercial IWADs must be supplied by the user — [ViZDoom docs](https://vizdoom.farama.org/environments/original_doom_levels/). This came with 1.3.0 ("original Doom levels as Gymnasium environments") — [ViZDoom releases](https://github.com/Farama-Foundation/ViZDoom/releases)

### Inferences
- The field has essentially no learned agents that handle keys, switches and lifts on real maps. Our MAP01-03 completions with a hand-written navigator are probably ahead of any published pure-RL agent on campaign levels. The lucas-moont result agrees with our experience: route knowledge for keys and lifts does not emerge from shaped RL on one map.
- Privileged map data (linedef specials, objects list) works best for measurement and for planner-side reasoning. That supports extending our planner's mechanism model (lift and switch linedef types) instead of hoping a learned policy discovers them.

### Gaps
- No Freedoom MAP01-32 success-rate tables from any RL paper were found. Whether any 2025-2026 papers use the new `VizdoomFreedoom*` environments is unknown.

## 4. Hierarchical/options RL, curiosity, exploration and memory agents

### Takeaway
Curiosity (ICM, 2017) and episodic-reachability curiosity (2018) solve sparse "My Way Home"-style navigation. The memory and SLAM-augmented work is mostly 2016-2019 and small-scale. None of it addresses combat-plus-progression on real maps.

### Cited Findings
- Episodic Curiosity through Reachability (Savinov et al., ICLR 2019): the novelty bonus is the number of environment steps (reachability) from memory, which avoids the "couch-potato" failure of ICM. On ViZDoom navigation it reaches a distant goal at least 2x faster than ICM — [arXiv 1810.02274](https://arxiv.org/abs/1810.02274v5). Code at [google-research/episodic-curiosity](https://github.com/google-research/episodic-curiosity), Apache-2.0, archived (GitHub API).
- ICM (Pathak et al., ICML 2017): self-supervised inverse/forward-model curiosity that solves ViZDoom "My Way Home" with very sparse reward and explores levels without any extrinsic reward **[prior knowledge, not re-fetched]**, [arXiv 1705.05363](https://arxiv.org/abs/1705.05363). Code at [pathak22/noreward-rl](https://github.com/pathak22/noreward-rl), custom licence, inactive since 2022 (GitHub API).
- World Models (Ha & Schmidhuber 2018): a controller trained inside a learned VAE+MDN-RNN model of VizDoom "Take Cover" learns to dodge fireballs — [arXiv 1803.10122](https://arxiv.org/pdf/1803.10122)
- Decision Transformer with multi-state-action tokenisation on ViZDoom Deadly Corridor and My Way Home (2024) — [arXiv 2407.01310](https://arxiv.org/pdf/2407.01310)
- Transformer-based deep RL in ViZDoom (AIRI, 2022) reports transformer agents beating conventional baselines — [AIRI](https://airi.net/articles/transformer-based-deep-reinforcement-learning-in-vizdoom)
- Other relevant work **[prior knowledge, not re-fetched]**: Bhatti et al. 2016, "Playing Doom with SLAM-Augmented Deep RL" (arXiv 1612.00380), which adds semantic SLAM maps as input; Chaplot et al., "Active Neural Localization" / "Gated-Attention" language-grounded ViZDoom navigation (2017-2018); and "Semi-parametric Topological Memory" (SPTM, Savinov et al. ICLR 2018), a graph of observations plus a reachability network for navigation in ViZDoom mazes. SPTM is the closest learned analogue of our navigator graph.

### Inferences
- A reachability network (SPTM / episodic-curiosity style) could be a learned complement to our navigator: given current and remembered frames, predict "reachable within k steps". This could help detect when we are stuck, or whether a lift or door has opened a new region.
- None of these agents fights. Curiosity agents in Doom are navigation-only demonstrations.

### Gaps
- No options/HRL paper with full-level Doom results was found. HRL Doom work found was limited to the hard-coded module splits in the competition bots.

## 5. Imitation learning, behavioural cloning and offline RL on Doom

### Takeaway
Human data helped the 2017 winners as pretraining (Marvin, Axon). The main recent BC study (Sony, 2024) gets roughly average-human deathmatch skill from about 41 matches of recordings, below RL performance but more human-like. No public large Doom demo dataset or offline-RL Doom benchmark was found.

### Cited Findings
- Marvin (VDAIC 2017 Track 1 winner) pretrained A3C on human replays. Axon pretrained on a small human dataset — [Wydmuch et al.](https://ar5iv.labs.arxiv.org/html/1809.03470)
- "Behavioural Cloning in VizDoom" (Spick, Bradley, Raina, Amadori, Moss; Sony Interactive; arXiv 2401.03993, Jan 2024): pixel-only BC on Doom 2 deathmatch. It reaches average-player performance and beats the worst players, but stays below RL agents, while being more human-like (camera movement, trajectories). Aggressive and passive styles can be cloned — [ar5iv](https://ar5iv.labs.arxiv.org/html/2401.03993). The dataset reportedly has 41 matches from 10 players, captured at 35 fps with 640x480 RGB, depth and segmentation. These dataset details come from a secondary AI-generated review, so verify them — [themoonlight review](https://www.themoonlight.io/en/review/behavioural-cloning-in-vizdoom)
- A search for .lmp-demo-based ML datasets and offline RL on Doom found nothing Doom-specific (only generic D4RL) — [D4RL](https://arxiv.org/pdf/2004.07219)

### Inferences
- `.lmp` demos (e.g. from the DSDA speedrun archive) are input-only recordings that replay deterministically in the matching source port. In principle they can be rendered through ViZDoom/ZDoom to produce (frame, action) pairs for full-level route BC, including keys, switches and lifts. No paper doing this was found. Port-compatibility (vanilla/Boom vs ZDoom demo formats) is the main obstacle (prior knowledge). This is the most promising unexplored source of route knowledge for unmodelled mechanisms.
- Our own bot's logs (successful MAP01-03 runs) are themselves a BC/offline dataset for Laya. Sony's finding that better players make better clones suggests filtering to the successful episodes.

### Gaps
- DoomTrainer: no source found.
- No public Doom human-demo dataset with frames was confirmed to be downloadable. Whether Sony released its dataset is unknown.

## 6. Recent work 2023-2026, tooling, licences and maintenance

### Takeaway
Post-2022 Doom RL is mostly benchmarks (LevDoom 2022, COOM 2023, HASARD 2025, all built on Sample Factory) plus tooling (ViZDoom 1.2.x to 1.3.x, with Apple-Silicon wheels and original levels). No 2023-2026 paper was found that substantially advances combat or full-level completion.

### Cited Findings
- LevDoom (Tomilin et al., IEEE CoG 2022): a generalisation benchmark with 4 scenarios x 5 visual-difficulty levels, Gymnasium on PyPI — [LevDoom paper](https://ieee-cog.org/2022/assets/papers/paper_30.pdf); [TTomilin/LevDoom](https://github.com/TTomilin/LevDoom), MIT, last push 2026-03 (GitHub API)
- COOM (NeurIPS 2023 Datasets and Benchmarks): a continual-RL benchmark with 8 scenarios in sequences of 4/8/16 tasks — [COOM paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/d61d9f4fe4357296cb658795fd7999f0-Paper-Datasets_and_Benchmarks.pdf); [TTomilin/COOM](https://github.com/TTomilin/COOM), MIT, last push 2026-03 (GitHub API)
- HASARD (arXiv 2503.08241, March 2025; ICLR 2025 per OpenReview): a vision-based safe-RL benchmark with 6 environments x 3 difficulties. It uses Sample Factory with Lagrangian PPO baselines and 14 keys (864 discrete combos) plus continuous mouse aim — [arXiv](https://arxiv.org/pdf/2503.08241), [OpenReview](https://openreview.net/pdf?id=5BRFddsAai); [TTomilin/HASARD](https://github.com/TTomilin/HASARD), MIT, last push 2026-06 (GitHub API)
- The ViZDoom docs list LevDoom, COOM and HASARD as third-party environments — [ViZDoom third party](https://vizdoom.farama.org/environments/third_party/)
- ViZDoom releases: 1.2.2 (27 Sep), 1.2.3 (16 Dec, Python 3.12), 1.2.4 (20 Aug: NumPy 2, **Apple Silicon arm64 wheels**), 1.3.0.dev1 (29 May: removed the old Gym wrapper), dev2 (29 Jun: Freedoom 0.13, truncation, MultiBinary), dev3 (22 Oct: notifications/audio buffers, segmentation label categories), 1.3.0 (11 Feb: original Doom levels as Gymnasium envs, audio and notification buffers, reward helpers, Py 3.13/3.14, dropped Intel-mac wheels), 1.3.1 (20 Sep: adds `category` to `state.objects`), 1.3.2 (13 Oct: labels-buffer bbox fix, fix for a crash on sectors with more than 128 lines) — [ViZDoom releases](https://github.com/Farama-Foundation/ViZDoom/releases). The page shows no years. The likely sequence (my inference) is 1.2.2 Sep 2023, 1.2.3 Dec 2023, 1.2.4 Aug 2024, dev1-dev3 May-Oct 2025, 1.3.0 Feb 2026, 1.3.1 Sep 2026, 1.3.2 Oct 2026. The repo was pushed 2026-10-21 and has about 2,102 stars (GitHub API). The ViZDoom code is MIT with GPL-licensed ZDoom components **[prior knowledge; the GitHub API reports no single SPDX licence]**.
- LeanRL ([vwxyzjn/LeanRL](https://github.com/vwxyzjn/LeanRL), licence NOASSERTION, last push 2024-09) and CleanRL (last push 2026-04) are general fast-PyTorch RL baselines. No Doom-specific LeanRL results were found (GitHub API).
- "AI Finds A Way" (arXiv 2608.23875, Aug 2026, Dharna, Lu, Sullivan, Lehman, Krakovna, Clune) is an anecdote collection on reward hacking. The abstract does not mention Doom — [arXiv](https://arxiv.org/abs/2608.23875)

### Inferences
- ViZDoom 1.3's `state.objects` categories, labels buffer and notifications buffer (e.g. "You need a blue key") are directly useful perception inputs for our planner and Laya, and they replace screen parsing.
- Since 1.2.4 the arm64 wheels allow local training on our Mac. Sample Factory supports macOS CPU workers, but throughput will be much lower than the paper's 146k FPS (inference, not measured).
- The best transfer candidates, ranked: (1) Arnold-style gating plus an SF-pretrained or SF-recipe combat sub-policy for `attack` episodes; (2) DFP-style measurement prediction to score Laya's typed commands for survival; (3) hard-coded reflexes from competition bots (anti-stuck run, strafe-while-firing, retreat at low health, don't fire splash weapons at close walls); (4) demo/self-log BC for route knowledge.

### Gaps
- No 2025-2026 paper on multi-task or foundation-model RL agents in ViZDoom with combat or level-completion results was found in this session's searches. Coverage of late-2026 arXiv may be incomplete.
- Papers with Code is defunct (shut down 2025, prior knowledge), so no leaderboard was checked.
