# World models, generative game engines, and generalization/benchmark work on Doom (2018 to Oct 2026)

Researched 2026-10-24. Scope: world models (Ha & Schmidhuber, Dreamer, IRIS/DIAMOND, GameNGen and reproductions), generalization benchmarks and map-split results, self-play combat training, small/edge models. Every claim below has a source. Where I could not find one, the claim is under Gaps.

## 1. World models and generative game engines on Doom

### Takeaway
Two families exist. (a) Small latent world models used to train a policy "in the dream" (Ha & Schmidhuber 2018, VizDoom Take Cover). (b) Large diffusion "game engines" that render Doom frames from actions (GameNGen 2024, plus 2025-2026 open reproductions). Neither is a practical way to rehearse our bot's fights or routes. GameNGen remembers only about 3 s of context and needed 128 TPU-v5e to train. The dream-trained policies exploit model errors. The useful by-products are the open datasets and recording pipelines.

### Cited Findings
**Ha & Schmidhuber, "World Models" (27 Mar 2018; NeurIPS 2018)**
- VizDoom Take Cover: the controller was trained entirely inside the learned model. It scored about 918 ± 546 steps in the dream and 1092 ± 556 in the real environment at temperature τ=1.15. "Solved" means more than 750 steps averaged over 100 rollouts. — [worldmodels.github.io](https://worldmodels.github.io/)
- Sizes: VAE 4.45M params (64-d latent), MDN-RNN 1.68M (512 LSTM units, 5 mixtures), controller 1,088 params. Training started from 10,000 random rollouts on a P100 plus a 64-core CPU VM. — [worldmodels.github.io](https://worldmodels.github.io/)
- At low τ (0.1-0.5) dream scores were about 2,060-2,086, but real scores were only about 193-196. In the dream the monsters stopped shooting (mode collapse), and the controller learned to exploit this. — [worldmodels.github.io](https://worldmodels.github.io/)

**GameNGen, "Diffusion Models Are Real-Time Game Engines" (Google; v1 27 Aug 2024, v2 24 Apr 2025, ICLR 2025)**
- Pipeline: a PPO agent (Stable Baselines 3, 50M environment steps, 8 parallel games, on CPU) plays ViZDoom and its trajectories are recorded. The training set is 70M examples drawn from those trajectories. A fine-tuned Stable Diffusion 1.4 is trained on 128 TPU-v5e for 700k steps at batch 128. — [arXiv 2408.14837v2 HTML](https://arxiv.org/html/2408.14837v2)
- Context: 64 past frames and actions. Inference: 4 DDIM steps at 20 FPS on one TPU-v5, or about 50 FPS with a 1-step distilled model at a small quality cost. — [arXiv 2408.14837v2](https://arxiv.org/html/2408.14837v2)
- Limitation: the model sees "a little over 3 seconds of history" and can forget state that is not on screen. Longer context gives only marginal gains. It may spawn enemies incorrectly when the player shoots repeatedly. — [arXiv 2408.14837v2](https://arxiv.org/html/2408.14837v2)
- Quality: next-frame PSNR 29.4. Humans are only slightly better than chance at telling short real clips from generated ones. — [arXiv abs](https://arxiv.org/abs/2408.14837)
- Data from the agent policy beat data from a random policy (PSNR after 3 s: 19.02 vs 16.84). The paper does not evaluate any policy inside the generated game. — [arXiv 2408.14837v2](https://arxiv.org/html/2408.14837v2)
- No official code or weights. A GitHub issue notes none were found. — [Awesome-REA issue #2007](https://github.com/reacher-z/Awesome-REA/issues/2007)

**Open reproductions and small Doom world models (2024-2026)**
- arnaudstiegler/gameNgen-repro (Apache-2.0): ViZDoom PPO agent, data generation, SD diffusion training, autoregressive rollouts. HF checkpoints `gameNgen-baseline-20ksteps` and `sd-model-gameNgen-60ksteps`. HF datasets `vizdoom-5-episodes-…` and `vizdoom-500-episodes-skipframe-4-lvl5`. The README says inference does not reach 20 FPS. It gives no hardware or compute details. — [GitHub](https://github.com/arnaudstiegler/gameNgen-repro)
- Masao-Taketani/GameNGen: unofficial reproduction built on the repo above. It adds more efficient diffusion training and a playable script. — [GitHub](https://github.com/Masao-Taketani/GameNGen)
- RohanNaga/Doom ("DoomDiT", references "April 2026 code"): compares DiT-XL/2 against the SD-1.4 U-Net as GameNGen-style backbones. It releases a lossless per-tic 320x240 ViZDoom deathmatch dataset recorded with the Arnold agent on 17 maps, with actions, health, ammo, kills, frags, x/y/angle. 10% of each map is held out, plus two fully held-out maps. It has a checkpoint release and no stated licence or metrics. — [GitHub](https://github.com/RohanNaga/Doom)
- jgalego/doom-world-model (Apache-2.0; Freedoom assets BSD-3): a 7.5M-param U-Net (EDM, 3 Euler steps). It takes the last 4 frames and actions and outputs 128x96 frames. It was trained on 200k frames from 217 random-policy episodes on Freedoom E1M1, in 16.5 min on one A10G. — [HF](https://huggingface.co/jgalego/doom-world-model)
- wendlerc/toy-wm: a toy diffusion-transformer world model for Pong or Doom, trained in latent space on PvP deathmatch data. — [GitHub](https://github.com/wendlerc/toy-wm)

**Dreamer / IRIS / DIAMOND / EDELINE**
- DreamerV3 (Jan 2023) reports Atari, DMC, Minecraft and others. My search found no ViZDoom result in the paper or its coverage. — [arXiv 2301.04104](https://arxiv.org/pdf/2301.04104); [GitHub danijar/dreamerv3](https://github.com/danijar/dreamerv3)
- R2I ("Mastering Memory Tasks with World Models", 2024) evaluates BSuite, POPGym, Atari 100k, DMC and Memory Maze, not ViZDoom. — [arXiv 2403.04253](https://arxiv.org/pdf/2403.04253)
- EDELINE (1 Feb 2025, rev. 15 Jun 2025) combines state-space models with a diffusion world model to extend memory beyond a fixed context. It claims gains on Atari 100k, Crafter and "3D first-person ViZDoom environments". A search snippet names the scenarios as Basic, DeadlyCorridor, DefendCenter, HealthGathering and PredictPosition, and says it beats DIAMOND on DeadlyCorridor and PredictPosition. The abstract alone does not confirm these details. — [arXiv 2502.00466](https://arxiv.org/abs/2502.00466)

### Inferences
- GameNGen-class models are trained per map and per policy, and forget after about 3 s. They cannot generate unseen Freedoom MAP05-15 mechanisms we do not already have, so they cannot teach route knowledge. Our real engine (ViZDoom/Freedoom) is already a perfect, fast simulator. Any world model would be a lossy copy of it.
- Fight rehearsal is better done in the real engine with save states or scripted scenario WADs than in a learned model. The Take Cover results show that dream-trained policies exploit model errors.
- The reusable parts are the data pipelines: the gameNgen-repro agent recording and the DoomDiT per-tic dataset with pose, HUD and frags. These suit imitation or offline training of a combat module.
- Training cost on our hardware: a 7.5M-param U-Net trains in minutes. SD-1.4-scale models (about 860M U-Net) would take days to weeks on an M-series GPU. GameNGen itself used 128 TPU-v5e for 700k steps.

### Gaps
- No official or credible DreamerV1-V3 or IRIS result on ViZDoom was found. DIAMOND's own benchmarks (Atari, CS:GO) were not verified here.
- No Doom world model on Apple Silicon, Core ML or MLX was found as of Oct 2026.
- No Oasis-like (Minecraft-style) commercial Doom world model was found beyond the reproductions above.
- The gameNgen-repro artifact sizes and training hardware are undocumented. EDELINE's numbers and code release were not verified.

## 2. Generalization benchmarks and train/test map splits

### Takeaway
Measured unseen-map results are few, and they come from combat or deathmatch, not from campaign levels with keys and switches. Arnold (2016) trained on 10 maps and tested on 3. It matched or exceeded its train-map K/D on the test maps, but those maps were simpler. LevDoom (2022) shows a consistent drop as visual and gameplay difficulty rises, with near-zero transfer to the hardest level. No published work measures generalization on held-out campaign maps such as Freedoom MAP05-15.

### Cited Findings
- Arnold (Lample & Chaplot, arXiv 1609.05521, AAAI 2017): in full deathmatch it trained on 10 maps and tested on 3, with randomised textures. With navigation, K/D was 4.58 on train maps and 6.94 on test maps (kills 66.8 vs 43.0, deaths 14.6 vs 6.0). The authors attribute the better test score to test maps having fewer stairs and level changes. Training "takes up to a week". At 65 h, the variant with game features reached K/D above 4 and the one without stayed under 2. — [arXiv 1609.05521](https://arxiv.org/pdf/1609.05521)
- Arnold repo: PyTorch, 17 maps, pretrained models for defend_the_center, health_gathering, track1, track2 and shotgun. The example command trains on maps 2-5 and tests on 6-8. No licence was shown on the page. — [GitHub glample/Arnold](https://github.com/glample/Arnold)
- ViZDoom competitions: Track 2 was full deathmatch on unseen maps. In the 2018 Track 2 (10 new unknown maps), the 2017 winner Arnold4 scored 128 frags and the 2016 IntelAct bot scored 95, against 342 for the 2018 winner. — [ViZDoom Competitions paper arXiv 1809.03470](https://arxiv.org/pdf/1809.03470) (figures via search summary of the paper and the [AIcrowd 2018 track page](https://www.aicrowd.com/challenges/visual-doom-ai-competition-2018-multiplayer-track-2), not re-verified in the table)
- LevDoom (Tomilin et al., IEEE CoG 2022): 4 scenarios (Defend the Center, Health Gathering, Seek and Slay, Dodge Projectiles), each with 5 difficulty levels built from 8 kinds of change (textures, enemy and item sizes and others). Agents trained on levels 0-1 were evaluated on levels 2-4. Rainbow was best in 16 of 28 environments, PPO in 10 and DQN in 2. PPO was very poor on Seek and Slay. "Performance in all scenarios drops as the difficulty level increases", and the level-4 curve "appears rather flat". Training ran about 10^7 timesteps on a GTX 1080 Ti with an i7-7700. — [LevDoom paper](https://ieee-cog.org/2022/assets/papers/paper_30.pdf); [GitHub TTomilin/LevDoom](https://github.com/TTomilin/LevDoom); [PyPI](https://pypi.org/project/LevDoom/1.0.0/)
- Affordance maps (Qi et al., ICLR 2020): used Oblige-generated ViZDoom maps with monsters, lava, barrels and unclimbable ledges, 60 train and 15 test. Affordance-augmented A* improved navigation success on unseen maps by about 45% over a geometry-only planner (the paper says "up to 55%" elsewhere). A PPO waypoint policy did not beat any affordance-based method. Absolute success rates appear only in a figure. — [arXiv 2001.02364](https://arxiv.org/html/2001.02364v2)
- Doom level GANs (Giacomello, Lanzi, Loiacono, arXiv 1804.09154, Apr 2018): GANs trained on images and features from about 1,000+ community maps from idgames. They "capture intrinsic structure" of Doom levels. A 2023 conditional follow-up found that room count and walkable-area features steer generation. Generated levels were not released. — [arXiv 1804.09154](https://arxiv.org/abs/1804.09154v1); [PCGamesN](https://pcgamesn.com/doom/ai-map-design-doom)
- Other: transformer-based DRL in VizDoom reports better generalization with larger training house sets and with segmentation inputs. — [ResearchGate](https://www.researchgate.net/publication/363062570_Transformer-Based_Deep_Reinforcement_Learning_in_VizDoom); a related Nov 2025 paper on semantic segmentation in ViZDoom — [arXiv 2511.11703](https://arxiv.org/pdf/2511.11703)

### Inferences
- The affordance-map result is closest to our design: a classical planner plus a learned hazard and affordance layer that generalizes to unseen Oblige maps. It suggests learning per-map "what is passable or dangerous" from self-supervised interaction, not end-to-end policies.
- Oblige-generated maps give a cheap, unlimited held-out set. A curriculum of generated maps with keys, doors and switches is a plausible way to train and evaluate mechanism handling beyond MAP05-15, and it keeps the real Freedoom maps as the final test.
- In combat, Arnold's unseen-map numbers suggest a learned combat policy transfers across deathmatch maps if textures are randomised.

### Gaps
- There are no published success rates for single-player campaign completion on held-out maps by any learned agent.
- I did not verify whether Sample Factory or Arnold trained on Oblige sets. Arnold used 17 curated maps.
- No Obstacle-Tower-style curriculum benchmark in Doom was found beyond LevDoom. Per-scenario numbers in LevDoom's Table V were not extractable.

## 3. Multi-agent and self-play combat training (as a combat module)

### Takeaway
Sample Factory (ICML 2020) gives the strongest open Doom combat policies. PBT agents beat the hardest built-in bots in 100% of Deathmatch and Duel matches. Self-play duel agents beat bot-trained ones 78-3 (19 ties). Pretrained checkpoints are on HuggingFace. Compute is large (2.5×10^9 frames per agent on a 36-core, 4-GPU server). Small fine-tunes are feasible, and the checkpoints could serve as a short-horizon combat module that the planner hands control to.

### Cited Findings
- Sample Factory: 10^5+ env FPS on one machine (up to 130k FPS). Hardware: System 1 is a 10-core CPU with a GTX 1080 Ti, System 2 a 36-core CPU with an RTX 2080 Ti. — [arXiv 2006.11751](https://arxiv.org/abs/2006.11751) (full text from PDF)
- Duel and Deathmatch: an action space of about 1.2×10^4 combinations including weapon switching and use (doors, lifts), with frameskip 2. A PBT population of 8 trained on a 36-core 4-GPU PC. Final agents beat the highest-difficulty bots in 100% of matches, scoring 80.5 vs 12.6 in Deathmatch and 34.7 vs 3.6 frags in Duel. — [arXiv 2006.11751](https://arxiv.org/pdf/2006.11751)
- Self-play duel (FTW-style): 8 agents at 2.5×10^9 frames each, about 18 years of experience in total. PBT mutated the bottom 70% every 5×10^6 frames. Against the bot-trained agent the self-play agent went 78 wins, 3 losses, 19 ties. The bot-trained agent "overfitted to a single opponent type". — [arXiv 2006.11751](https://arxiv.org/pdf/2006.11751)
- Pretrained HF models (Sample Factory 2): `andrewzhang505/sample-factory-2-doom-battle` (59.37 ± 3.93), `…-doom-battle2` (36.40 ± 4.20), `andrewzhang505/doom_deathmatch_bots` (85.66 ± 28.53), `andrewzhang505/doom_duel_bots_pbt` (55.39 ± 17.13), `andrewzhang505/doom-duel-selfplay` (commits Nov 2022). Battle training uses 4×10^9 env steps. The deathmatch-bots run caps at 3.6M s (about 1,000 h). Throughput is 50k+ FPS on a 10-core i9-7900X. — [Sample Factory docs](https://www.samplefactory.dev/09-environment-integrations/vizdoom/); [HF doom-duel-selfplay](https://huggingface.co/andrewzhang505/doom-duel-selfplay/tree/74ff6fb676219133fea79e9bba855a5ba59b7b4b)
- Community HF course checkpoints, e.g. `RyanAA/rl_course_vizdoom_doom_deathmatch_bots` (APPO, mean reward 2.30 ± 1.42), are much weaker. — [HF](https://huggingface.co/RyanAA/rl_course_vizdoom_doom_deathmatch_bots)

### Inferences
- A combat module is plausible: when enemies are visible and the planner flags a fight, hand control to a Sample Factory policy (conv encoder plus GRU, on pixels) for N seconds, then return to the navigator. These policies are trained on deathmatch maps against players and bots. Freedoom monsters (projectiles, melee, hitscan) differ, so fine-tuning on Freedoom fight scenarios is likely needed.
- On Apple Silicon there is no CUDA, and ViZDoom rendering is CPU-bound. A 10-core-class M-series machine might reach tens of thousands of FPS (unmeasured). That makes 10^8-10^9-frame fine-tunes take days, not hours. A full self-play run from scratch at the paper's scale is impractical locally.
- The Sample Factory model is small (conv plus RNN), so a Core ML or ANE port is technically feasible. Its action space, observation (128x72 RGB in SF defaults, unverified) and frameskip would need matching.

### Gaps
- The Sample Factory licence was not verified on the page fetched (believed MIT, unconfirmed). Checkpoint licences on HF were not checked.
- No published work combines a learned combat policy with a symbolic navigator or planner in Doom single-player. Arnold's action/navigation split (DQN for navigation, DRQN for combat, switched by enemy detection) is the nearest precedent. — [arXiv 1609.05521](https://arxiv.org/pdf/1609.05521)
- No measured Sample Factory throughput on Apple Silicon was found.

## 4. Small or fast models for real-time control on edge devices

### Takeaway
The clearest recent Doom data point is SauerkrautLM-Doom-MultiVec (arXiv 2604.07385, Apr 2026). A 1.3M-param model gets 178 frags against 13 for all LLMs combined on defend_the_center, at 31 ms per decision on CPU. It supports small specialist combat policies over LLMs for reflex tasks. I found no Doom policy deployed on the ANE or Core ML.

### Cited Findings
- SauerkrautLM-Doom-MultiVec (Golchinfar, Vaziri, Marquardt; arXiv submitted 8 Apr 2026): a 5-layer ModernBERT encoder with hash embeddings, 1,319,300 params, about 5 MB FP32. Each frame becomes a 40x25 ASCII brightness grid plus 16-bin depth embeddings, mapped to 4 actions. It was trained on 31,645 frames (about 2 h of one human, defend_the_center) on one RTX 6000 Ada, reaching 57.7% top-1 validation accuracy. — [arXiv 2604.07385](https://arxiv.org/html/2604.07385v1)
- Results over 10 episodes: SLM 178 frags at 31 ms; Gemini Flash Lite 8 frags at 920 ms; Nemotron-120B 3 at 8.9 s; Qwen3.5-27B 2 at 13.3 s; GPT-4o-mini 0 at 646 ms. In "fair mode" (frameskip 20) the SLM scored 10 and Flash Lite 5. Caveats: one scenario, one demonstrator, fewer episodes for the slow LLMs. The Raspberry Pi claim was untested. — [arXiv 2604.07385](https://arxiv.org/html/2604.07385v1)
- Ha & Schmidhuber's Take Cover controller is 1,088 parameters on top of a 6M-param VAE plus RNN, showing that tiny controllers work over a learned latent. — [worldmodels.github.io](https://worldmodels.github.io/)
- jgalego's 7.5M-param world model trains in 16.5 min on an A10G, showing small-scale Doom world models are cheap. — [HF](https://huggingface.co/jgalego/doom-world-model)

### Inferences
- A Laya-adjacent combat head (about 1-10M params, ANE-hosted, at 10-35 Hz instead of 2 Hz) trained by behaviour cloning on recorded fights (DoomDiT data, Sample Factory rollouts or our own logs) is the most compute-feasible combat improvement. Training takes minutes to hours on an M-series machine.
- The 57.7% accuracy and single-scenario caveats mean such a head would need on-policy correction (DAgger-style, or RL fine-tuning in ViZDoom) before it is robust.

### Gaps
- No published Core ML or ANE deployment of a Doom policy or world model was found. The SLM-Doom repository URL and licence were not given in the paper text.
- No quantized (int8/4-bit) Doom policy benchmarks were found.

## 5. Repos, licences, checkpoints (summary list)

### Takeaway
The usable open assets are the Sample Factory Doom checkpoints (HF), Arnold (17 maps plus 5 pretrained models), LevDoom (PyPI), gameNgen-repro (Apache-2.0, HF data and checkpoints), DoomDiT (deathmatch dataset with pose and HUD), and jgalego/doom-world-model (Apache-2.0, Freedoom-based). Licence status is unverified for several.

### Cited Findings
- Sample Factory Doom models (5 repos listed above) — [Sample Factory docs](https://www.samplefactory.dev/09-environment-integrations/vizdoom/)
- Arnold: PyTorch, 17 maps, 5 pretrained models, no licence shown — [GitHub](https://github.com/glample/Arnold)
- LevDoom: Gymnasium API, PyPI — [GitHub](https://github.com/TTomilin/LevDoom); [ViZDoom third-party envs](https://vizdoom.farama.org/environments/third_party/)
- gameNgen-repro: Apache-2.0 — [GitHub](https://github.com/arnaudstiegler/gameNgen-repro); Masao-Taketani/GameNGen — [GitHub](https://github.com/Masao-Taketani/GameNGen)
- DoomDiT: per-tic deathmatch dataset on 17 Arnold maps, licence not stated — [GitHub](https://github.com/RohanNaga/Doom)
- jgalego/doom-world-model: Apache-2.0, Freedoom E1M1 — [HF](https://huggingface.co/jgalego/doom-world-model)
- toy-wm — [GitHub](https://github.com/wendlerc/toy-wm)

### Inferences
- Priority for our bot: (1) Sample Factory deathmatch or duel checkpoints as a combat-module baseline, or as a teacher to distil into an ANE head; (2) Oblige-generated maps for curriculum and held-out evaluation; (3) the affordance-map approach for learning hazards and passability beside the hand-written navigator. World-model and GameNGen work is low priority for this project.

### Gaps
- Licences for Arnold, DoomDiT, the Sample Factory HF checkpoints and LevDoom were not confirmed.
- Whether the Sample Factory checkpoints run under current Sample Factory 2.x and ViZDoom 1.2+ on macOS arm64 was not tested.
