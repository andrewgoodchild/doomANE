# LLM, VLM and agentic systems that play Doom (2023 to October 2026)

Research date: 2026-10-24. About 20 web searches and fetches. "Verified" means I read the primary page (repo, PR or arXiv) in this session. "Claimed" means a vendor's or press claim, or something I did not re-check. Items marked [training knowledge] come from my prior knowledge; I did not re-fetch them this session.

## Catalogue: Doom-specific LLM/VLM agents

### Takeaway
Only two published LLM-in-the-loop Doom agents have completed a level more than once. The first is azalio's doomLaya (Laya/Jev): command-level, it uses WAD geometry, and it has cleared MAP01-03 in single runs. The second is lucas-moont's doom-player (Claude Opus 5.5 over MCP): 5/5 on E1M1 with automap and notebook, with the game paused between actions and about 12.6M tokens per attempt. Agents that read raw pixels and press raw keys (de Wynter's GPT-4V, VideoGameBench) clear 0% of levels.

### Cited Findings

**1. azalio/doomLaya (Laya and Jev on Freedoom, 2026)**: https://github.com/azalio/doomLaya (verified)
- Licence Apache-2.0. 16 commits, 4 stars, 2 forks. Releases v0.1.0 to v0.3.1. Comparison dated 22 Sep 2026. The idea came from a Telegram channel post. — [doomLaya README](https://github.com/azalio/doomLaya)
- Perception and state: the model gets a **text state, not images**. Both models receive the **WAD geometry and the exit position**. Command options include specific visible monsters and items already seen. The README says it does not test image-only control or per-button neural control. — [README](https://github.com/azalio/doomLaya)
- Action interface: by default, two typed questions (command + weapon). A `factorized` mode asks separately for action, target, weapon and combat movement. A `committed` mode adds the current target and an explicit "continue" command. Commands include `exit` (route to the exit), `explore` (nearest unvisited area) and `open_door`. — [README](https://github.com/azalio/doomLaya)
- Controller: hand-written. It does aiming, pathfinding, obstacle avoidance and button presses, but leaves item value, combat priority and weapon choice to the model. Auto weapon switch on pickup is disabled. Unreachable commands are excluded after a real pathfinding failure and restored after the position or a door changes. Responses older than 2 s, or from another episode, are dropped. The game keeps running during requests. — [README](https://github.com/azalio/doomLaya)
- Keys, lifts and switches: the README mentions only doors (intermediate doors need an explicit `open_door`). Nothing on keys, lifts or switches. — [README](https://github.com/azalio/doomLaya)
- Results (single seeds): v0.3.1 cleared MAP01 168.4 s, MAP02 420.4 s, MAP03 347.1 s, 0 deaths. v0.3.0 cleared MAP03 in 819.9 s with 4 deaths, but the quality check failed (141.83 s of stillness), and a regression check failed MAP01/MAP02 on earlier seeds. v0.2.0: MAP01 to MAP02 in 174.9 s; MAP02 to MAP03 in 835.6 s with 2 deaths; `passed=false`. — [README](https://github.com/azalio/doomLaya)
- Jev vs Laya (v0.1.0, MAP01, skill 3, seed 48, **one run each**): Jev 1.13 exited in 68.97 s with 13 kills, 0 deaths, HTTP p50/p95 357/445 ms, $0.00458 over 139 responses. Adapted Laya v3 took 61.43 s with 12 kills, 0 deaths, 108/133 ms, $0. Original Laya did not finish in 180 s. On seed 46, Laya adaptation v2 failed and Jev passed in 51.49 s. The authors say this does not show either model is generally better. Jev was not trained on the task; Laya was. — [COMPARISON.en.md](https://github.com/azalio/doomLaya/blob/main/docs/COMPARISON.en.md)

**2. TypeSafe Jev Doom demo (launched 2026-09-15)**: claimed
- Jev is a "System One" typed-decision model: developers define the answers, and it returns choices, scores or yes/no with calibrated confidence. Up to 255 choices. $0.042/M input tokens; output free. — [letsdatascience](https://letsdatascience.com/news/typesafe-ai-launches-jev-decision-model-889a38c0), [startuphub](https://startuphub.ai/ai-news/artificial-intelligence/2026/typesafe-jev-model-kills-chat)
- Launch demo: a Doom bot making about 10 structured queries/s to Jev, at about $7/hour. No independent evals at launch. — [letsdatascience](https://letsdatascience.com/news/typesafe-ai-launches-jev-decision-model-889a38c0)
- Earlier project research (memory note, 2026-10-23, not re-checked): the launch post asked typed FIRING/GOAL/DODGE questions and claimed no level completion.

**3. lucas-moont/doom-player (Claude Opus 5.5 vs PPO vs hybrid, Oct 2026)**: https://github.com/lucas-moont/doom-player (verified)
- 117 commits, 0 stars, no licence shown. ViZDoom; bring your own `doom.wad`. Compares an LLM through a Doom MCP server, PPO from screen pixels, and a planned **hybrid "fast learned Driver steered by a slow LLM Navigator"** (described; no results yet). — [repo](https://github.com/lucas-moont/doom-player)
- PR #28 (merged 3 Oct 2026), E1M1, skill 3, 5 seeds:
  - H0 (screen + actions): 1/5 cleared, progress 0.48, 24.4M tokens, 23.7 min per attempt.
  - H1 (+ automap): 4/5, 0.98, 5.3M tokens, 7.0 min.
  - H2 (+ automap + notebook): 5/5, 1.00, 12.6M tokens, 11.7 min.
  - Random: 0/5.
  - In total, 211M tokens and 3.5 h of play. The author says the H1 vs H2 gap (one attempt) is within chance. — [PR #28](https://github.com/lucas-moont/doom-player/pull/28)
- Failure modes: H0 got lost looping near the start (3 seeds) or ran out of time on the right route (1). H1 died in combat once, at progress 0.89. An H0 seed took 3461 tics over 193 actions, about 18 tics per action. The PR contrasts its setup with "a paused game", which implies pausing, but the text I read does not state it outright. — [PR #28](https://github.com/lucas-moont/doom-player/pull/28)
- PPO cleared E1M1 15/15 from pixels (PR #31, 4 Oct). On E1M2 ("keys, lifts", PR #32), PPO with key-shaped reward reached 80% on two 10M-step runs (PR #34, 10 Oct); most other E1M2 runs got 0%. With exit reward only, PPO learns nothing. — [repo README](https://github.com/lucas-moont/doom-player), [PR list](https://github.com/lucas-moont/doom-player/pulls?q=is%3Apr)

**4. de Wynter, "Will GPT-4 Run DOOM?" (arXiv 2403.05468, 8 Mar 2024)**: verified
- Architecture: a Manager (cyDoomgeneric Python binding plus a Matplotlib UI; keeps move history), Vision (GPT-4V turns 640x400 screenshots into structured text, including the HUD), Agent (GPT-4 picks one keystroke action per turn; unparsable output becomes WAIT), Planner (GPT-4 turns a **hand-written level walkthrough** into a step plan every 30 frames), and "Experts" (k=2 extra agent calls every 60 frames). Memory is a short history inside a 32K context, about 16 screenshots. — [arXiv HTML](https://arxiv.org/html/2403.05468v1)
- Results: E1M1, Hurt Me Plenty, 10 runs per prompt, **0 completions**. Naive: 40% death / 60% timeout, rarely left room 1. Walkthrough: 90% / 10%. Plan: reached room D, died to an imp; 90% / 10%. K-levels: reached room C consistently; once found a secret room but could not backtrack; 80% / 20%. — [arXiv HTML](https://arxiv.org/html/2403.05468v1)
- Failure modes: forgot enemies once they left view; poor aim; shot walls and barrels; walked into acid despite instructions; got stuck on walls and corners; hallucinated rationales; about 1 minute per frame with the naive prompt. Code and videos at https://adewynter.github.io/Doom. — [arXiv HTML](https://arxiv.org/html/2403.05468v1)
- Press coverage ("ChatGPT can play Doom, but it's terrible"). — [Decrypt](https://decrypt.co/221840/can-chatgpt-play-doom-yes-but-its-terrible)

**5. VideoGameBench (Zhang et al., arXiv 2505.18134, May 2025; v3 May 2026)**: verified
- Doom II is in the test set (the original Doom is in dev). DOS games run in JS-DOS in a browser driven by Playwright. Model calls are asynchronous to real-time play. A "Lite" mode pauses the game while the model thinks. The agent keeps "reflections / internal memory". MIT licence; 375 stars. — [GitHub](https://github.com/alexzhang13/VideoGameBench)
- Raw pixels, high-level description of controls and objectives, no game-specific scaffolding. Best overall: 0.48% (full) and 1.6% (Lite). The authors name inference latency as a major limit. — [arXiv](https://arxiv.org/abs/2505.18134)
- **Doom II: 0% for Gemini 2.5 Pro, GPT-4o, Claude 3.7 Sonnet, Llama 4 Maverick and Gemini 2.0 Flash, in both full and Lite.** Models show basic combat and navigation but fail at strategic planning, spatial reasoning and tracking objectives. — [vgbench.com](https://www.vgbench.com/)
- Launch posts (Apr 2025): no model passed the first level. Sonnet 3.7 got furthest ("found the blue room") and kept shooting already-dead enemies. — [r/LocalLLaMA mirror](https://redlib.ssps.io/r/LocalLLaMA/duplicates/1k28f3f), [Decrypt ES](https://decrypt.co/es/315636/relajate-sigues-siendo-mejor-jugando-doom-que-la-inteligencia-artificial)

**6. gunnargrosch/doom-mcp (2026)**: verified
- MCP server that embeds doomgeneric through FFI. Three tools: `doom_start(skill, episode, map)`, `doom_action(actions, ticks 1-105, default 7)` and `doom_screenshot`. Each action returns text (HP, armour, ammo, kills, position, weapon, **enemies with direction, distance and HP, nearby items, doors and switches in range**) plus a roughly 6 KB PNG thumbnail. Virtual time, so the game advances only on calls. No automap. MIT (doomgeneric GPL-2.0, Freedoom BSD). 19 commits, 13 stars. — [GitHub](https://github.com/gunnargrosch/doom-mcp)
- Results: the author reports about 5-10 kills per session on E1M1 medium, no level completions. Struggles with enemies behind partial cover and "complex door sequences"; better when a human directs it. Death ends the session. — [GitHub](https://github.com/gunnargrosch/doom-mcp)

**7. Not found / unrelated**
- "Rootly Doom Agent Arena": no evidence that it exists. Rootly's MCP server is for incident management, not Doom. — [getdrio Rootly MCP](https://www.getdrio.com/mcp/io-github-wyre-technology-rootly-mcp/md)
- "Softmax ViZDoom Deathmatch league": no evidence found. Searches return only the 2016-2018 Visual Doom AI Competitions (RL from pixels; the 2018 Track 1 was finishing randomly generated single-player levels). — [ViZDoom Competitions paper](https://arxiv.org/abs/1809.03470)
- "Intermission" (Jarrod Watts) is a Claude Code mod that lets *humans* play Odamex deathmatch while Claude thinks. It is not an agent. — [WeAreDevelopers](https://www.wearedevelopers.com/magazine/780-play-doom-with-developers-while-claude-code-works.md)
- "Paul Bauer GPT-4V Doom": I found nothing under that name. The 2024 "GPT-4 plays Doom" stories all trace back to de Wynter. — [80.lv](https://80.lv/articles/gpt-4-destroys-everyone-on-its-way-in-doom)

### Inferences
- Results line up with how much structure the agent gets. Raw pixels with raw keys gives 0% (VideoGameBench, de Wynter). Pixels plus automap plus a notebook, with the game paused, gives 100% on one map, at about 12M tokens and 12 min per attempt. Text state from WAD data with typed commands and a scripted controller gives real-time clears of MAP01-03 at 2-10 Hz. Our bot is in the third group, and there is no published comparison point beyond Freedoom MAP01-03.
- Automap matters more than anything else in doom-player. Adding it moved the clear rate from 20% to 80% and cut tokens about 5x. That supports giving the planner or LLM a map-level, route-level representation instead of the current view.
- No published LLM or VLM Doom agent reports results on a held-out map. Our 0/18 on MAP05-15 (and doomLaya v0.3.1's 0/18) is new information, not a regression against a published baseline.

### Gaps
- No public Jev level-completion rate beyond doomLaya's one run per seed; TypeSafe's own demo post was not re-read this session.
- Whether doom-player's MCP server pauses the game is implied but not stated in the text I read. Its notebook format and how it encodes the screen were not shown.
- No source gives multi-map or held-out-map completion rates for any LLM/VLM agent.

## System 1 / System 2 splits (fast reflex controller + slow planner)

### Takeaway
Fast/slow designs are common in robotics, navigation and open-world games. Few target an FPS: doom-player plans one but has no results yet. Recurring patterns: the slow layer runs on events (stuck, failure, new area), its plans are cached over many fast steps, and the fast layer can override a stale plan.

### Cited Findings
- **SPIKE** (arXiv 2605.18636, 18 May 2026): a fast Reactive Controller with a strict token budget, plus a slow Strategic Controller for global planning, failure analysis and recovery. An **Event Trigger** (visual change, task progress, repeated actions, failure signals) escalates to the slow layer. Strategic proposals are reused over several reactive steps, and the reactive layer can override stale plans. Memory is short-term (State-Action Memory Bank) plus a State-Action Knowledge Graph. On StarDojo Lite-100: +5.0 points success, +9.3 points budgeted success, 54.9% fewer tokens, 40.8% lower latency. Ablations show the trigger, the override and the memory each help. — [arXiv](https://arxiv.org/abs/2605.18636), [project](https://wencanjiang.github.io/projects/SPIKE/)
- **Lumine** (ByteDance Seed, arXiv 2511.08892, Nov 2025): Qwen2-VL-7B. Perceives pixels at 5 Hz and emits keyboard and mouse at 30 Hz through action chunks. **Reasons only when it judges it necessary** ("adaptive thinking"). Finishes Genshin Impact's 5-hour Mondstadt storyline at about human efficiency. Transfers zero-shot to Wuthering Waves and Honkai: Star Rail, less efficiently there. 2400 h of gameplay data. — [arXiv](https://arxiv.org/abs/2511.08892v1)
- **doom-player hybrid**: "a fast learned Driver steered by a slow LLM Navigator". Planned, no results (as of PR #35, 10 Oct 2026). — [repo](https://github.com/lucas-moont/doom-player)
- **DualVLN "Ground Slow, Move Fast"** (Dec 2025): a slow VLM global planner predicts mid-term waypoints, and a fast diffusion-transformer policy follows them. Navigation, not games. — [Liner review](https://liner.com/review/ground-slow-move-fast-dualsystem-foundation-model-for-generalizable-visionandlanguage)
- **Fast-in-Slow** (arXiv 2506.01953, Jun 2025): a fast execution module inside a VLM; 117.7 Hz control (robot manipulation). — [arXiv](https://arxiv.org/pdf/2506.01953)
- de Wynter's Doom agent already had a crude slow/fast split: a Planner every 30 frames over a walkthrough, plus a per-frame Agent. It was still far too slow to play in real time. — [arXiv HTML](https://arxiv.org/html/2403.05468v1)
- [training knowledge, not re-fetched] Voyager (Minecraft, arXiv 2305.16291) uses an automatic curriculum, a **skill library of verified code**, and iterative prompting with environment feedback. It acts through Mineflayer's API, not pixels. — [arXiv](https://arxiv.org/abs/2305.16291)
- [training knowledge, not re-fetched] Cradle (arXiv 2403.03186) is general computer control from screenshots plus keyboard and mouse, with reflection, task inference, skill curation and memory modules (RDR2 and others). SIMA (arXiv 2404.10179) is DeepMind's language-instructed agent across 3D games, using about 10-second skills. — [Cradle](https://arxiv.org/abs/2403.03186), [SIMA](https://arxiv.org/abs/2404.10179)

### Inferences
- Lessons for our bot: (a) make the Qwen/System 2 call event-triggered (objective failure, stuck timer, new area, death) rather than periodic; we already partly do this. (b) Cache its plan and let the executor override it when stale; SPIKE's ablations show the override matters. (c) Keep a persistent per-map memory across deaths (the SA-KG idea, a doom-player-style "notebook", Voyager's skill library), so route knowledge such as "lift at X, press switch Y first" survives the level reset.
- Voyager-style verified-skill libraries map naturally onto our executor: store a sequence that worked (for example "ride lift: stand on sector, use, wait") as a reusable typed macro the decision model can pick.

### Gaps
- No FPS-specific System 1/System 2 paper with results was found. CS:GO agents (e.g. the 2021 behaviour-cloning work) are pure imitation policies, not LLM hybrids. I did not re-verify any CS:GO LLM agents.
- BALROG and LMGame-Bench: [training knowledge, not re-fetched] BALROG (arXiv 2411.13543) covers BabyAI, Crafter, TextWorld, BabaIsAI, MiniHack and NetHack, and LMGame-Bench (arXiv 2505.15146) covers Mario, Sokoban, Tetris, 2048, Candy Crush and Ace Attorney. As far as I know, **neither includes Doom**. Not confirmed this session.

## Keys, doors, switches, lifts and exits: how systems handle mechanisms

### Takeaway
Doom LLM agents mostly stop at doors. The agents that succeed rely on external map knowledge: WAD geometry and the exit position (doomLaya), the automap (doom-player), or a hand-written walkthrough (de Wynter). No published LLM Doom agent shows general key, switch or lift reasoning on unseen maps. The only lift/key result is RL with key-shaped reward on E1M2.

### Cited Findings
- doomLaya: WAD geometry and the exit position are given to the model, and there is an explicit `open_door`. Keys, lifts and switches are not mentioned. — [README](https://github.com/azalio/doomLaya)
- doom-mcp: exposes "doors/switches within range" in the text state, and `use` opens doors and switches. "Complex door sequences" remain a weak point. — [GitHub](https://github.com/gunnargrosch/doom-mcp)
- doom-player: the automap took E1M1 clears from 20% to 80%, and the notebook to 100%. The current milestone M5 is "Maps with keys" (an E1M2 ruler for keys, lifts and an eval spec). Only PPO results exist there so far: 80% on 2 runs with key shaping, mostly 0% otherwise. — [PR #28](https://github.com/lucas-moont/doom-player/pull/28), [PRs](https://github.com/lucas-moont/doom-player/pulls?q=is%3Apr)
- de Wynter: a hand-written walkthrough was required for any map progress. It found a secret room once but could not backtrack. — [arXiv HTML](https://arxiv.org/html/2403.05468v1)
- VideoGameBench: models fail at "keeping track of objectives" and spatial reasoning. — [vgbench.com](https://www.vgbench.com/)

### Inferences
- For unmodelled mechanisms, the available evidence favours parsing the WAD linedef specials (switch, walk-over, lift and door trigger types, sector tags, key-locked line types) into a typed objective graph, rather than asking an LLM to discover them visually. A "switch S raises lift L to reach exit" dependency is statically derivable for most vanilla maps. The LLM's job would then be to choose among graph objectives, which its proven skill (doom-player h1/h2) suggests it can do.
- Our 2026-10 navigator and planner work (blocked edges, keys that list the switches they unlock) already goes further than any published LLM Doom harness I found.

### Gaps
- No source describes an LLM/VLM Doom agent that parses linedef specials or sector tags for lifts and switches.

## Success rates, failure modes, combat and deaths

### Takeaway
Combat death is a leading failure mode everywhere: de Wynter had 40-90% deaths, doom-player's H1 died at 0.89 progress, and doomLaya v0.3.0 had 4 deaths on MAP03. VLM agents also forget enemies that leave the view and waste ammo on corpses. Nobody publishes deaths per minute in a form comparable to our 6-12 per 20 min.

### Cited Findings
- de Wynter: deaths in 40-90% of runs depending on prompt; enemies forgotten once off-screen; poor aim; barrels and acid. — [arXiv HTML](https://arxiv.org/html/2403.05468v1)
- VideoGameBench: Sonnet 3.7 kept shooting dead enemies. Latency is especially damaging in FPS games. — [Decrypt ES](https://decrypt.co/es/315636/relajate-sigues-siendo-mejor-jugando-doom-que-la-inteligencia-artificial), [arXiv](https://arxiv.org/abs/2505.18134)
- doomLaya: zero-death clears in v0.3.1 but deaths in v0.2.0/v0.3.0, and long stillness (141.83 s) flagged as a quality failure. — [README](https://github.com/azalio/doomLaya)
- doom-mcp: struggles with enemies behind partial cover; death ends the session. — [GitHub](https://github.com/gunnargrosch/doom-mcp)
- PPO from pixels cleared E1M1 15/15 in about 0.8-0.9 s of compute per attempt. A cheap learned reflex policy can do the combat and movement that LLMs fail at. — [doom-player](https://github.com/lucas-moont/doom-player)

### Inferences
- Combat should belong to System 1 (a scripted or RL reflex layer: aim, strafe, retreat, target priority at 35 Hz), with the decision model only choosing engage, avoid or retreat. doomLaya's `factorized` mode, with a separate "combat movement" question, is one published version of this.
- Since each death resets the level, persistent cross-death memory (a notebook) is the cheapest published lever for raising completion: doom-player's H1 to H2 step removed its one combat-death failure, though n=5.

### Gaps
- No agent reports deaths per minute, damage taken or hit rate.

## Repos, licences and activity (summary)

### Takeaway
All the Doom LLM repos are small (0-13 stars) and from 2026, except VideoGameBench (375 stars, MIT) and de Wynter's 2024 code page.

### Cited Findings
- azalio/doomLaya: Apache-2.0; 16 commits, 4 stars; v0.1.0 to v0.3.1, Sep 2026. — [GitHub](https://github.com/azalio/doomLaya)
- lucas-moont/doom-player: no licence shown; 117 commits; PRs from 1 to 10 Oct 2026, PR #35 open. — [GitHub](https://github.com/lucas-moont/doom-player/pulls?q=is%3Apr)
- gunnargrosch/doom-mcp: MIT (plus GPL-2.0 doomgeneric); 19 commits, 13 stars; npm 0.1.x. — [GitHub](https://github.com/gunnargrosch/doom-mcp)
- alexzhang13/VideoGameBench: MIT; 375 stars, 43 forks, 24 commits. — [GitHub](https://github.com/alexzhang13/VideoGameBench)
- de Wynter: code and videos at adewynter.github.io/Doom; uses cyDoomgeneric (github.com/wojciech-graj/cyDoomgeneric). — [arXiv HTML](https://arxiv.org/html/2403.05468v1)

### Inferences
- doom-mcp and doom-player's MCP server are the most reusable pieces for an "LLM as System 2 over MCP" test. doomLaya is the nearest peer for direct comparison: same Freedoom maps, typed commands, Apache-2.0.

### Gaps
- I did not check the last-commit dates for doomLaya or doom-mcp.
