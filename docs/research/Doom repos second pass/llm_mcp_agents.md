# Doom repos, second pass: LLM/MCP agents and doomLaya upstream (2026-10-24)

I read the source of four repos, cloned shallowly into the session scratchpad at `repos2/<name>`. I did not run or install any of it. Paths and line numbers below refer to those clones.
"Ours" means `ext/doomLaya-dev` plus `doom/*.py` in this repo.

Authority key: **FACT** = code may give it to the model as a fact or option (allowed). **EXEC** = executor or navigator mechanics (allowed). **SCRIPT** = code makes the choice (not allowed as is; at most it can be turned into a fact).

---

## 1. azalio/doomLaya: upstream after v0.3.1

- **Nothing new.** `main` HEAD is `3a1e514` (v0.3.1, 2026-09-28). The tags are v0.1.0 to v0.3.1. The only other branch, `codex/map02-release` (78fcc7e), is an ancestor of main. GitHub `pushed_at` is 2026-09-28T05:40Z.
- Forks: `fox3000foxy/doomLaya` and `sabaki-js/doomLaya` have no commits beyond main. The one PR (#1, a reorganisation) is closed and already merged.
- `diff -rq` against `ext/doomLaya-v031` shows only local artefacts (.venv, checkpoints, runs, __pycache__). Our v0.3.1 copy matches upstream, so there are no upstream fixes to port.
- Licence: Apache-2.0.

---

## 2. arthurjdam/doom-mcp (Rust + doomgeneric C shim; last commit 2026-10-05)

Licence: the README says the vendored doomgeneric is GPL-2.0, "which makes the combined binary GPL-2.0 too". There is no separate licence for the Rust code. Treat it as GPL-2.0: **copy ideas, not code.**

This repo was not in the first-pass survey. It is the closest design to ours: 16-unit grid, step 24, 56 headroom, prerequisite chaining, a `debug_cut`. We appear to have taken the navigator ideas already. What follows is what we still lack.

### Architecture
- In real-time mode, a rule-based **pilot** plays every tic and the LLM acts as **commander**. The commander gives standing `Orders`: goal, stance, focus target, move/hold, a one-shot use. It wakes on important events (`ARCHITECTURE.md`; `src/game/realtime.rs`; `src/game/events.rs`).
- Event log (`src/game/events.rs:1-80`): sequence-numbered, with an `important` flag that wakes `wait_for_events` early. Identical text is de-duplicated within 350 tics.
- Important events: stuck, no route, door won't open, out of ammo, died, level briefing (`realtime.rs:215-249`).

### Combat (all SCRIPT in their code; the facts behind it are what we can use)
- **Attacker attribution** (`csrc/dmcp.c:273-277`, `src/world/observe.rs:281-306` `diff_events`). It reads `player->attacker` and reports damage as "You took N damage (health a -> b), attacked by Imp at bearing +135°". It works for attackers that are off screen. **We have nothing like this** (no `attacker` anywhere in doomlib). Unseen attackers cause 13-31% of our damage.
- **Awareness** (`dmcp.c:322`): `targeting_player = mo->target == player && alive`. Observations carry `targeting_you` per monster (`observe.rs:121-122`). Things are sorted with threats in line of sight first, then threats out of sight, then everything else (`observe.rs:219-227`). Monsters within 384 units are listed even without line of sight (`observe.rs:176`).
- Target pick (`src/pilot/combat.rs:101-121`): focus target, then the current target (sticky), then aware-before-unaware, then nearest. Stance filter (`combat.rs:84-99`): aggressive engages all; balanced engages aware or within 1024; cautious engages only aware monsters and backs off inside 320; hold_fire never fires.
- Fire gate (`combat.rs:169-176`): fire when |bearing| <= atan(20/dist), clamped to 2.5-10°.
- Movement (`combat.rs:178-201`): strafe direction flips every 30 tics. Back off from Demon, Spectre and Lost Soul within 160. Back off when rocket-armed within 256. Close in when only fists are available.
- Weapon (`combat.rs:215-250`): rockets are "usable" only beyond 256. Fixed priority plasma > SSG (≤500) > shotgun (≤600) > chaingun > ... Weapon switch cooldown is 35 tics.

### Navigation and map knowledge (EXEC)
- One static grid. **Edges are checked against live sector heights**: `Level::dynamic` (`src/world/nav.rs:522-545`) snapshots sector floors and ceilings, current line specials (one-shot triggers clear themselves once used) and keys each query. `enter_normally` (`nav.rs:570-627`) and `edge` (`nav.rs:630-665`) apply step, headroom and door rules to those heights. **The grid is never rebuilt.**
- **Relaxed A\*** (`nav.rs:667-776`, `route_with` 991-1130):
  - First, a normal search.
  - If it fails, a single relaxed search in which locked doors and trigger-moved sectors are passable at a soft cost. Each costs +2048, against +64 for a door and +256 for a lift.
  - The first `Door(locked)` or `Trigger(sector)` crossing on that path becomes the prerequisite. The planner recurses on it with that key or sector forbidden, up to 6 deep.
  - So each step is one goal-directed search, not a fixpoint over the whole map.
- Path costs: hazard sectors (specials 4, 5, 7, 11, 16) ×4; cells near a wall (`COMFORT` 40 units) ×1.6 (`nav.rs:754-760`). Our `navigation.py:296` has neither; hazards are only a fact (`floor_hazards.py`).
- Stairs (`nav.rs:432-463`) mirror `EV_BuildStairs`: they follow two-sided lines to neighbours with the same floor texture. Our specials table has 'stairs', but check whether the solver moves the whole chain or only the tagged sector.
- Teleports are graph edges to the MT_TELEPORTMAN cell (`nav.rs:465-490`, `search` 708-718). Shoot-triggered lines are special-cased (GUN_SPECIALS 24, 46, 47).
- Explore goal (`nav.rs:1006-1020`): any open cell next to a line without `ML_MAPPED`. This is an automap frontier. ViZDoom does not expose the mapped flag; our visited-cells approach is the nearest equivalent.
- Boss levels: no handling. It just says "On boss levels the way out opens once the bosses are dead" (`nav.rs:1027-1031`). Tags 666/667 are unmodelled here too.

### Travel, stuck handling, death (EXEC)
- `src/pilot/travel.rs`:
  - Stuck check: less than 12 units moved in 10 tics (`travel.rs:482-510`).
  - If a sector between the player and the waypoint is moving (`mover_between`, 535-544), it waits up to 140 tics. Otherwise it backs up 6 tics, then sidesteps 8 tics toward the waypoint side. A second failure raises the `Blocked` event.
  - A jump of more than 64 units in one tic is a teleport. An unexpected teleport triggers a replan.
  - Lift phases: use, then wait until `floor - z <= 24`, then board, then ride until the lift stops; then replan.
  - Doors are "open" when `ceiling - floor >= 56`.
- Death: respawn after 3 s (`realtime.rs:22-23`). On (re)start, a **level briefing** (`src/game/session.rs:130-200`) gives the exit route length, or the prerequisite text when there is no direct route, plus locked-door colours and the keys on the level. It is marked "(restarted)" after a death. Nothing else carries over between deaths.

### Prompting and state
- Compact text plus JSON, with signed bearings ("+ = right of crosshair"). The model keeps its own plan with `set_plan`, and the plan is echoed in every observation (`observe.rs:404-412`). A "hint" field tells the model what to do in menu, dead and intermission states.

---

## 3. gunnargrosch/doom-mcp (Rust + doomgeneric; MIT for the Rust part, doomgeneric GPL-2.0; last commit 2026-03-14)

This is a turn-based MCP server: `doom_start`, `doom_action(actions, ticks)`, `doom_screenshot`. All the cleverness is in feedback text built by `src/main.rs` `GameTracker` (351-477):
- **Stale-fire / behind-cover detection** (`main.rs:655-690`). If the enemy nearest the crosshair keeps the same HP across 3 or more consecutive firing steps at |angle| ≤ 20°, it reports "not taking damage — likely behind cover. Advance or reposition." (FACT)
- **Recently-seen memory** (`main.rs:712-728`). For 3 steps after enemies leave view, it reports "Recently seen nearby: Imp (HP:60) was ~hard left close". (FACT; ours has `last_seen` only for the current target, `combat.py:43-64`)
- **Receding-item suppression** (`main.rs:589-603`). An item whose distance grows by more than 10 for 2 consecutive steps is hidden as "unreachable". It also hides "dominated" items: health at 100 HP, armour at 100 armour (`main.rs:584-588`). (FACT, if rewritten as "approach failed" rather than hiding.)
- Stuck: 2 or more movement steps with less than 5 units moved produces a "STUCK" hint (`main.rs:402-420`).
- Danger: "TOO CLOSE" under 100 units. A critical-HP hint (≤20) says retreat with strafe+backward, never turn in place (SCRIPT as written).
- Interactables (`csrc/platform.c:322-400`): usable door, locked-door and exit lines within 400 units and ±60° are listed with key colour.
- Natural-language direction and distance buckets plus "turn_left ~N ticks" hints (`main.rs:626-653`).
- Death ends the session; there is no level knowledge beyond what is in view; there is no automap. The author reports no level completions.

---

## 4. lucas-moont/doom-player (Python, ViZDoom; **no licence file**, so all rights reserved: read only, do not copy)

There are three contenders: Claude over MCP, PPO from pixels, and a planned learned-Driver plus LLM-Navigator hybrid (ADR 0001, no results yet).
- **MCP harness** (`src/doom_player/mcp_server.py`):
  - Tools: `look`, `act(buttons, tics≤35)` and `automap` (the screen PNG of the Tab map).
  - Rung h2 adds a **notebook**: `write_note` / `read_notes`, up to 50 notes of 500 chars (`mcp_server.py:114-128`, 205-215). Its prompt says the notebook "survives even if your earlier messages are summarised away" (`harness.py:45-50`).
  - Each action reports HUD-diff events: damage, health, armour, ammo, weapon (`mcp_server.py:147-161`).
  - The prompt is a "game manual" only, with measured turn rates: 10/20/35 tics ≈ 10/45/100° (`harness.py:12-38`).
  - Results (from the first pass): h0 1/5, h1 (automap) 4/5, h2 (notebook) 5/5 on E1M1.
- **Progress ruler** (`progress.py`):
  - 8-unit grid, step 24, headroom 56. Doors that can be opened (manual and tagged 46/103) count as open, and lift sectors are not ledges (`_walls` 160-188).
  - Multi-source Dijkstra from the exit line's front side, cached per (map, locked colours, goal) with `@cache` (`distance_field` 236-281).
  - **KeyedRoute** (`progress.py:~100-125`): remaining route = min(direct to exit with the keys held, leg to key k + route from k with K∪{k}). It is memoised over at most 2^3 key sets. Which keys are needed is never written by hand.
- **Training lessons** (`maps.py:105-138` ProgressShaping; `docs/milestones/M4`, `M5`):
  - Shaping pays only new best distance (no farming), with −25 on death.
  - The "doors-open" reward trapped PPO at the red door. The keyed reward got it 80% on E1M2.
  - RND novelty made things worse (0.2% vs 9.2%): the policy camped in an armour alcove.
  - PPO policies "learned to leave out fighting". The remaining failures were deaths near the exit to monsters "mostly out of view" (M5:153).
- Death ends the attempt. There is no in-game stuck detection.

---

## Ideas we lack, ranked by expected impact for us

1. **Damage attribution fact: "hurt by <type> at bearing <b>, visible: no".** (arthurjdam `dmcp.c:273-277`, `observe.rs:297-304`)
   - Targets the 13-31% unseen-attacker damage.
   - FACT: the model still chooses turn, target and retreat. It also gives outcome-labelled training a direct "turn to attacker" label.
   - ViZDoom does not expose the attacker. In vanilla `P_DamageMobj`, `target->target = source` applies to the player mobj too (ZDoom does the same). So a small ACS library loaded through LOADACS can, each tic, set the activator to the player's AAPTR_TARGET and copy x/y/class into ACS globals. ViZDoom reads those as USERn variables. This route is unverified; check the ZDoom 2.8 ACS pointer semantics first.
   - Fallback: infer the bearing from the damage tic plus known projectiles and hitscanners in the labels buffer, and label "unseen" when none are visible.
2. **Route solver speed: stop rebuilding the Navigator per heights tuple.**
   - Ours: `route_solver.py:55-60` caches a full `Navigator` per heights tuple, and `planner.py:168,186` builds trial navigators.
   - arthurjdam keeps one grid and evaluates crossings against a live heights array (`nav.rs:522-665`). It finds prerequisites with one relaxed A* (soft +2048 trigger and lock costs, then recurse; `nav.rs:991-1130`) instead of a forward fixpoint.
   - lucas-moont's memoised multi-source distance fields (`progress.py:236-281`, KeyedRoute) give cheap leg costs.
   - EXEC, no authority issue. This is the best lead on MAP12's 2.5 h.
3. **Cross-death memory and restart briefing.**
   - Death resets our level, but nothing in our stack is told what happened.
   - Add typed facts at restart: where and how the last deaths happened (attacker type, side, whether seen), triggers already known to matter, and the route found. Also the arthurjdam-style briefing: exit route or its prerequisite, locked colours and keys (`session.rs:130-200`).
   - lucas-moont's notebook rung was the step from 4/5 to 5/5 (`mcp_server.py:114-128`).
   - FACT (memory supplied as facts; the model chooses).
4. **Awareness fact per monster ("targeting you" / awake) and listing near monsters without line of sight.** (`dmcp.c:322`, `observe.rs:176,219-227`)
   - This lets the model rank threats it cannot see and tell "dormant, skip it" from "hunting me".
   - FACT, and it fits our enemy-ranking question.
   - Needs the same ACS route as idea 1 (monster target == player). A weaker fallback: monsters within 384 units of the player that have moved since the last check.
5. **Stale-fire / behind-cover and approach-failed facts.** (gunnargrosch `main.rs:655-690`, 589-603)
   - "Target HP unchanged after N shots", "item distance growing for 2 steps", and "recently seen: X, side, distance" for a few seconds after enemies leave view (`main.rs:712-728`).
   - FACT. This cuts wasted ammo and time spent exposed while shooting walls. It is cheap to add from labels and object health.

Lesser ideas:
- Hazard ×4 and near-wall ×1.6 path costs (EXEC; `nav.rs:754-760`).
- Mover-wait before unstick (`travel.rs:486-492`).
- Teleport-jump detection that forces a replan.
- Stance as a standing option the model picks (aggressive/balanced/cautious/hold_fire), so the executor carries it between decisions. FACT/option; the thresholds would be executor parameters, not choices.
- The keyed-progress metric (best distance through keys), for eval and outcome labels.

Not found anywhere: boss-map handling (tags 666/667), secret routes, savestate-based learning, sound cues.
