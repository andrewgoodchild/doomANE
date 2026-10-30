# Classic scripted Doom bots, map analysis and navigation tooling (non-learning approaches)

Scope: what we can borrow for (a) modelling linedef and sector specials and building a key, switch, door, lift and exit progression graph, and (b) scripted combat that keeps a bot alive. Research date 2026-10-24. doomwiki.org returned HTTP 403 or a Cloudflare challenge to every fetch, so Doom wiki pages are cited only from search snippets, or not at all.

## 1. Source-port bots: how they navigate and fight

### Takeaway
The ZDoom "Cajun" bot (still in GZDoom, and also in ViZDoom) is a deathmatch bot. Its navigation is a monster-style chase toward the current item or enemy; it has no route planner. Its combat loop is small and worth copying almost line for line: dodge missiles by backpedalling and strafing, strafe with a direction flip when stuck, keep a weapon-specific distance, react with a delay, lead projectiles, and fire rockets only when the impact point is far enough away. Polge's Quake ReaperBot learned its node graph by exploring. Zandronum bots use map-placed nodes and compiled bot scripts. ZetaBot (GZDoom ZScript) is the most complete modern node-graph bot and supports coop.

### Cited Findings
**ZDoom/GZDoom Cajun bot** (files `b_bot.h`, `b_think.cpp`, `b_move.cpp`, `b_func.cpp`, `b_game.cpp`)
- Source: [GZDoom src/playsim/bots](https://github.com/ZDoom/gzdoom/tree/master/src/playsim/bots). The file headers say "Copyright 1999 Martin Colberg, 1999-2016 Randy Heit, 2005-2016 Christoph Oelckers" under a **3-clause BSD-style licence**, so the code can be borrowed with attribution — [b_think.cpp](https://github.com/ZDoom/gzdoom/blob/master/src/playsim/bots/b_think.cpp)
- Movement is `DBot::Roam`, which turns toward `dest` (an item, weapon or enemy) and otherwise calls `NewChaseDir`. That function copies monster chase-direction tables from `p_enemy.cpp`, so the bot moves like a monster in 8 directions and has no global path planner. `b_bot.h` mentions "paths created between items" (`MAXTHINGNODES 100`) — [b_move.cpp](https://github.com/ZDoom/gzdoom/blob/master/src/playsim/bots/b_move.cpp), [b_bot.h](https://github.com/ZDoom/gzdoom/blob/master/src/playsim/bots/b_bot.h)
- The priority order each tic in `b_think.cpp` ("HOW TO MOVE") is:
  1. **Missile avoidance.** If a tracked missile is within `AVOID_DIST` (about 686 map units, `45000000/65536`), face it, strafe at full run speed and run backwards ("Back IS best"). If position changed less than `THINKDISTSQ`, flip the strafe direction with a 5-tic cooldown. Keep firing if the enemy is within `SHOOTFOV`.
  2. **Fight** when the enemy is visible (`P_CheckSight`). First check whether an item is worth grabbing mid-fight: a Medikit, Stimpack, Soulsphere or Megasphere when health is below the skill threshold, any powerup, or any item closer than `GETINCOMBAT/4`, provided it is `Reachable`. Otherwise face the enemy. Move forward if it is beyond the weapon's `MoveCombatDist`, otherwise back away. Strafe side to side, walking against monsters and running against players. Randomly flip the strafe direction (1 in 30) or flip when stuck. Melee weapons disable `MF_DROPOFF` so the bot does not jump off ledges mid-fight.
  3. **Follow a teammate**, then **roam**.
  — [b_think.cpp](https://github.com/ZDoom/gzdoom/blob/master/src/playsim/bots/b_think.cpp)
- Constants in `b_bot.h`: `SHOOTFOV 60°`, `ENEMY_SCAN_FOV 120°`, `SAFE_SELF_MISDIST 140`, `MAXMOVEHEIGHT 32`, `AFTERTICS 2 s` (stay alert on an enemy that just went out of sight), `MAXROAM 4 s` (change roam target after this), `DARK_DIST` and `WHATS_DARK 50` (enemies in sectors darker than light level 50 are seen only up close) — [b_bot.h](https://github.com/ZDoom/gzdoom/blob/master/src/playsim/bots/b_bot.h)
- `Dofire` in `b_func.cpp`:
  - It does not fire while `damagecount` exceeds a skill threshold, i.e. while the screen is still red from being hit.
  - A reaction delay `t_react = (100-reaction+1)/(rand%3+3)` applies to the first shot.
  - Melee weapons fire only within 4× melee range.
  - For projectile weapons it aims at `enemy.pos + enemy.vel * (dist/projectile_speed) * 2`.
  - For rockets, `FireRox`/`FakeFire` spawns an invisible trace projectile and fires only if the impact point is at least 140 units away and the aim is within 12°.
  — [b_func.cpp](https://github.com/ZDoom/gzdoom/blob/master/src/playsim/bots/b_func.cpp)
- In non-deathmatch games, `Set_enemy` keeps the current enemy and looks for a new one only when it has none (comment: "[RH] Don't even bother looking for a different enemy if this is not deathmatch") — [b_think.cpp](https://github.com/ZDoom/gzdoom/blob/master/src/playsim/bots/b_think.cpp)
- ViZDoom (based on ZDoom) ships the same Cajun bot code (`src/vizdoom/src/b_bot.cpp`, `b_game.cpp`) and uses it as built-in deathmatch opponents via `addbot` in `examples/python/cig_multiplayer_bots.py` — [ViZDoom repo](https://github.com/Farama-Foundation/ViZDoom/blob/master/examples/python/cig_multiplayer_bots.py)

**ZetaBot (GZDoom, ZScript)**
- It has five states (Wandering, Following, Attacking, Hunting, Fleeing), an "advanced pathing system" with more than 13 path-node types, path editing during play, saving and loading of node networks, and per-map node lumps. It aims to be "competent in both coop and deathmatch" in Doom, Heretic and Strife. GitHub reports its licence as NOASSERTION (a LICENSE file exists but its type was not identified) — [ZetaBot](https://github.com/zeta-group/ZetaBot)

**Zandronum / Skulltag bots**
- They are a separate implementation from Cajun, said to have been written from scratch by Carnevil (forum recollection) — [Doomworld thread](https://www.doomworld.com/forum/post/1367444)
- Behaviour comes from compiled bot scripts. `botc` (compiler) and `debotc` (decompiler of the bot bytecode lumps in Zandronum's PK3) exist, and the author says new coop modes would need engine changes — [Zandronum forum](https://zandronum.com/forum/viewtopic.php?p=115780)
- The ACS `AddBot` function fails "if the map disables bot nodes", which implies node-based navigation — [Zandronum wiki AddBot](https://wiki.zandronum.com/AddBot)

**Quake ReaperBot (Steven Polge, 1996)**
- The bot "learns levels as it plays", and understands doors, teleporters and triggers — [Reaper beta readme](https://discmaster.textfiles.com/file/11997/IMG%2051%20Vol%205-8.iso/Goodies/More%20For%20Your%20Games/Quake/MODs/Reaper%20Bots/Reaprb80.txt)
- A Doomworld article classifies the Reaper as a bot that drops its own nodes while exploring. Its coverage is only as good as its exploration. It contrasts this with Omicron (nodes dropped along a human player's path) and Frogbot (hand-coded nodes) — [Doomworld "mellow" articles](https://www.doomworld.com/mellow/articles.htm)
- Polge owns the copyright, and its code may not be reused in public mods (fan-site claim) — [InsideQC](https://www.insideqc.com/qctut/rip1.shtml)

### Inferences
- The Cajun combat block is roughly 100 lines and maps directly onto our executor. Each item below is cheap and attacks the deaths-per-run problem:
  - missile within ~690 units: backpedal and strafe
  - strafe while engaged, flipping direction when stuck or at random
  - hold weapon-specific engage distances
  - stop firing and reset reaction after a heavy hit
  - lead projectile shots
  - check rocket splash with a fake projectile
  - grab health mid-fight when hurt and the item is reachable
- ViZDoom exposes projectiles as Objects, so the AVOID_DIST missile-dodge rule can be built on `state.objects` without engine changes.
- The Reaper/Omicron idea of self-built node graphs matches our failure mode: on a new map, we could record the cells and edges actually traversed, including lift rides and teleports, as learned edges that survive deaths, since each death resets the level.

### Gaps
- Eternity Engine and Odamex: no source found describing built-in bots. I believe neither has one, but this is not verified.
- "Doombot by Andrew Apted" and "Autodoom": no source found. Apted's verified work here is Oblige (below).
- Zandronum bot navigation internals (node format, path search) are not documented in the sources reached.

## 2. Bots that finish single-player levels

### Takeaway
The closest public analogue to our stack is **doom-mcp** (Rust plus doomgeneric, last pushed 2026-10-06). It plans on a 16-unit grid using Doom's real movement rules and automatically chains prerequisites (key, switch or walk-over trigger, including stair builders). A rule-based pilot fights. It completes shareware E1M1 and E1M2, and it produces a route or a chained plan for every E1 level except E1M8. Its known holes are lifts (handed back to the model), boss-triggered exits and some trigger types: almost exactly ours.

### Cited Findings
- doom-mcp's pilot fights each tic:
  - target priority: commander's focus, then the current target, then the most threatening (aware of us, then closest)
  - best weapon for the range (never the BFG; no rockets inside 256 units)
  - fires once the crosshair is within the monster's angular size (at least 2.5°)
  - strafes, flipping every 30 tics
  - backs off from melee monsters (Demon, Spectre, Lost Soul) closer than 160 units
  - closes in when it can only punch
  - 35-tic weapon-switch cooldown

  Weapon order: plasma, super shotgun up to 500, shotgun up to 600, chaingun, shotgun, rockets, pistol, chainsaw — [combat.rs](https://github.com/arthurjdam/doom-mcp/blob/HEAD/src/pilot/combat.rs)
- Travel: it follows waypoints, turns in place on sharp corners, stops to open doors, waits for moving floors, unsticks itself (it must move a minimum distance every N tics) and optionally rides lifts in phases (wait for it to lower, step on, ride until it has risen and stopped) — [travel.rs](https://github.com/arthurjdam/doom-mcp/blob/HEAD/src/pilot/travel.rs)
- "Playing E1M1 on its own, the pilot clears the level, all 6 monsters included, in about 26 s of game time." It says it is "competent against episode 1's monsters, but it doesn't manage health or ammo strategically" — [doom-mcp README](https://github.com/arthurjdam/doom-mcp)
- Licence: GitHub detects no top-level licence. The README states the embedded doomgeneric is GPL-2.0, "which makes the combined binary GPL-2.0 too" — [doom-mcp README](https://github.com/arthurjdam/doom-mcp)
- ZRDOOM Enhanced 3.0 is a Doom 2 mod in which a bot plays generated maps, and the layouts are changed so the bot can get unstuck. It is a mod, not a general solver — [itch.io](https://vicious-games.itch.io/zrdoom-enhanced-30)

### Inferences
- doom-mcp's nav.rs (about 1,500 lines, Rust) solves the same problem as our navigator, and its specials tables and relaxed-search prerequisite method (section 3) can be ported to Python. Because of GPL-2.0, read it as a design reference unless our repo is GPL-compatible; porting the ideas is safe.

### Gaps
- No public scripted bot was found that reliably completes arbitrary full Doom II or Freedoom levels with monsters. No public multi-map completion rates were found, consistent with the memory note on Jev's Doom results.

## 3. Progression and solvability: key, switch, door, lift and exit graphs

### Takeaway
There are three borrowable designs:
1. doom-mcp's **relaxed search plus first-obstacle prerequisite chaining** over a geometric grid.
2. crustygen's **state-space BFS over (sector, keys held)**, which catches one-way drops that strand the player; a plain flood over places and items misses them.
3. Oblige/Obsidian's **quest tree**: a binary tree of locked connections, with goals of kind KEY or SWITCH and a backtrack list.

No dedicated "WadSolver" or "DoomLevelSolver" tool was found.

### Cited Findings
**doom-mcp prerequisite planner** ([nav.rs](https://github.com/arthurjdam/doom-mcp/blob/HEAD/src/world/nav.rs))
- The grid uses 16-unit cells. Cells closer to a wall than about the player's 16-unit radius are blocked, and cells near walls cost extra. Moving between cells checks every two-sided linedef crossed: step up at most 24, headroom at least 56, openable doors (with the key), lowerable lifts, hazard floors, solid things and teleporters.
- When the goal is unreachable, it re-searches with a `Relax` set that assumes locked doors and triggered sectors can open, excluding keys and sectors already being sought in the current chain. It takes the first relaxed crossing on that path and routes to its prerequisite: the key, or the switch, walk-over or gun line whose tag moves that sector. Chains are depth-limited. Example: "E1M4 is switch → blue key → yellow key → switch → exit".
- A sector counts as "opened by a trigger" when a line with that tag has a special not in `NOT_OPENING`. Lifts lower to the lowest neighbouring floor (`lift_low`).
- Stair builders mirror `EV_BuildStairs`: from each tagged sector, follow two-sided lines to back sectors with the same floor texture.
- Teleporters resolve to the `MT_TELEPORTMAN` thing in the tagged sector.
- One-shot specials are cleared once used, read live from the engine.
- `debug_cut` prints the actual cut: refused edges between "reachable from player" and "can reach goal". This is a diagnostic we could copy for MAP04.
- Vanilla special tables it uses (Doom II numbers):
  - `MANUAL_DOORS` [1,31,117,118]
  - `LIFTS` [10,21,62,88,120,121,122,123]
  - `STAIRS` [7,8,100,127]
  - exits: switch 11, walk 52, secret switch 51, secret walk 124 (also 197, 198 in map.rs)
  - `TELEPORTS` [39,97]
  - locked doors [26,27,28,32,33,34,99,133–137]
  - `HAZARD_SECTORS` [4,5,7,11,16]
  - `GUN_SPECIALS` [24,46,47]
  - `WALK_SPECIALS` (full list in file)
  - `NOT_OPENING` [3,6,11,12,13,16,17,25,35,39,41,42,43,44,49,50,51,52,57,73–77,79,80,81,97,104,107,110,113,116,124,125,126,138,139,141]

  — [nav.rs](https://github.com/arthurjdam/doom-mcp/blob/HEAD/src/world/nav.rs), [map.rs](https://github.com/arthurjdam/doom-mcp/blob/HEAD/src/world/map.rs)

**crustygen reachability (Rust, MIT/Apache-2.0, last pushed 2026-10-21)**
- Key finding: "Reachability of *places* and *items* is not the property; the property is over **states** — (where the player is, which keys they hold)". A set-union flood wrongly passes a map where the only key sits in a pit reached by a one-way drop. The fix is a product-graph BFS over sectors × 2^keys, which is small — [docs/reachability.md](https://github.com/masriamir/crustygen/blob/HEAD/docs/reachability.md)
- Crossing rules, verified against vanilla source:
  - A line is passable if `min(ceilings) − max(floors) ≥ height` (`P_LineOpening` plus `P_TryMove`).
  - The step limit of 24 applies to stepping up only.
  - The player has `MF_DROPOFF`, so drops of any height are allowed.
  - "37.77% of vanilla's passable two-sided lines exceed the step cap", so one-way drops are common.

  — [docs/reachability.md](https://github.com/masriamir/crustygen/blob/HEAD/docs/reachability.md)
- Its lifter (`crustygen-lift`) reads UDMF and binary WADs. Its recognisers:
  - teleporters: resolved the way `EV_Teleport` does — "walk the sectors carrying the tag in declaration order; the first holding a destination marker wins"
  - platforms
  - a histogram census of linedef and sector specials across a corpus

  — [docs/lift.md](https://github.com/masriamir/crustygen/blob/HEAD/docs/lift.md), [README](https://github.com/masriamir/crustygen)

**Oblige/Obsidian quest system (Lua, GPL-2.0-or-later)**
- `scripts/quest.lua` (Andrew Apted, 2006-2017) arranges quests in a binary tree. QUEST leaves are room sets, and QUEST_NODEs split at a connection (`before`/`after`/`conn`). A GOAL has a `kind` (KEY, SWITCH, START, EXIT, SECRET_EXIT), an `item`, a `room`, a `backtrack` list of rooms the player must pass back through, and for switches an `action` special and a `tag`. Teleporter connections can never be locked. Other functions include `Quest_order_by_visit` and `Quest_find_backtracks` — [quest.lua](https://github.com/obsidian-level-maker/Obsidian/blob/master/scripts/quest.lua)
- Obsidian is a fork of the last Oblige release, with ObAddon merged in — [Doom wiki OBSIDIAN (snippet)](https://doomwiki.org/wiki/OBSIDIAN)

**Academic**
- Giacomello, Lanzi and Loiacono (2018) trained GANs on more than 1,000 Doom levels using images (floor map, height map, walls, things) plus topological features. There is no explicit key or lock graph — [arXiv:1804.09154](https://arxiv.org/abs/1804.09154)
- VGLC contains a Doom WAD parser and rasteriser (SVG or tile output) — [VGLC arXiv:1606.07487](https://arxiv.org/pdf/1606.07487)

### Inferences
- Recommended design for our planner: a BFS over states (sector or region, keys held, set of triggers fired), using crustygen's crossing rules and doom-mcp's mapping from triggers to tagged sectors. When the goal is unreachable, run the relaxed search and target the first blocked edge's prerequisite. Add per-map "learned edges" for mechanisms the model gets wrong.
- Modelling one-way drops explicitly (allowed down, not up) should stop the planner from sending the bot into a pit and then "exploring".

### Gaps
- No tool named WadSolver, DoomLevelSolver or doom-level-graph was found. UDB and SLADE map-analysis features for progression were not documented in the sources reached.
- Oblige's `doc/Quests.txt` overview is referenced in quest.lua but was not found in the current Obsidian or Oblige repos.

## 4. Linedef and sector special tables, simulator code, and reading sector state live from ViZDoom

### Takeaway
For vanilla specials, use Chocolate Doom's `p_spec.c`/`p_switch.c`/`p_floor.c`/`p_plats.c`/`p_doors.c` as the reference (GPL-2). Boom generalized types (0x2F80–0x7FFF) are bit-packed and can be decoded in about 50 lines. ViZDoom gives live floor and ceiling heights per sector each state (so lifts and doors can be observed moving) and line `is_blocking`, but **no line special, tag or sector special**. Those must come from the WAD (omgifol, MIT) and be joined by index.

### Cited Findings
- Boom generalized ranges:
  - crushers 0x2F80–0x2FFF
  - stairs 0x3000–0x33FF
  - lifts 0x3400–0x37FF
  - locked doors 0x3800–0x3BFF
  - doors 0x3C00–0x3FFF
  - ceilings 0x4000–0x5FFF
  - floors 0x6000–0x7FFF

  Bit fields:
  - trigger bits 0–2
  - speed bits 3–4
  - floor/ceiling: model bit 5, direction bit 6, target bits 7–9, change bits 10–11, crush bit 12
  - door: kind bits 5–6, monster bit 7, delay bits 8–9
  - locked door: kind bit 5, lock bits 6–8 (any/rc/bc/yc/rs/bs/ys/all), skull=card bit 9
  - lift: monster bit 5, delay bits 6–7, target bits 8–9 (LnF/NnF/LnC/perpetual)
  - stairs: step bits 6–7, dir bit 8, ignore-texture bit 9

  — [Boom reference v1.3](https://doomlegacy.sourceforge.net/docs/boomref.html)
- The trigger enum order is `WalkOnce, WalkMany, SwitchOnce, SwitchMany, GunOnce, GunMany, PushOnce, PushMany` (W1, WR, S1, SR, G1, GR, D1, DR = 0..7). `GenFloorBase 0x6000`, `TriggerType 0x0007` — [dsda-doom p_spec.h](https://github.com/kraflab/dsda-doom/blob/master/prboom2/src/p_spec.h)
- Boom extended (non-generalized) types relevant to routing:
  - perpetual lifts 181/162/87/53, stop 182/163
  - instant lift toggle 211/212
  - raise 24/32 by texture 148/143, 149/144
  - elevators: next-higher 227–230, next-lower 231–234, current floor 235–238
  - teleporters: 195/174 thing; silent 207–210 thing, preserves angle; 243/244 and 262–267 line-to-line
  - scrollers 250–255 (floor and carry types relevant to "scrolling floors"), 245–249 displacement, 214–218 accelerative
  - friction 223, wind/current 224/225, point push 226
  - deep water 242

  — [Boom reference v1.3](https://doomlegacy.sourceforge.net/docs/boomref.html)
- Chocolate Doom dispatch entry points: `P_CrossSpecialLine` (walk-over, p_spec.c line ~518), `P_ShootSpecialLine` (~997), `P_PlayerInSpecialSector` (~1047), and `P_UseSpecialLine` in p_switch.c (~263). These big switch statements are the authoritative vanilla tables of special → action — [chocolate-doom p_spec.c](https://github.com/chocolate-doom/chocolate-doom/blob/master/src/doom/p_spec.c), [p_switch.c](https://github.com/chocolate-doom/chocolate-doom/blob/master/src/doom/p_switch.c)
- omgifol (Python WAD library) is MIT-licensed and was last pushed 2026-07 — [omgifol](https://github.com/devinacker/omgifol)
- ViZDoom game state:
  - `Sector`: `id`, `floor_height`, `ceiling_height`, `lines`. Floor height was negative before 1.4.0.
  - `Line`: `x1`, `y1`, `x2`, `y2`, `is_blocking`.
  - `Object`: `id`, `name`, `category` (1.3.1+), `sector_id` (1.3.1+), plus position.
  - `Label`: `object_id`, bounding box in `labels_buffer`.

  Enable with `set_sectors_info_enabled` and `set_objects_info_enabled` — [ViZDoom game_state.md](https://github.com/Farama-Foundation/ViZDoom/blob/master/docs/api/python/game_state.md)
- In ViZDoom's source, sectors are rebuilt every state update when `viz_sectors` is on (`if (*viz_sectors) VIZ_GameStateUpdateSectors();`):
  - Sector `id` is the engine index `i`.
  - Heights come straight from the live `floorplane`/`ceilingplane`.
  - `isBlocking` = `ML_BLOCKING|ML_BLOCKEVERYTHING|ML_BLOCK_PLAYERS`.
  - `frontSector`/`backSector` are commented out. `VIZLine` carries no special or tag.

  — [viz_game.cpp](https://github.com/Farama-Foundation/ViZDoom/blob/master/src/vizdoom/src/viz_game.cpp), [viz_game.h](https://github.com/Farama-Foundation/ViZDoom/blob/master/src/vizdoom/src/viz_game.h)

### Inferences
- Practical recipe:
  - Parse LINEDEFS, SIDEDEFS and SECTORS with omgifol or our own parser to get specials, tags and front/back sectors.
  - Join by index: ViZDoom sector `id` equals the engine sector index, which should equal the WAD SECTORS order for binary-format maps.
  - Poll `floor_height`/`ceiling_height` each step to know when a door is open, a lift is down or a floor has risen, and use that to confirm that a switch had an effect.
  - `Object.sector_id` gives the sector of the player and of keys, for state-space planning.
- Line indices in ViZDoom are assigned via `lineIds`, presumably engine linedef order. Verify this on MAP04 by comparing endpoints with the WAD.
- Diffing `floor_height` before and after a `use` is a cheap generic detector of "switch did something to tag N" for unmodelled specials.

### Gaps
- Whether ViZDoom's line list order equals WAD LINEDEFS order was not confirmed from source (it is built through a `lineIds` map).
- ViZDoom exposes no flag for used switches or one-shot triggers. That state must be tracked in our code, or inferred from switch texture changes (not exposed) or from sector movement.

## 5. Speedrun and TAS knowledge for Freedoom Phase 2 MAP04

### Takeaway
Freedoom Phase 2 MAP04 is "Containment Bay" by Mark "Macro11_1" Pedersen. No speedrun route, TAS or walkthrough for it was found.

### Cited Findings
- "MAP04: Containment Bay by Mark "Macro11_1" Pedersen" (MAP03 is "Crude Processing Center", MAP05 is "Sludge Burrow") — [Freedoom CREDITS-LEVELS](https://github.com/freedoom/freedoom/blob/master/CREDITS-LEVELS)
- The Doom wiki page for Mark Pedersen lists Freedoom Phase 2 MAP04 (2005) — [Doom wiki (snippet)](https://doomwiki.org/wiki/Mark_Pedersen_(Macro11_1))

### Inferences
- Freedoom maps are revised between releases, so any route must come from the exact WAD we run. Self-analysis with the state-space planner plus `debug_cut`-style output is more reliable than an external route.

### Gaps
- No DSDA demos, TAS or written walkthrough for Freedoom2 MAP04 were found. doomwiki.org blocked fetches, so a Doom wiki page for the map, if one exists, was not read.
