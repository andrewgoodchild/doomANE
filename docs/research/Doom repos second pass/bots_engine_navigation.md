# Doom bots and engine movement rules: second-pass source review (2026-10-24)

Read-only review of source (shallow/sparse clones, not built or run) in
a scratch directory.
Each finding is checked against our code: `ext/doomLaya-dev/doomlib/navigation.py`, `doomlib/mission.py`,
`doomlib/specials.py` and `doom/route_solver.py`.

| Source | What was read | Licence |
|---|---|---|
| chocolate-doom `src/doom` (vanilla rules) | p_map.c, p_maputl.c, p_spec.c, p_floor.c, p_switch.c, p_telept.c, p_enemy.c, p_mobj.c, p_inter.c, info.c, g_game.c | GPL-2.0 |
| gzdoom `src/playsim` (the ZDoom family; **ViZDoom is a ZDoom 2.8 fork, so this is the closer reference**) | p_map.cpp (P_TryMove, P_UseTraverse), p_teleport.cpp, bots/b_*.cpp (Cajun) | GPL-3.0 overall; the Cajun `bots/` files carry a 3-clause BSD header (Colberg/Heit/Oelckers) |
| ZetaBot (zeta-group) | ZScript.zs, ZetaCode/Pathing.zs | MIT |
| crustygen (masriamir) | src/reach.rs, src/check/flood.rs, src/lift/{plat,teleport}.rs, docs/reachability.md, KNOWN-GAPS.md | MIT OR Apache-2.0 |
| Zandronum (Skulltag bots) | src/astar.cpp, src/botpath.cpp | Skulltag licence: BSD-like, but clause 4 requires free redistribution (a non-commercial condition, not OSI). Take ideas only, no code |
| Odamex / ZDaemon | Odamex has no bot code that I know of. ZDaemon is closed source. Not checked further | — |

---

## A. Bugs and gaps in our route knowledge (highest value)

### A1. Stairs are never applied, so their step-ups look unexplained
- `specials.destination()` has no `stairs` branch and returns `None`. `map_data` therefore gives stair switches an
  empty `moves`, and the solver logs "no move". It also moves only the *tagged* sector, but the stair builder raises a
  whole chain of sectors.
- Vanilla `EV_BuildStairs` (chocolate p_floor.c:454-560):
  - for each tagged sector: `height = floor + stairsize` (8 for 7/8, 16 for turbo 100/127);
  - then repeatedly look for a **two-sided line whose FRONT sector is the current sector** and whose **back sector has
    the same floor flat**. That back sector is the next step, set to `height += stairsize`;
  - the increment happens *before* the "already moving" check, so a skipped sector still uses up a step;
  - destination heights are **first floor + k·stairsize**, not each sector's own floor plus a step.
  ZDoom's Stairs_BuildUpDoom keeps these semantics (with a compat flag for the index quirk).
- **Fix:** add `stairs_chain(tagged, lines, sides, sectors)`, which returns `{sector: height}` for the whole chain, and
  use it in `map_data`. This is likely the main cause of "step-up >24 it can't explain".

### A2. Other movers the solver ignores
- **Gun (G1/GR) lines are filtered out.** `map_data` keeps only `rule.trigger in ('use','walk')`, so specials 24, 46
  and 47 are never offered.
  - Vanilla fires them from `PTR_ShootTraverse` when **any hitscan** crosses or hits the line (chocolate
    p_map.c:945 calls `P_ShootSpecialLine`, p_spec.c:997-1037).
  - Monsters can trigger only 46. Projectiles never trigger them: ZDoom needs MF2_IMPACT, see gzdoom p_map.cpp:2325-2336.
  - Fix: add a "shoot line X from a cell with line of sight to its midpoint" mechanism. The executor fires a hitscan
    weapon at it.
- **Donut (9)** is marked `other`. `EV_DoDonut` (p_spec.c:1283):
  - s1 = tagged sector, which lowers;
  - s2 = the sector across `s1->lines[0]`, which rises;
  - s3 = the back sector of the first line of s2 that is not s1;
  - both s1 and s2 move to **s3's floor**.
  Easy to add.
- **Raise to shortest lower texture (30/96)** returns `None`. Vanilla (p_floor.c:371) uses
  floor + the smallest bottom-texture height over the sector's two-sided lines. This needs texture heights from
  TEXTURE1/PNAMES. Even without them, an upper bound of floor+64 would be better than nothing.
- **40 (W1)** is `RaiseCeilingLowerFloor` in vanilla (p_spec.c:702-707: `EV_DoCeiling raiseToHighest` plus
  `EV_DoFloor lowerFloorToLowest`). It can *open* a low passage, but our table classes it as `ceiling`, which "never
  opens a route".
- **Boss specials (MAP07, plus the boss map).** In vanilla `A_BossDeath` (p_enemy.c:1671):
  - on MAP07, when every MT_FATSO is dead, tag 666 runs lowerFloorToLowest;
  - when every MT_BABY is dead, tag 667 runs raiseToTexture;
  - Keen deaths open tag 666 (A_KeenDie);
  - Icon-of-Sin death runs G_ExitLevel (A_BrainDie).
  ZDoom gets these from MAPINFO `map07special`, which ViZDoom applies to Freedoom 2. These sectors have **no
  linedef**, so a linedef-only solver can never explain them. Add pseudo-mechanisms "kill all <type>" with these tags.

### A3. Use, cross and teleport rules our executor and solver should mirror
- **Use range and geometry.**
  - `USERANGE` = 64 (p_local.h:53). The use ray is a single line from the player **centre along the yaw**, not a cone
    (P_UseLines, p_map.c:1170-1190; gzdoom p_map.cpp:5962-5970 uses the per-class UseRange, 64).
  - So the switch line must lie within 64 units of the centre, straight ahead. Our `approach` point is 40 units off
    the line, which is fine only when facing it squarely.
- **The first line hit ends the use.**
  - Vanilla: any line *with a special* stops the traverse (p_map.c:1135-1161), so a W1 trigger line in front of a
    switch blocks the use.
  - ViZDoom/ZDoom: only use-activated lines stop it (gzdoom p_map.cpp:5829: `special==0 || !(activation & SPAC_Use...)`
    passes through).
  - Either way, a closed or zero-opening line in front ends it.
- **Back side.**
  - Vanilla `P_UseSpecialLine` returns false on side 1 except 124 (p_switch.c:263-280).
  - ZDoom treats a back-side hit as an ordinary line and *continues the trace* (p_map.cpp:5878-5885). A switch used from
    behind does nothing, but a switch further on can still fire.
  - Manual doors act on the line's **back** sector (`EV_VerticalDoor`), so a door can only be opened from the side it
    faces.
- **Teleports are front-only**, although they are walk lines (`EV_Teleport`: `if (side == 1) return 0;`,
  p_telept.c:62-65). Our `portal_edges` are directed, entry = front, which is correct.
- **Walk exits (52) and other walk specials fire from either side**, but only if the move that crosses them is legal.
  A crossing blocked by an `ML_BLOCKING` line never fires (crustygen check/flood.rs doc, citing p_map.c:214-217).
- **Teleport destination (ZDoom).** `SelectTeleDest` (gzdoom p_teleport.cpp:282-340) picks at random only when a TID is
  given. Doom-format lines (tid 0) keep "first destination found in a tagged sector", so `destinations[0]` is right
  for Doom-format maps.
- **Monster-only lines.** `ML_BLOCKMONSTERS` blocks monsters only (`!tmthing->player`, chocolate p_map.c:236). 125 and
  126 teleport monsters only.
  - **Check:** what ViZDoom's `Line.is_blocking` means. If it includes BLOCKMONSTERS, our `walls` are too strict.
  - One test line in a known map settles it.

### A4. Movement-geometry fidelity: false walls and false step-ups
- **The body is a square, not a circle.**
  - Doom's collision box is axis-aligned ±radius (16 for the player). `P_CheckPosition` takes floorz as the **max**
    floor and ceilingz as the **min** ceiling over every line the box touches.
  - We use `area.buffer(16)`, which is round. It misses 16→22.6-unit diagonal corners, so floors at stair corners and
    ledge corners come out too low.
  - Fix: use `buffer(16, cap_style=3, join_style=2)` (mitre/square). The same applies to `walk.buffer(-18)` and the
    walls `buffer(18)`.
- **Substep the step check.**
  - The engine checks each move in substeps no longer than MAXMOVE/2 (vanilla, 30/2) or radius (ZDoom).
  - Zandronum's `BOTPATH_TryWalk` (botpath.cpp:416) splits a candidate edge into ≤MAXMOVE pieces and applies the step
    rule to each.
  - Our edge compares the floors of two cell centres 16 units apart, or 22.6 diagonally. On stairs with steps less
    than 16 deep (e.g. 16-high, 8-deep), adjacent cells can differ by 32 although every real substep is 16. That gives
    a **false "step up 32"**.
  - Fix: when `diff > 24`, re-test the edge at 4–8-unit samples, with floor = max over the square box at each sample.
    Accept it if no consecutive pair rises by more than 24.
- **Solid things are missing.**
  - Barrels (r10, h42), columns, lamps and trees (r16–32) block the player (vanilla things are infinitely tall; in ZDoom
    their real heights are all >24).
  - ZDoom players can *stand on* an actor whose top is ≤24 above them (gzdoom p_map.cpp:2385-2398, MF2_PASSMOBJ). That
    is rare in Doom maps.
  - Add static MF_SOLID things from THINGS as blocked cells: a square of (thing radius + 16). This should remove a class
    of "rejected edge" stalls.
- **Headroom.** Ours matches P_TryMove: `tmceilingz - tmfloorz < height`, plus `tmceilingz < thing->z + height` for a
  drop into a low pocket (gzdoom p_map.cpp:2414-2426). Keep it.
- **Drops.** Players have MF_DROPOFF and never fail a dropoff. Monsters fail when `floorz - dropoffz > 24` (gzdoom
  p_map.cpp:2512-2526). So **a drop of more than 24 units is an escape route monsters cannot follow**, apart from
  floaters (caco, pain elemental, lost soul).
- **Jump and crouch.**
  - The ZDoom bots *jump* ledges of 24–32 units. Cajun `CleanAhead` sets BT_JUMP when the step is between
    MaxStepHeight and `MAXMOVEHEIGHT` 32 (b_move.cpp:293-316, b_bot.h:43). Zandronum uses `CalcJumpHeight()`
    (≈36 for JumpZ 8) and 36 for climbing onto actors (botpath.cpp:470-490).
  - ViZDoom exposes JUMP and CROUCH. Unless MAPINFO or dmflags forbid it, ZDoom allows jumping.
  - Freedoom maps never need either, but if JUMP works in our build, 25–32-unit steps become usable "jump edges" for
    the executor and an explanation for the solver.
  - **Test once** in ViZDoom: does JUMP raise z on MAP01?

### A5. Damaging floors
We have no cost for damaging sectors. In vanilla (`P_PlayerInSpecialSector`, p_spec.c:1047), damage applies only while
`z == floor`, every 32 tics (0.91 s):

| Special | Damage per hit | Notes |
|---|---|---|
| 7 | 5 | |
| 5 | 10 | |
| 16, 4 | 20 | even the radiation suit fails 5/256 of the time |
| 11 | 20 | exits when health ≤10 |

Zandronum A* adds +16…+64 per cell for these (astar.cpp:1003-1035). Cajun refuses them outright (IsDangerous, used in
P_TryMove for bots, gzdoom p_map.cpp:2548). Add a cost per cell, and a fact for the model ("route crosses N cells of
nukage: about 5 damage per second").

---

## B. Solver speed (MAP12 took 2.5 h)

What we do now:
- `Solver.navigator(heights)` builds a **new Navigator per distinct height tuple**: shapely polygonize and union per
  sector, buffers, meshgrid, and point-in-polygon tests.
- Each lift adds another build at its low end, and `reach()` floods from scratch every iteration.

Speed-ups, in order of payoff:

1. **One geometry build, then incremental floors.**
   - Free cells, `body_sectors` and `sector_cells` do not depend on heights, except the `height>=56 or door` filter in
     `configure` and `closed_unmapped`.
   - Include every sector's area once, and treat "ceiling − floor < 56" through the existing `headroom()` and `locked`
     tests.
   - After that, a mechanism only changes `sector_heights` and `sector_ceilings`, which is exactly what the existing
     `update_floors()` already refreshes for the affected cells.
   - Expect seconds, not hours.
2. **Monotone frontier BFS.**
   - In optimistic mode, reach only grows. Keep the reached set and, after a move of sectors S, re-seed the BFS from
     reached cells adjacent to `sector_cells[S]`.
   - That needs no whole-map flood per step.
3. **Lifts as edges, not height variants** (crustygen reach.rs:66-85 `EdgeKind::Lift`, `lift_edges` at :910).
   - A lift is an undirected edge between each caller or neighbour sector at the lift's low floor and the platform at
     rest. Step and window rules are skipped on it because the platform moves to the caller.
   - That replaces the second navigator built at the low end.
4. **Region graph instead of cells for the solver.**
   - Contract cells into connected components of identical `(body-sector set)` (or of equal floor and headroom class).
   - Precompute, once, the critical heights at which each component-to-component edge opens.
   - The solver then BFSes over a few hundred nodes, and a mechanism flips only the edges that touch its sectors.
   - Keep the cell grid for the executor.
5. **State search for softlocks** (crustygen reach.rs:1-40, :376-470 `check`).
   - Use BFS over `(region, keys ∪ fired-actions bitmask)`, plus a backward pass that finds states which cannot reach
     the exit ("holding the blue key in a pit": a softlock).
   - This is small: regions × 2^(keys+one-shot actions). It fixes the documented optimism ("reach is a union, a
     one-way drop is not noticed").
   - Only one-shot (S1/W1) actions fit a monotone bit. Repeatable ones need a toggle bit or should be treated as
     always available.

---

## C. Combat facts to expose to the model (code lists facts, the model decides)

All from vanilla p_enemy.c, p_mobj.c and info.c. ZDoom keeps the same DECORATE values.

1. **Hitscan accuracy falls about as 1/distance.**
   - Monster bullets get `angle += P_SubRandom()<<20` (A_PosAttack p_enemy.c:808): a triangular spread of ±22.4°.
   - Damage per pellet is `(P_Random()%5+1)*3`, 3–15 with a mean of 9.
   - Against the player's 16-unit half-width:

     | Distance | Hit chance per bullet | Expected damage, zombie shot | Expected damage, sergeant (3 pellets) |
     |---|---|---|---|
     | 64 | 0.86 | 7.7 | 23 |
     | 128 | 0.54 | 4.8 | 14.5 |
     | 256 | 0.29 | 2.6 | 7.9 |
     | 512 | 0.15 | 1.4 | 4.2 |
     | 1024 | 0.08 | 0.7 | 2.2 |

   - Hitscan cannot be dodged once fired. Only distance and line of sight matter. This is the fact behind the 20–35 %
     hitscan deaths.
2. **How often a monster attacks** (`P_CheckMissileRange`, p_enemy.c:197-252). An attack is tried only when
   `movecount == 0` after a chase step (A_Chase :731), never twice in a row (MF_JUSTATTACKED), and
   `P(attack) = 1 − min(d',200)/256`, where:
   - `d' = dist − 64`, minus another 128 when the monster has no melee attack (zombies, sergeants, chaingunners: they
     fire more);
   - revenant: halved, and no missile within 196;
   - cyberdemon, mastermind and lost soul: halved; cyberdemon capped at 160;
   - arch-vile: none beyond 896.

   Floor: at long range every monster still attacks with about 22 % chance per check. **After pain (MF_JUSTHIT,
   p_inter.c:907-910) the next check always attacks.**
3. **Projectiles aim at your current position and never lead** (P_SpawnMissile: angle to `dest->x,y` at fire time;
   momz set for the current z).
   - So **steady lateral motion dodges every non-homing projectile.** It must cover the player radius plus the
     projectile radius: 16 + 6 = 22 (imp, caco, baron, mancubus) or 16 + 11/13 (rocket/tracer, plasma) before impact.
   - Speeds in units per tic (info.c):

     | Projectile | Speed |
     |---|---|
     | imp, caco, revenant tracer | 10 |
     | baron, knight | 15 |
     | mancubus, rocket | 20 |
     | arachnotron plasma | 25 |

   - Player strafe-run terminal speed is 13.3 u/tic, forward run 16.7 (sidemove 0x28, forwardmove 0x32, friction
     0xE800).
   - From rest it takes about 6 tics to move 22 units sideways; already strafing, about 2.
   - Facts to expose per incoming projectile: `tics_to_impact = dist/speed`, `lateral_clearance_needed`, and whether
     the current lateral velocity already clears it.
   - Exceptions:
     - mancubus fires pairs offset by ±11.25° (FATSPREAD, p_enemy.c:1352), which covers a one-way strafe;
     - revenant tracers re-aim every 4 tics by up to 16.9° (A_Tracer :1021-1060, TRACEANGLE 0xc000000), so break line of
       sight instead;
     - the arch-vile's attack needs line of sight when it lands (break LOS to cancel);
     - rockets and the cyberdemon splash 128 units.
4. **Pain chance /256** (info.c), useful for target choice and for "can I stun-lock it":

   | Monster | Pain chance /256 |
   |---|---|
   | lost soul | 256 |
   | zombie, imp | 200 |
   | demon, spectre | 180 |
   | sergeant, chaingunner, SS | 170 |
   | caco, arachnotron, pain elemental | 128 |
   | revenant | 100 |
   | mancubus | 80 |
   | baron, knight | 50 |
   | mastermind | 40 |
   | cyberdemon | 20 |
   | arch-vile | 10 |

   Health: zombie 20, sergeant 30, chaingunner 70, imp 60, demon 150, caco 400, knight 500, baron 1000, revenant 300,
   mancubus 600, arach 500, vile 700.
5. **Unseen attackers.**
   - Every shot calls P_NoiseAlert, which floods sound through every sector connected by a non-zero opening and stops
     at the **second** ML_SOUNDBLOCK line (P_RecursiveSound, p_enemy.c:99-140). Monsters in those sectors wake.
   - Ambush (deaf) monsters wake only on sight. Monster sight is a 180° front cone (P_LookForPlayers :494-535).
   - Fact to expose: "firing here wakes ~N monsters (WAD things in sound-connected sectors, minus deaf ones), M of them
     hitscanners."
   - This targets the 13–31 % "unseen" deaths.
6. **Escape geometry.** Walking monsters cannot drop more than 24 units or cross BLOCKMONSTERS lines. Fact: "a
   monster-proof drop is reachable within N units."
7. **Cajun bot rules** (useful as candidate options, not hard-coded choices):
   - Missile avoidance (b_think.cpp:134-149): when a visible missile is within ~687 units (AVOID_DIST), backpedal plus
     strafe, and flip the strafe side when position changed by less than ~0.76 units since the last tic.
   - Strafe in combat and flip when stuck or at random (1/30 per tic).
   - Back off when inside the weapon's MoveCombatDist.
   - Do not drop off ledges while fighting with melee weapons (b_think.cpp:188, 339).
   - Rocket self-safety: `FireRox`/`FakeFire` (b_func.cpp:506-560) simulate the rocket 140 units (SAFE_SELF_MISDIST),
     lead the target by `vel × (dist/speed + 2)`, and do not fire if a wall is closer.
   - Projectile lead for the player's own projectiles: `enemy.pos + vel × 2·dist/speed` (b_func.cpp Dofire).
   - Item rule (b_think.cpp:150-180): in combat, divert to an item only if it is health while below the threshold, a
     power-up, or very close (< ~134 units), and it is reachable.
   - These are option lists the model could choose among: "dodge-left, dodge-right, backpedal, retreat to cover."

---

## D. Navigation ideas from the bots

- **Experience graph (ZetaBot).**
  - ZetaBot "plops" a node when the current node is out of sight or more than 300 units away (CheckPlopGround,
    ZScript.zs:2797). It drops USE nodes where it pressed use (:575) and SHOOT nodes where it fought (:2914).
  - It detects teleports as a jump of more than 256 units after crossing a special line, and adds a directed edge
    (PlopTeleportNodes :2956, :2990-3016).
  - Edge cost is `dz²/50` plus avoid and "candy" bonuses (Pathing.zs:782-794). Edges need line of sight, at most 24
    units of rise per 16 units, and redundant same-direction links are culled (PostCanConnect :1155-1187).
  - For us: **log every observed successful cell transition, teleport and use outcome per map across episodes.**
    Observed transitions override the geometry rules (this explains rises the rules miss), and repeated failures
    persist beyond the 60 s expiry.
- **Zandronum A***: a 64-unit grid, edges tested by TryWalk in substeps no longer than MAXMOVE, a 1.5× penalty for
  changing direction (smoother, more executable paths), costs for damaging sectors, and flags STAIRS, JUMPABLELEDGE,
  DROPOFF, DOOR and TELEPORT on each edge (botpath.h:60-66).
  - For us: tag each path edge with its kind so the executor knows when to press use or expect a drop, and add a
    turn penalty.
- **Cajun Reachable** (b_func.cpp:61-125): a straight-line traverse over lines that checks each crossed sector for
  floor ≤ last + 32, room ≥ height, and no damage. It is cheap enough to use every tic as a "direct line to target is
  walkable" fact.
- **Cajun stuck handling**: the Doom monster-style NewChaseDir (b_move.cpp:161-260). Try the direct diagonal, then the
  two axis directions, then the old direction, then a random sweep, then turning around. When blocked, use the
  `spechit` lines if they are use-activatable (Move, b_move.cpp:92-140). It is a good local fallback when A* says
  "path" but the executor is blocked.

---

## E. Top 6 ideas, ranked by expected impact

1. **Model stairs, donut, G1 shoot lines, raise-to-texture, special 40, and the boss-death tags 666/667 in the specials
   and solver** (A1, A2). This directly targets the stuck maps ("unexplained" step-ups, boss map). It is small code
   with high payoff.
2. **Restructure the solver: one geometry build plus `update_floors`, frontier BFS, lifts as edges** (B1–B3). This
   should take MAP12 from 2.5 h to under a minute. Then add the region graph and the crustygen state BFS for softlock
   and one-way-drop detection (B4–B5).
3. **Hitscan facts from distance and line of sight**: hit chance per bullet, expected damage, attack chance per check,
   and pain-retaliation (C1–C2). This targets the 20–35 % hitscan deaths. "Close the distance or break LOS" becomes
   something the model can reason about.
4. **Projectile facts**: tics to impact, clearance needed, "current strafe already clears it", the no-lead fact, and
   tracer/mancubus/vile exceptions (C3, C7). This targets the ~30 % projectile deaths.
5. **Geometry fidelity**: square body, substepped step test, solid things as obstacles, damaging-floor cost, and
   checking `is_blocking` versus BLOCKMONSTERS (A4, A5). This removes false walls and false step-ups and the executor
   stalls at barrels and columns.
6. **Wake and unseen-threat facts from sound propagation** (C5), plus a per-map experience graph (D) that records
   observed traversals and failures across episodes.

Lower priority: the use-ray exactness checks (A3). The executor already handles the front side, but facing
squarely within 64 units and the "first use line eats it" rule are worth one test each. Also consider JUMP edges if
ViZDoom allows jumping (A4).
