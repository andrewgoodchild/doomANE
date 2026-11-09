# seenmap lab notebook

## Milestone 1: what the current agent's objectives look like from a seen map (pre-registered 2026-11-08)

Question: when the current agent (WAD planner, engine perception) chooses an objective, how often had the
objective already been in view, and how often was it reachable through free space the player had already seen?
The answer decides what the seen-map planner must offer: routes over the seen map alone, or exploration toward
unseen space as a first-class objective.

Offline, no games. Nothing here changes the agent.

### Data

Recorded development-map games of the final engine-perception stack (`planner_ext_safe`, re-plan on gains where
noted), two per map, one finished and one not:

| Map | Finished | Not finished |
|---|---|---|
| MAP01 | `pix1_engine` seed 54 (177 s) | `pix1_engine` seed 3 (180 s timeout) |
| MAP02 | `pix1_engine` seed 54 (1,444 s) | `pix1_engine` seed 1 (1,800 s timeout) |
| MAP03 | `pix1_engine` seed 1 (538 s) | `pix1_engine` seed 54 (1,200 s timeout) |

If time allows, one more per map: MAP01 seed 1, MAP02 seed 2, MAP03 seed 3. Each run has `telemetry.jsonl`
(position, height, facing, items, switches and the exit per tic) and a planner log (`<run>-planner.jsonl`: every
objective choice with its tick, options and the reason it was asked).

### Replay

The path is replayed in ViZDoom every 6th tic (`doom/seenmap/replay.py`): warp to the recorded position, turn to
the recorded facing, read the depth buffer, the sprite mask (labels buffer, a stand-in for the detector) and the
exact pose. Monsters and doors do not follow the recorded game: doors the player walked through are closed in the
replay, so frames rendered inside a closed door are skipped, and the player's own walked cells count as free
(where I have stood is free). The map persists across deaths within a run, as the agent's memory does.

### Map

`doom/seenmap/mapper.py`, fixed before running: 16-unit cells (0.5 m); camera fx = cx = 160, fy = 132, cy = 100
at half resolution; 7.6 map units per depth step for walls and ceilings; floor placed by row geometry at the
player's floor height with the floor-depth lookup from the splat POC; depth cap 250; sprite pixels skipped.
A wall point counts for a cell when its height is 16 to 72 units above the player's floor (a step or a ledge is not
a wall, a torso-height surface is); a floor point counts as free when within 24 units of the player's floor.
Cell state: wall if wall votes >= 3 and >= half the free votes; else free if free votes >= 2 or the cell was walked;
else unknown. Every 2nd pixel is back-projected.

### Definitions, per objective choice (explore choices are counted separately)

- **Target point.** Switches: the planner's approach point from telemetry, the board for the seen test. Items:
  the item's position from telemetry. Keys: the first marker of that colour in the WAD (what the planner itself
  uses; read here only to score). Exit: the exit's approach point, its centre for the seen test.
- **Seen before chosen (view).** In some replayed frame before the choice tick, the target lay within 45 degrees
  of the facing and no nearer wall hid it: its distance was at most the depth along its column (horizon rows) plus
  32 units, and under the 250-step cap.
- **Seen before chosen (map).** The target's cell, or a neighbour, was free or wall in the seen map at the choice
  tick.
- **Reachable through seen free space (strict).** A* on free cells only from the player's cell to any free cell
  within 48 units of the target; unknown and wall block. This is the primary metric.
- **Variants, reported for the reading.** Unknown passable: unknown cells cost double but do not block. Doors
  passable: cells in the WAD's door sectors are passable even when seen as a wall (a WAD diagnostic, to size the
  door problem; the agent would get this from seeing doors open).
- **Path ratio.** Seen-map path length over the straight-line distance; for objectives the run then completed
  ("objective done" at the next choice), the path the player actually walked, over the seen-map path.
- **Coverage.** Free and known cells every 10 s, as a share of the WAD's walkable cells (ceiling at least 56 above
  the floor), for context only.
- **Cost.** Milliseconds per replayed frame for mapping, and per A* call.

### Pass line (fixed before running)

Primary: the share of chosen objectives (explore excluded) that were reachable through seen free space (strict)
at the time they were chosen, over all six runs.

- Under 70 %: the seen-map planner must offer exploration toward unseen space (frontiers) as first-class
  objectives, and this milestone says so.
- 70 % or more: routing on the seen map alone is a viable first agent.

Secondary readings, also fixed: if the unknown-passable variant is at least 20 points above strict, the first
agent should plan optimistically through unseen space rather than only to frontiers. If the doors-passable
variant is at least 10 points above strict, the planner must offer seen doors as objectives. If fewer than half of
the objectives had been seen before they were chosen, the current agent's objectives come mostly from the WAD,
and the seen-map agent cannot be a drop-in: its objective list will be shorter and exploration will carry more.

Whatever the numbers, they are recorded here with the reading.

### Amendments after the smoke test, before the full evaluation (2026-11-08)

The first smoke run of `evaluate.py` on the MAP01 seed-54 game (one of the six registered runs) showed no strict path to
any of its seven objectives, including the exit at the end of a finished game, with the map's free space visibly made
of isolated dots. That is a mapping defect, not a result, and it was fixed before any other run was evaluated. The
fixes were chosen against the WAD on that run (wall cells within 16 units of a WAD wall line; free cells inside the map
and at least 8 units from a wall; walked cells called wall), never against the milestone metric:

1. **Floor fill.** One image row covers a long stretch of ground at range (a row near the horizon spans hundreds of
   units), so floor samples alone leave gaps of several cells. The floor between two floor samples in adjacent rows of a
   column is now filled every 8 units, and from the farthest floor sample to the wall at the horizon when that wall is
   within one and a half row gaps. `mapper.floor_fill`.
2. **Vote range.** Wall points farther than 400 units from the camera and floor points farther than 600 no longer vote
   (the splat POC found far walls 27 to 67 % within 16 units). On the smoke run this took wall cells within 16 units
   of a WAD wall from 82 % to 95 %.
3. **Vote rule.** A cell is wall when wall votes >= 3 and at least equal to the free votes (the registered rule said half
   the free votes). With half, far scatter won in cells the player had walked; with equal votes the WAD scores were
   the same for walls and slightly better for free cells.
4. **Step height.** Wall points count from 24 units above the player's floor (Doom's step height), not 16: a riser up to
   24 is walked over.
5. **Walked segments.** The player's path is rasterised between consecutive tics into 4-connected cells, not sampled
   per tic: at run speed consecutive positions skip cells diagonally, and A* without corner cutting could not follow
   the path through a doorway (closed in the replay, so seen as a wall) the player had in fact walked through.
6. **Planner records after the level ended** (the first choice on the next map) are dropped, and the exit objective of
   a finished game counts as done.

Still true and recorded: 13 % of the replayed frames are skipped as rendered inside closed doors or lifts or facing a
wall closer than 23 units, and about 16 % of walked cells carry more wall than free votes; they lie a median of 8 units
from a WAD wall line, so they are cells that straddle a wall the player hugged, and the walked override handles them.

After the fixes the smoke run gave strict paths to the three objectives that had been in view and to none of the four
that had not. That number is from the run the fixes were chosen on and is reported again below with the others.

### Second round of amendments, after the first full evaluation (2026-11-08)

The first full evaluation of all nine runs was read for its pictures and its unresolved objectives before its numbers
are taken as the result, and two defects showed:

7. **Phantom floor beyond the walls.** The top-down maps of MAP02 and MAP03 showed free cells radiating outside the
   level, and MAP02's free cells exceeded the WAD's walkable cells. The cause is the floor rule in the rows just under
   the horizon (rows 101 to 114, no exact lookup): its depth band there also fits walls 200 to 400 units away, which
   were then placed as floor 400 to 1,500 units out, through the wall. Floor is now recognised only by the exact
   per-row lookup (rows 115 on, about 350 units and nearer), the near ramp for rows 143 on, and the zero band; far
   floor stays unknown until approached. Chosen against the WAD (free cells inside walkable sectors), not the metric.
8. **Weapon markers.** 46 objectives, every `item_weapon_<n>` choice, were unresolved: these are the planner's WAD
   weapon markers (`doomlib/mission.py`), whose id is the index into the WAD's THINGS, not items the player had seen.
   They are now resolved from THINGS and scored as their own kind, `weapon_marker`, since by construction the agent
   had them from the WAD.
9. **A validity check on the metric.** Beside strict reachability, the evaluation now also reports strict reachability
   with every free cell outside a walkable WAD sector removed (a diagnostic; the agent never gets this). If the two
   agree, the seen map's free space is trustworthy; if the strict share is higher, phantom free cells carried paths.

The first full evaluation's numbers (strict 71 % over 334 objectives with 46 unresolved) are superseded by the run
below and are not used.

### Third amendment, after the second full evaluation (2026-11-08)

The second full evaluation still showed free cells outside the walls on MAP02, and its new validity diagnostic put
strict reachability at 69 % against 37 % with phantom free cells removed, so its numbers are not the result either.
Floor-rule variants were compared on the MAP02 seed-54 game by the share of free cells that lie in a walkable WAD sector
(the walked cells themselves score 99 % on that test, so the test is sound):

| Variant | Free cells | In a walkable sector | Wall cells within 16 units of a WAD wall |
|---|---|---|---|
| A, as evaluated (below-floor rule on, lookup from row 115) | 13,637 | 71 % | 97 % |
| B, below-floor rule off | 8,720 | 82 % | 96 % |
| C, B with the lookup from row 120 | 8,596 | 83 % | 96 % |
| D, B with the lookup from row 125 | 8,409 | 84 % | 96 % |

10. **Below-floor rule off.** The splat POC read any pixel whose depth and row put it below the player's floor as the
    player's own floor seen with the planes' broken depth. On a flat MAP01 that is right; on MAP02 the player often
    looks down into lower areas, and the rule put phantom floor at the player's height over them. Such pixels now
    count as nothing. The lookup row stays at 115: rows 120 and 125 buy one to two points of validity for a third less
    reach. About a sixth of free cells still fall outside walkable sectors, so the strict share is reported together
    with the WAD-valid share below, as an upper and a lower bound.

### Milestone 1: results (2026-11-08)

Nine runs, three per map (the six registered plus the three optional ones), 334 objective choices scored and 12
`explore` choices counted separately. All numbers are from the mapper as amended above (`results/seenmap/milestone1.json`,
per-run detail and pictures beside it). The WAD-valid column removes every free cell outside a walkable WAD sector
(door sectors counted walkable) before planning; it agrees with the strict column on 221 of 222 reachable objectives,
so the phantom free cells that remain (about a sixth of free cells, lying outside the level) carry no paths.

| Scope | Objectives | Seen before chosen (view) | Known in the map | Reachable, strict | Strict, WAD-valid | Unknown passable | Doors passable | Completed by the run | Path over straight line (median) | Walked over path, completed (median, n) |
|---|---|---|---|---|---|---|---|---|---|---|
| MAP01, 3 runs | 24 | 46 % | 46 % | **46 %** | 42 % | 100 % | 46 % | 42 % | 2.9 | 0.6 (4) |
| MAP02, 3 runs | 156 | 81 % | 78 % | **73 %** | 73 % | 100 % | 73 % | 41 % | 1.6 | 3.1 (43) |
| MAP03, 3 runs | 154 | 67 % | 64 % | **63 %** | 63 % | 100 % | 63 % | 62 % | 1.1 | 3.0 (68) |
| All | 334 | 72 % | 69 % | **66 %** | 66 % | 100 % | 66 % | 51 % | 1.4 | 2.9 (115) |

Per run: MAP01 43, 50, 44 %; MAP02 70, 76, 73 %; MAP03 44, 77, 46 % (the MAP03 timeout game, 87 objectives, is the
high one). The first objective of every run, chosen at tick 0 with nothing seen, was reachable in 0 of 9.

| Kind | n | Seen (view) | Known in the map | Reachable, strict | Completed |
|---|---|---|---|---|---|
| switch | 220 | 68 % | 67 % | 66 % | 49 % |
| item (seen by the agent) | 30 | 90 % | 87 % | 80 % | 53 % |
| weapon marker (from the WAD) | 46 | 65 % | 59 % | 59 % | 33 % |
| key (from the WAD) | 35 | 89 % | 74 % | 69 % | 80 % |
| exit | 3 | 67 % | 100 % | 33 % | 100 % |

Reachability follows sight almost exactly: of the 222 reachable objectives 218 had been in view, and of the 112
unreachable ones 90 had not. 22 objectives had been seen and were still unreachable through seen free space; 4 were
reachable without having been in view (the player had walked past them).

Cost: mapping 2.2 ms median per replayed frame (p90 at most 3.0 ms) at one frame per 6 tics, pixel stride 2; A*
4.4 ms median per call, 10 ms on the long MAP02 games. Frames skipped as rendered inside closed doors or lifts or
facing a wall under 23 units: 10 to 18 %. Coverage at the end of a game, free cells over the WAD's walkable cells:
MAP01 0.13 to 0.17, MAP02 0.67 to 0.75, MAP03 0.25 to 0.31; the mapper is conservative, since floor is only accepted
within about 350 units and where the depth buffer's floor value matches the lookup exactly, and MAP01's rooms are
seen mostly from their corridors.

### Reading against the pre-registered pass line

**Primary: 66 % of chosen objectives were reachable through seen free space, under the 70 % line.** The seen-map
planner must offer exploration toward unseen space as first-class objectives. The by-map numbers say where: on the
small MAP01 the agent's choices run ahead of what it has seen (46 %), on the larger maps the agent has usually seen
the way (63 to 73 %), and in every run the first choice has nothing to stand on.

Secondary readings, as fixed: unknown-passable is 34 points above strict, so the first agent should plan
optimistically through unseen space to a target it has seen or been told of, not only walk to a frontier (the 100 %
is by construction, unknown never blocks; the informative part is that no seen target was blocked by a wall). Doors
passable adds nothing: no seen objective was cut off by a door the player had not opened, with the caveat that the
replay's doors were closed and were crossed only through the walked cells. Seen before chosen is 72 %, above half, so
a seen-map agent keeps about three quarters of the current agent's objective choices; the other quarter the current
agent takes from the WAD unseen, mostly switches (32 % of 220 unseen) and WAD weapon markers (35 % of 46).

Two things this milestone does not show. It is offline: whether an agent that is offered only seen objectives and
frontiers makes choices that finish levels is milestone 2's question. And the "seen" test is geometric, within 45
degrees and not hidden by a nearer wall; whether a switch or a door is recognised in that view is a perception
question this milestone assumes away.

### Milestone 2 proposal (not started)

What the seen-map planner must offer, from the numbers:

1. **Seen objectives.** Switches, doors, keys, weapons and the exit that have been in view, with the seen map's route
   to them (strict) or an optimistic route through unseen space when none exists. Items and keys come from the
   labels buffer as the detector's stand-in, as in level 1. Switches, doors and the exit are wall textures, not things,
   so they need a recogniser the project does not have: a small classifier on wall cells of the view, trained on
   development maps with labels projected from the WAD's linedef specials. Until it exists, milestone 2a uses an
   explicit stand-in: the WAD's switches and exit are offered only once their line has been in view by the geometric
   test above, so the planner's design is tested apart from the recogniser.
2. **Frontiers as objectives.** "Explore toward unseen space" with a direction and the frontier's size, offered on
   every question and the only option when nothing seen is reachable (every first choice). The model chooses, as
   it does today between `explore` and the rest.
3. **Doors as what opened.** A wall cell that later reads free is a door; the mapper needs an update rule that lets
   later evidence overturn earlier votes (today votes only accumulate), and the planner can then offer "the door at
   X" once seen.
4. **Mapper changes before the games:** free space from wall bases (the floor between the player and a wall whose
   base row sits on the player's floor plane is free), which does not depend on the depth buffer's broken floor
   values and would lift MAP01's coverage from a sixth of the walkable area; and the vote rule replaced by a
   latest-wins or decaying update for doors and lifts.

What still comes from the engine at this level, stated in the README: pose, height and facing; the depth buffer;
the counters; the labels buffer as the detector's stand-in for things; and in milestone 2a the WAD's switch and
exit positions gated by sight.

**First in-game experiment (milestone 2a), to be pre-registered with its own entry before it runs:** `planner_ext_safe`
with the seen-map planner in place of the WAD planner, MAP01 at skill 1 (survival removed, so the route is what is
tested), eight seeds, 300 s. Pass line to fix then: at least 6 of 8 finished (the WAD planner finishes MAP01 at skill
1 in every game), with time to exit reported against the WAD planner's. MAP02 and MAP03 at skill 1 follow only if
MAP01 passes.

## 2026-11-09: depth calibration corrected

The splat POC's fit against the development map's one-sided walls (`doomsplat scripts/planes2_depthcal.py`) shows the depth
buffer is affine, not proportional: forward distance = 7.1386 x value + 14.297 map units, median error 1.7 units on the
recorded 320x200 walks, against 10.2 units (bias +10) for the 7.6 x value used so far. `mapper.py` now uses it
(`CALIB['depth_offset']`, `depth_units()`), and `evaluate.py`'s view test with it.

Milestone 1 rerun on the same nine runs (`results/seenmap/affine_depth/`, old results kept in `results/seenmap/`):
reachable (strict) 66.5 % both, done 50.9 % both, seen in map 68.9 → 69.2 %, seen in view 71.9 → 70.7 % (the view test
now reads near walls about 11 units farther, so a few objectives just behind a wall stop counting as in view). The
16-unit grid absorbed the old bias; milestone 1's conclusions are unchanged.

## 2026-11-09: vertical focal length corrected

fy is 195 at 320x200, not 132 (fitted on wall-base rows against the development map's floor heights on stationary
frames, median residual 0.65 units; see doomsplat README.md). Every floor point the mapper placed by row geometry was at
68 % of its true distance. Rerun (`results/seenmap/calib_fy/`): seen in map 69.2 → 70.1 %, reachable (strict) 66.5 %
and done 50.9 % unchanged. Not yet handled: view bob moves the eye while walking (fits on moving frames are noisy by
about 11 units); ViZDoom's camera position variables would give the true eye height.
