# seenmap: an agent that plans on a map it has seen

Every doomANE result leans on the WAD: the route rules, the planner's objectives and the navigator's paths know
the whole level before the first tic. This project removes that. The agent builds its own map in play from what
the player has seen, the view, the depth buffer and the player's own position and facing, and the planner's
objectives and routes come from that map. The engine still renders; the agent never reads the level's geometry.

## Where it comes from

- The pixel-perception phase (lab notebook, pix1 onward) put enemies and items on the screen reader; the map
  stayed in the WAD. Its backlog called this step "level 2".
- The splat proof of concept (2026-11-08, now in the doomsplat repo; see `poc/splat/README.md`) showed that depth plus exact pose along a real MAP01 path
  rebuilds the visited rooms' walls to a median of 5.8 map units against the WAD, marks 99 % of the walked path
  free, and costs 8.5 ms per frame on the CPU. It also showed that the splat attributes add nothing to the
  geometry: an 8-unit occupancy grid gives the same numbers. So the planning structure here is a grid, and the
  Gaussian attributes are an optional export for viewing the map.
- Two renderer facts found there and used here: the depth buffer's walls and ceilings follow about 7.6 map units (refined on 2026-11-09 to the affine 7.14 x depth + 14.3, see NOTEBOOK.md)
  per step (the agent's 8.1 is a little off), and its floor values follow no scale at all, so floors are placed
  by row geometry at the player's floor height.

## Perception level

Level 2 geometry: no WAD lines, sectors or things reach the agent. What this project still takes from the engine,
stated plainly: the player's own position, height and facing (a player knows where they stand), the depth buffer
(the player's sense of depth), the counters, and, until the detector improves, the labels buffer for monsters and
items as a stand-in for the pixel detector. Doors, switches and keys as objects are what the player has seen of
them: a door is a wall that opened, a switch is a thing on a wall, a key is a pickup.

## Rules carried over

- The model chooses the objective, the target, the weapon and the movement; code builds the map, lists what it
  holds and executes. An API error is never replaced by a heuristic.
- Wall and game time agree within 5 %.
- Development maps only for data, tuning and debugging (`doom/map_split.json`); held-out maps only from a frozen
  snapshot. No hosted model.
- Every experiment is pre-registered in `docs/seenmap/NOTEBOOK.md` with its reading fixed before it runs.

## What is new

No published Doom agent plans on a map built from its own view; the ViZDoom competition agents learned
navigation end to end, and every LLM or decision-model agent read the engine's state or the WAD. The design
question that decides the planner is in milestone 1: when the current agent chooses an objective, how often had it
already seen that objective, and how often was it reachable through the free space it had already seen?

## Layout

| Path | What |
|---|---|
| `doom/seenmap/mapper.py` | Online occupancy map (free, wall, unknown) from depth and pose, per tic |
| `doom/seenmap/planner.py` | A* on the grid, frontier cells |
| `doom/seenmap/replay.py` | Replays recorded runs in ViZDoom and saves what the player saw |
| `doom/seenmap/evaluate.py` | Milestone 1: objectives against the seen map, tables and pictures |
| `docs/seenmap/NOTEBOOK.md` | Dated record, pre-registrations first |
| `results/seenmap/` | Tables and pictures; large files in `ext/poc-data/seenmap/` |
