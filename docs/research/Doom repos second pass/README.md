# Second pass over key Doom repos: ranked ideas (2026-10-24)

Sources: llm_mcp_agents.md, bots_engine_navigation.md, rl_agents_code.md in this folder. Items marked
"verified" were checked against our code or measured; the rest are from code reading only.

1. **Run while moving (verified).** No SPEED button, `cl_run=false`: the bot always walks. Measured in
   ViZDoom: running strafe 1.7x, running backward/forward 2.1x the walking distance. Arnold, sample-factory,
   LevDoom and HASARD all run. Fight-rollout labels so far are at walking speed, so relabel after the change.
   Execution detail only: the model still chooses the movement.
2. **Stairs and other missing movers (verified: `specials.destination` has no stairs branch).** Stair
   builders (7/8/100/127) raise a chain of sectors (next step = back sector of a two-sided line whose front is
   the current step, same floor flat; +8 or +16 per step, chocolate p_floor.c EV_BuildStairs). Also missing:
   gun-triggered lines (24/46/47), donut (9), raise-to-texture (30/96), 40 (also lowers the floor). Boss-death
   tags 666/667 have no linedef. Probably behind the unexplained step-ups on MAP18/24/26.
3. **Solver speed.** Build the geometry once, apply heights with `update_floors`, grow reachability from the
   last frontier, treat lifts as two-way edges (crustygen `lift_edges`); a relaxed search with penalty costs
   for locked or moved sectors finds prerequisites in one pass (doom-mcp nav.rs). MAP12: 2.5 h now.
4. **Who hit me, and from where; a restart briefing.** doom-mcp reports attacker type and bearing per hit,
   including off-screen ones (needs an ACS script to pass the damage source out; ViZDoom does not expose
   it, so check that first). After a death, give the model facts on where and how it died and what blocks
   the route (lucas-moont's notebook took E1M1 from 4/5 to 5/5). Aimed at the 13-31% of damage from
   attackers never seen.
5. **Command-level outcome labels at low HP.** Roll out pickup health / retreat / continue to the key for
   10-20 s from saved states with the frozen model playing on; this targets the red-to-blue-key deaths
   on MAP04. DFP-style multi-horizon measurements (32/64/128 tics) and capped dealt damage.
6. **Vanilla combat facts.** Hitscan hit chance falls steeply with distance (about 0.86 at 64 units, 0.29 at
   256), so only distance or breaking line of sight helps. Projectiles aim at the current position and do
   not lead, so a steady (running) strafe dodges them. Gunfire wakes monsters through connected sectors.
   Expose distance and line-of-sight facts; add diagonal and turn options labelled by rollouts.
7. **Geometry details.** Square 32x32 player box (we use a circle), step test in 15-unit substeps (false
   "step up 32" on shallow stairs), solid things (barrels, columns), extra cost on damaging floors.
   Check whether ViZDoom's `is_blocking` includes monster-only lines, and whether JUMP works.

doomLaya upstream: nothing after v0.3.1. Licences: copy ideas only from GPL (doom-mcp, chocolate-doom),
from no-licence repos (lucas-moont, Arnold) and from Zandronum; DFP, sample-factory, LevDoom, HASARD, COOM,
ZetaBot and crustygen are MIT or permissive.
