# Doom player guides: generic combat and survival knowledge for doomLaya

Date: 2026-10-25. Scope: generic (map-independent) knowledge only; no walkthroughs.

## Sources and how they were used

doomwiki.org, doom.fandom.com and the breezewiki mirror refused automated fetches (HTTP 403 / 402 / Anubis).
I did not try to get around those blocks. The numbers below therefore come mostly from the **primary source**, the
id Software Doom source release (linuxdoom-1.10). It is authoritative for vanilla behaviour. ViZDoom runs a
ZDoom-derived engine, but it keeps vanilla monster and weapon tables for Doom II-compatible maps. The guide-style
advice comes from Freedoom's own manual, community pages and search-engine summaries of doomwiki pages, which are
marked "(via search summary)".

| Tag | URL |
|---|---|
| [info.c] | https://raw.githubusercontent.com/id-Software/DOOM/master/linuxdoom-1.10/info.c |
| [p_enemy.c] | https://raw.githubusercontent.com/id-Software/DOOM/master/linuxdoom-1.10/p_enemy.c |
| [p_inter.c] | https://raw.githubusercontent.com/id-Software/DOOM/master/linuxdoom-1.10/p_inter.c |
| [p_pspr.c] | https://raw.githubusercontent.com/id-Software/DOOM/master/linuxdoom-1.10/p_pspr.c |
| [p_map.c] | https://raw.githubusercontent.com/id-Software/DOOM/master/linuxdoom-1.10/p_map.c |
| [p_local.h] | https://raw.githubusercontent.com/id-Software/DOOM/master/linuxdoom-1.10/p_local.h |
| [p_spec.c] | https://raw.githubusercontent.com/id-Software/DOOM/master/linuxdoom-1.10/p_spec.c |
| [p_switch.c] | https://raw.githubusercontent.com/id-Software/DOOM/master/linuxdoom-1.10/p_switch.c |
| [g_game.c] | https://raw.githubusercontent.com/id-Software/DOOM/master/linuxdoom-1.10/g_game.c |
| [FD manual] | https://raw.githubusercontent.com/freedoom/freedoom/master/manual/freedoom-manual-en.adoc |
| [doom2.net SR] | https://doom2.net/doom2/facility.html (Deathmatch Training Facility: straferunning is 28% faster than running) |
| [DSDA advice] | https://dsdarchive.com/advice ("the less time you spend in any area, the less damage you'll take from hitscans") |
| [DW tips] | https://test.doomworld.com/forum/topic/117223-post-your-tips-for-doom (via search summary: strafe constantly, prioritise hitscanners, pulse the chaingun) |
| [DW sound] | https://doomwiki.org/wiki/Sound_propagation, https://doomwiki.org/wiki/Monster_behavior (via search summary) |
| [DW infight] | https://doomwiki.org/wiki/Monster_infighting, https://zdoom.org/wiki/Infight (via search summary) |
| [DW CG] | https://doomwiki.org/wiki/Chaingun, https://doomwiki.org/wiki/Pain_state, https://www.doomwiki.org/wiki/Heavy_weapon_dude (via search summary) |
| [DW imp] | https://www.doomwiki.org/wiki/Imp (via search summary: fireballs slow, strafe to dodge; hard when close or surrounded) |
| [melee guide] | https://mirrors.lug.mtu.edu/idgames/levels/doom2/Ports/m-o/meleebtl.txt (chainsaw keeps cacos, PEs and arachnotrons from attacking) |

Units: 1 tic = 1/35 s. 1 m in doomLaya = 32 map units (mu), as in `doomlib/combat.py`. One Laya decision = 0.5 s = 17.5 tics.

---

## 1. Monsters (Doom II roster, Freedoom names)

Freedoom names come from [FD manual]. HP, speed, painchance and reactiontime come from [info.c]. Attack formulas come
from [p_enemy.c]. Projectile damage is `(rand%8+1) * damage` [p_map.c]. Pain % = painchance/256.

| Doom name (Freedoom) | HP | Attack | Damage per hit | Pain % | Move speed (mu/step, step tics → m/s) | Notes |
|---|---|---|---|---|---|---|
| Zombieman (Zombie) | 20 | hitscan, 1 bullet | 3-15 | 78% | 8 / 4 → 2.2 | 10-tic wind-up before the shot (S_POSS_ATK1) |
| Shotgun guy (Shotgun zombie) | 30 | hitscan, 3 pellets | 3-15 each (9-45) | 66% | 8 / 3 → 2.9 | 10-tic wind-up |
| Chaingunner (Minigun zombie) | 70 | hitscan, continuous | 3-15 per bullet, 2 bullets per ~9 tics (~70 dps point blank) | 66% | 8 / 3 → 2.9 | Keeps firing while it can see you. Each refire has a 40/256 chance to stop (A_CPosRefire). Stops when sight is lost [p_enemy.c]. "Locks onto you… until you take cover" [FD manual] |
| Imp (Serpentipede) | 60 | fireball 10 mu/tic (10.9 m/s); claw | ball 3-24; claw 3-24 | 78% | 8 / 3 → 2.9 | 16-tic wind-up before the throw |
| Demon (Flesh worm) | 150 | melee bite | 4-40 | 70% | 10 / 2 → 5.5 | Fastest common monster, but still slower than a walking player |
| Spectre (Stealth worm) | 150 | as demon | 4-40 | 70% | 5.5 | Partly invisible |
| Lost soul (Hatchling) | 100 | charge at 20 mu/tic | 3-24 | 100% | 8 | Always flinches when hit |
| Cacodemon (Trilobite) | 400 | ball 10 mu/tic; bite | ball 5-40; bite 10-60 | 50% | 8 | Flies |
| Pain elemental (Matribite) | 400 | spits lost souls | n/a | 50% | 8 | Kill it first or it multiplies |
| Hell knight (Pain bringer) | 500 | ball 15 mu/tic; claw | ball 8-64; claw 10-80 | 20% | 8 | Knights and barons count as one species for projectiles |
| Baron (Pain lord) | 1000 | as knight | as knight | 20% | 8 | |
| Revenant (Octaminator) | 300 | homing missile 10 mu/tic; punch | 10-80; punch 6-60 | 39% | 10 | No missile under 196 mu (6 m) |
| Mancubus (Combat slug) | 600 | 6 fireballs 20 mu/tic, spread | 8-64 each | 31% | 8 | |
| Arachnotron (Technospider) | 500 | plasma 25 mu/tic | 5-40 | 50% | 12 | |
| Arch-vile (Necromancer) | 700 | line-of-sight fire blast | 20 + 70 blast | 4% | 15 | Only attacks within 896 mu (28 m). Revives corpses. Break line of sight [FD manual] |
| SS (absent in Freedoom) | 50 | hitscan | 3-15 | 66% | | |
| Spider Mastermind (Dreadnought), Cyberdemon (Assault tripod) | 3000 / 4000 | chaingun / rockets | | | | Immune to splash [p_map.c] |

### AI facts that matter for tactics (all from [p_enemy.c] unless marked)

- **Vision.** A monster that has not been alerted sees only the half-plane in front of it, unless the player is within melee range (P_LookForPlayers).
  A monster woken by *sound* gets 360-degree vision [DW sound].
  An "ambush" monster hears but waits for line of sight.
- **Sound.** Every player shot calls P_NoiseAlert (P_FireWeapon in [p_pspr.c]), fists included.
  The noise floods through every sector joined by an open two-sided line. A closed door (zero opening) stops it.
  It crosses at most one "sound-blocking" line.
  A sector's sound target is never cleared, so a monster in a sector that has heard you can wake again at once [DW sound].
  Consequence: firing in one room can wake large parts of the map. Closed doors contain the noise.
- **Reaction time.** reactiontime = 8 [info.c]. It drops by one per A_Chase call (one per run frame of 2-4 tics).
  While it is non-zero, the monster makes no ranged attack (P_CheckMissileRange returns false).
  That gives about 0.5-0.9 s after waking before the first shot.
  **Damage sets reactiontime to 0** [p_inter.c], and a monster hit while in pain is flagged MF_JUSTHIT, which makes its next missile check succeed ("fight back").
  So a monster that is hit but not killed shoots back sooner than one that is left alone.
- **How often monsters fire.** The check is `fire if P_Random() >= dist`, where dist = (distance − 64, minus 128 more for monsters without melee) in mu, capped at 200.
  Zombiemen, shotgun guys and chaingunners have no melee, so they will fire every chance within about 6 m.
  At long range about 22% of checks lead to a shot.
  Below Nightmare, after a missile attack the monster moves (MF_JUSTATTACKED) and only re-checks when movecount reaches 0.
- **Melee range.** A monster can bite when centre distance < MELEERANGE − 20 + player radius = 64 − 20 + 16 = **60 mu (1.9 m)** [p_local.h, p_enemy.c].
  A player walking backwards gains on every melee monster (demon 5.5 m/s vs walking 9.1 m/s, running 18.2 m/s; see section 4).
  Monsters also only move in 8 directions.
- **Monster hitscan spread.** The aim angle gets `(P_Random()-P_Random())<<20`, a triangular spread of up to ±22.4 degrees.
  The chance that one bullet hits the 16-mu-radius player, derived from that spread:

  | distance | 2 m | 4 m | 8 m | 16 m | 32 m |
  |---|---|---|---|---|---|
  | P(hit) per bullet | ~0.86 | ~0.53 | ~0.29 | ~0.15 | ~0.08 |

  At 16 m a chaingunner does about 1/6 of its point-blank damage. **Distance and breaking line of sight are the main defences against hitscan.**
  Hitscan cannot be dodged by strafing, since it lands in the same tic [DW tips].
- **Pain.** Each damage event has a painchance/256 chance to put the monster in a pain state (4-6 tics for zombies, imps and demons [info.c]).
  Pain interrupts the current state, including an attack wind-up (zombies, shotgun guys and chaingunners wind up for 10 tics, imps for 16).
  The chaingun fires every 4 tics, so against 66-78% pain monsters it cancels most of their attacks ("stunlock") [DW CG].
  It does little against barons and knights (20%).
- **Infighting** [p_inter.c, p_map.c, DW infight].
  - A monster damaged by another monster switches target to it when its threshold is 0, then chases it exclusively for BASETHRESHOLD = 100 chase calls ("3 seconds" in the source comment).
  - A projectile hitting a monster of the same species (knight and baron count as one) explodes without damage, so imps never infight imps.
  - Hitscan damages anyone, so zombies hurt each other.
  - Nobody retargets onto an arch-vile.
  - Placing a monster between you and a shooter of another species starts a fight [FD manual: "Position yourself so enemies at the back shoot those in front"].
- **Doors monsters can open.**
  - Use (bump): specials 1 (normal manual door) and 32-34 (one-time *keyed* doors; the key is not checked for monsters). Never secret-flagged lines [p_switch.c].
  - Walk-over: 4 (W1 door), 10/88 (lifts), 39/97/125/126 (teleports) [p_spec.c]. Shoot: 46.
  - Monsters **cannot** open keyed repeatable doors (26-28), fast doors (117/118), one-time open doors (31), or anything worked by a switch.
  - A manual door re-closes about 4.3 s after opening (150 tics). Behind a door of type 1, a monster will follow; behind the others it will not.

## 2. Player weapons (Freedoom names in parentheses)

Hitscan damage per bullet or pellet is 5, 10 or 15 [p_pspr.c].
Cycle times come from [info.c] attack states, minus the A_ReFire state, which is skipped while fire is held.

| Weapon | Damage per shot | Cycle | ~DPS (all hits) | Spread | Notes |
|---|---|---|---|---|---|
| Fist | 2-20 (×10 with berserk: 20-200) | 17 tics | 23 (berserk ~225) | small | Reach: trace of 64 mu from player centre to the target's edge, so ~64 + target radius (≈2.6 m to a zombie's centre). doomLaya uses 1.5 m |
| Chainsaw (Ripsaw) | 2-20 | 8 tics (2 hits) | ~96 | small | Reach 65 mu + radius. Pulls you onto the target. Its 70% pain targets cannot attack while sawn [FD manual, melee guide] |
| Pistol (Handgun) | 5-15 | 14 tics | 25 | first shot exact, then ±5.6° | |
| Shotgun (Pump-action) | 7 × 5-15 (avg 70) | 37 tics | 66 | ±5.6° per pellet | One shot kills zombies and shotgun guys up close. Imps often take 1-2 shots |
| Super shotgun (Double-barrelled) | 20 × 5-15 (avg 200) | 57 tics | 123 | ±11.2° horizontal plus vertical jitter | Kills a demon in one shot up close. 2 shells |
| Chaingun (Minigun) | 5-15 | 4 tics per bullet | 87 | the first two shots of a burst are exact, then ±5.6° | Stunlocks high-pain monsters. Pulsing restores accuracy [DW tips, DW CG] |
| Rocket launcher (Missile launcher) | 20-160 direct, plus splash 128 − d | 20 tics | ~150+ | projectile 20 mu/tic | **Splash hurts the shooter**; nothing in P_RadiusAttack excludes the source. d is the Chebyshev distance minus target radius. Cyber and Spider are immune [p_map.c] |
| Plasma (Polaric energy weapon) | 5-40 | 3 tics | ~260 | projectile 25 mu/tic | |
| BFG (SKAG 1337) | ball + 40 rays × 15d8 | | | | 40 cells |

- **Weapon switch costs about 0.9 s with no fire.** The gun lowers 96 mu at 6 mu/tic (16 tics) and raises the same (16 tics).
  Firing is impossible meanwhile, and the switch starts only from the ready state [p_pspr.c].
  This is close to two Laya decisions with no fire.
- **Rocket splash geometry.** The rocket explodes at the target's surface. With the player 4 m (128 mu) from a 20-mu zombie, the blast is about 97 mu from the player centre: 128 − (97 − 16) ≈ **47 self-damage**.
  The no-damage distance is about 175-190 mu (**5.5-6 m**) centre to centre, more for wide monsters.
  The current `ROCKET_SAFE_METERS = 4.0` in `doomlib/combat.py` is too short. It is a factual reach threshold, so fixing it is within the rules.
- **Autoaim** (P_BulletSlope) tries the facing line, then ±5.6°, within 1024 mu (32 m). The vertical window is ±0.625 slope (±32°).
  Hitscan reaches up to 2048 mu (64 m), but beyond 32 m autoaim gives no vertical correction [p_pspr.c, p_map.c].
- **Items** [p_inter.c].
  - Stimpack (Small health refill) +10 and medikit (Large health refill) +25. Both are refused at 100 or more health.
  - Health bonus +1 up to 200. Soulsphere (Ectoplasmic surge) +100 up to 200.
  - **Berserk** (Strength symbiote): health goes to 100, and fists do ×10 damage for the rest of the level.
  - Green armour (100) absorbs 1/3 of damage. Blue (200) absorbs 1/2. Green 100 turns 100 HP into ~150 effective HP.
  - Armour bonus +1 up to 200. It gives green-type absorption if the player has no armour.

## 3. Generic tactics from player guides

1. **Prioritise hitscanners.** You can dodge fireballs but not bullets [DW tips]. Chaingunners die in 2-3 shots [DW CG]. Use hard cover against lock-on enemies [FD manual].
2. **Strafe constantly**, and circle-strafe projectile throwers. "Every hit, even from an imp can be fatal" [DW tips]. Imp fireballs are slow and easy to strafe in open space. Dodging is hard when close, surrounded, or in a teleport trap [DW imp].
3. **Minimise time in the open** [DSDA advice]. Always-run is universal among experienced players. Straferunning (forward plus strafe) is 28% faster still [doom2.net SR]. SR50 (strafe component 50) is a speedrun technique, noted only.
4. **Corners, doors and choke points.** Back into a narrow corner or corridor so that few attackers have line of fire [FD manual: "back into a narrow corner to limit attackers"].
   Fight at a doorway so that monsters funnel in. Step back out of sight to stop a chaingunner's burst (engine: refire stops without sight).
   "Door dancing": open, shoot, step back as the door closes. This works only against monsters that cannot open that door type (section 1).
5. **Infighting.** Put a monster of a different species between you and a shooter. Circle crowds to bunch them [FD manual].
6. **Weapon per target.** Chaingun for high-pain monsters, shotgun or super shotgun for burst damage, chainsaw for demons, spectres and cacos [melee guide, FD manual]. Don't mash fire; hold it [FD manual].
7. **Barrels** hurt monsters. Avoid chain reactions near yourself [FD manual].
8. **Health timing** (community consensus and engine facts).
   - Small heals do nothing at full health (they are refused).
   - Pick up armour and health *before* opening the next door, since fights after a door are the ones that kill.
   - Retreat to collected-but-unused health in a cleared area instead of fighting at low HP.
9. **What kills players most** (guides and engine numbers):
   - close-range hitscanners, especially chaingunners and shotgun guys at < 5 m (expected 25-70 dps);
   - being surrounded in the open after a teleport or closet ambush (walk-over lines open monster closets);
   - melee monsters reaching you while you stand still or walk forward;
   - your own rocket splash;
   - fighting on at low HP instead of disengaging.

## 4. Engine mechanics a bot can use legitimately

- **Player speed** [g_game.c].
  - Inputs: forward 25 walk / 50 run; side 24 walk / 40 run. Thrust = move × 2048 per tic, and friction is 0.90625.
  - Top speed: walking 8.3 mu/tic (**9.1 m/s**), running **16.7 mu/tic (18.2 m/s)**, running strafe 13.3 mu/tic (14.6 m/s), walking strafe 8 mu/tic (8.75 m/s).
  - Forward plus strafe while running: ~21 mu/tic. The axes are clamped separately, which is why straferunning is faster.
  - It takes about 0.66 s to reach 90% of top speed.
  - Walking, the bot is 1.6× a demon's speed. Running, it is 3.3×.
- **Dodge lead time.** The bot must clear ~22 mu (player radius 16 + fireball radius 6).
  From standing, a running strafe covers that in about **6 tics (0.17 s)**; a walking strafe needs about 9 tics (0.26 s).
  An imp fireball covers 10 mu/tic. With 0.5 s decisions (up to 17 tics of latency) plus the lead time, the ball must be seen at least **~0.75 s ≈ 8 m** away to dodge reliably. Closer than ~5 m, dodging by strafing is luck; distance or cover is the answer.
- **Hitscan accuracy versus distance** for monsters (table in section 1) and for the player (first pistol and chaingun shots exact; shotgun ±5.6°).
- **Pain stunlock.** The pain chance is rolled per damage event, so every pellet and bullet counts.
  Seven shotgun pellets on an imp give almost certain pain. Continuous chaingun fire on zombies, imps and demons cancels most of their attacks.
- **Reaction window.** A freshly woken monster cannot shoot for about 0.5-0.9 s unless it is damaged. Kill it in one burst inside that window, or don't touch it.
  Partial damage removes the delay (section 1).
- **Monster melee needs < 1.9 m.** Backpedalling at walking speed alone beats every Doom II melee monster.
- **Line of sight** ends chaingunner and spider refire and arch-vile attacks. Closed doors block sight and sound.
- **Doors monsters can't open** (section 1) make real safe zones. Type-1 doors do not.

---

## 5. Ranked proposals

The rules being checked:
- (a) code states facts;
- (b) code executes the chosen command;
- (c) code excludes a command only after an actual execution failure;
- no priorities, usefulness or exit chosen by code; no map ids or coordinates; labels from game outcomes.

### P1. Outcome-scored counterfactual rollouts for retreat, cover and health

- **Knowledge.** Experts disengage at low HP and take health before fights. The model rarely retreats because nothing in its labels rewards it.
- **Form.** Training labels.
  - At decision points with visible enemies, save the ViZDoom state and branch each command and movement for N = 3-5 s.
  - Score from game outcomes only: −(damage taken) − death penalty + monster HP removed + progress. HP and damage values are engine readings.
  - Label with the best-scoring branch.
  - Add a "damage taken in the next 3 s" term to existing rollout labels.
- **Rules.** Labels come from outcomes, not from hosted LLMs. No code priority.
- **Impact.** Highest. This targets the core failure directly, and the facts below give the model something to condition on. Expect −30-50% deaths if branches cover retreat and health. Costly in compute.

### P2. Run everywhere (executor)

- **Knowledge.** Running doubles speed (9.1 → 18.2 m/s). Less time exposed means less hitscan damage [DSDA, g_game.c]. Every melee monster becomes far slower than the player.
- **Form.** Executor capability. Make `--run-navigation` the default and also run during attack-stationary repositioning. The flag already exists in `doomlib/executor.py`.
  Option: hold forward plus strafe when a route segment allows it (straferun +28%).
- **Rules.** Pure execution of the chosen command (b).
- **Impact.** Medium-high on deaths in transit and chip damage, and it shortens the 20-minute clock. **Risk:** overshooting pickups, ledges and lift edges. Needs an A/B on the held-out maps.

### P3. Monster attack-class and hit-chance facts in enemy options

- **Knowledge.** The hitscan hit chance falls from ~86% at 2 m to ~15% at 16 m. Projectiles can be dodged; melee needs < 1.9 m.
- **Form.** Fact line per enemy option. Example: `Zombie #12, 5.1 m: hitscan, about 45% of its bullets hit at this distance; can see you: yes.`
  Imp: `throws dodgeable fireballs (10.9 m/s); claws within 1.9 m.` Demon: `melee only, moves 5.5 m/s (slower than you walking).`
  Derive the class from the engine's monster type, the same on every map. That is like the existing `hitscan_visible` in `combat_facts.py`.
- **Rules.** Measured or derived facts, not "shoot this first". The authority test is that there is no ranking or "dangerous" adjective; numbers only.
- **Impact.** Medium. It lets the model learn "back off from hitscanners" and "backpedal from demons" from P1 labels.

### P4. "Break line of sight" retreat (executor)

- **Knowledge.** Chaingunner refire stops when sight is lost. Closed doors block sight and sound. Experts duck behind corners [FD manual, p_enemy.c].
- **Form.**
  - New option `retreat_cover`: "Move out of sight of the visible enemies; nearest such spot X m away along known paths."
  - The executor finds the nearest reachable navmesh point with no line of sight to any currently visible enemy (blockmap or sector LOS), runs there, and faces back.
  - If none is found within, say, 10 m, the option is absent, with the stated fact "no cover within 10 m".
  - Today's `retreat` only walks backward (`executor.py:312`).
- **Rules.** Execution of a model-chosen command. The spot is computed geometrically per tick, not from map ids.
- **Authority flag (mild).** The code picks *which* spot. Keep the rule purely geometric (nearest out-of-sight point) so that it is execution, not tactics.
- **Impact.** High against chaingunner and shotgun-guy deaths.

### P5. Health, armour and berserk effect facts in item options

- **Knowledge.** Stim/medikit are refused at 100 HP. Green armour = 1/3 absorption. Berserk heals to 100 and gives ×10 fists [p_inter.c].
- **Form.** Fact suffix on `collect_*`: `Restores 25 health (now 38 → 63)`, `No effect at 100 health`, `Armor: absorbs 1/3 of damage, 100 points`, `Berserk: health to 100, fists ×10 this level`.
  Optionally add a state line: `Damage taken in the last 3 s: N` (DamageHistory exists).
- **Rules.** Facts about effects; usefulness is left to the model. **Do not** write "useful" or "needed".
- **Impact.** Medium, together with P1.

### P6. Weapon facts: switch delay, real rocket safe distance, shots-to-kill

- **Knowledge.** A switch costs ~0.9 s with no fire. Rocket splash reaches ~5.5-6 m centre to centre, while the code says 4 m. Shotgun up close kills zombies and shotgun guys in one shot.
- **Form.**
  - Weapon option fact `Switching takes about 0.9 s without firing.`
  - Change `ROCKET_SAFE_METERS` to ≈ 5.5 plus the target's radius. This is a fact correction to `Can hit visible enemies from here`.
  - Optional: `Average shots to kill <nearest enemy> at this distance: N`, computed from HP and expected pellet hits.
- **Rules.** Facts. **Authority flag:** the existing weapon instructions ("Prefer shotgun to pistol", "Pistol: use only when stronger guns have no ammo") are already priorities in the text. Shots-to-kill is safer than those and could replace them.
- **Impact.** Medium for rocket self-kills (check the logs for rocket deaths); low-medium otherwise.

### P7. Event-triggered re-decision for close projectiles

- **Knowledge.** A fireball seen under ~8 m cannot be dodged with 0.5 s latency.
- **Form.** Decision timing. When a new projectile with time-to-impact < 0.6 s appears, ask the movement question at once, without waiting for the 0.5 s tick.
  The movement option text already carries the incoming side and time. Add `sidestep needs about 0.2 s running`.
- **Rules.** Scheduling only. The model still chooses the direction.
- **Impact.** Low-medium (imp and caco chip damage).

### P8. Door facts: "monsters can / cannot open this door"; doors block sound

- **Knowledge.** Section 1, door specials.
- **Form.** Fact on the `open_door` option and in the state line. Derived from the line special number, which is generic across maps.
- **Rules.** Fact.
- **Impact.** Low-medium. Enables learned door dancing and quieter fights.

### P9. Monster awareness facts

- **Knowledge.** Reaction delay; partial damage triggers instant return fire.
- **Form.** Per enemy: `facing you: yes/no`, `has attacked you: yes/no`, `hit by you: yes/no`. From the actor angle and damage events.
- **Rules.** Fact.
- **Impact.** Low; useful for the "kill in one burst or leave it" pattern.

### P10. Infighting fact (optional)

- **Form.** `Imp #5 is between you and Zombie #9` (line-of-fire occlusion).
- **Rules.** Fact.
- **Impact.** Low on these maps.

### Avoid

Any combined "threat score", "most dangerous enemy" ordering, "retreat recommended" text, or automatic health detours. Each sets priority in code and breaks the authority rule.
