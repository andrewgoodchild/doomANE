# Planner v6 (commit a17a006), frozen for A/B against later versions; config planner6 in repeat_map.py.
"""System 1 planner for doomLaya: an event-driven `objective` choice asked of the same model.

Laya chooses a command every 0.5 s from one snapshot and has no plan or memory, so on MAP02/MAP03
it flips between route targets and loops (see docs/LAB_NOTEBOOK.md). Here code does the search and Laya does
the choosing:

- Code lists objectives from the live navigator and mission: each missing key, each unused switch
  or lift, the exit, and exploring. Each option says whether it is reachable now and what it would
  newly make reachable, from a what-if copy of the navigator (key held, remote door open, lift
  raised), plus memory: how often it was chosen, failed or completed.
- Laya answers one ordinary `choice` question, "objective", through the same /predict endpoint.
  It is asked only on events: no objective, objective done or failed, keys/doors/floors changed,
  respawn, or 30 s without one of those.
- The objective narrows the route options of the regular 2 Hz questions (use_switch only for its
  switch, exit only when it is the exit, item errands to keys plus anything within 6 m). Fighting,
  doors, looking and nearby pickups stay free. Every executed command is still Laya's answer.

With $PLANNER_FIGHT=1 a second event-driven choice, "fight", is asked while enemies are visible
(every 3 s, or sooner when health drops): fight here, heal at a named health item, or continue the
objective (backing off is behind $PLANNER_BACK_OFF=1). Code writes the facts into each option (enemies, ammo, health lost in the
last 3 s and how long health lasts at that rate, the health item's value and distance), and the
answer narrows the 2 Hz command question to attack, retreat, pickup of that item, or the route.

Every objective and fight decision is appended to $PLANNER_LOG (one JSON line each).

Run instead of agent.py, from ext/doomLaya-dev:
  .venv/bin/python ../../doom/planner.py <agent.py arguments>
"""
import copy
import json
import math
import os
import runpy
import sys
import time
import urllib.request
from pathlib import Path

REPLAN_SECONDS = 30  # check progress on the objective this often
PROGRESS_METERS = 2  # closer than this over a check counts as progress
NEAR_ITEM_METERS = 6
NEAR_WEAPON_METERS = 15  # farther weapons are objectives of their own, not errands
STUCK_PICKUP_SECONDS = 15  # an item targeted this long without getting 1 m closer is set aside
STUCK_FORGET_SECONDS = 30  # brief switches to other commands do not reset an item's stuck clock
SET_ASIDE_SECONDS = 60
OBJECTIVE_FAILURES = 2  # an objective that failed this often is set aside
OBJECTIVE_SET_ASIDE_SECONDS = 90
LOW_HEALTH = 50
LIFT_REST_SECONDS = 45  # a lift just ridden is not offered again straight away (no riding back)
FIGHT_SECONDS = 3  # a fight plan is re-asked this often while enemies are visible
FIGHT_HP_DROP = 15  # or sooner when health fell this much since it was asked
HEAL_METERS = 25
BACK_OFF = os.environ.get("PLANNER_BACK_OFF") == "1"
# Doom monster health, and rough sustained damage per second by weapon slot at half accuracy
ENEMY_HEALTH = {"Zombieman": 20, "ShotgunGuy": 30, "ChaingunGuy": 70, "DoomImp": 60, "Demon": 150, "Spectre": 150,
                "LostSoul": 100, "Cacodemon": 400, "HellKnight": 500, "BaronOfHell": 1000, "Revenant": 300, "Fatso": 600,
                "Arachnotron": 500, "PainElemental": 400, "Archvile": 700, "Cyberdemon": 4000, "SpiderMastermind": 3000,
                "WolfensteinSS": 50}
# rough damage per second each visible monster deals at medium range
ENEMY_DPS = {"Zombieman": 3, "ShotgunGuy": 6, "ChaingunGuy": 12, "DoomImp": 4, "Demon": 8, "Spectre": 8, "LostSoul": 5,
             "Cacodemon": 6, "HellKnight": 8, "BaronOfHell": 10, "Revenant": 8, "Fatso": 12, "Arachnotron": 12,
             "PainElemental": 5, "Archvile": 15, "Cyberdemon": 40, "SpiderMastermind": 40, "WolfensteinSS": 6}
WEAPON_DPS = {1: 10, 2: 12, 3: 35, 4: 45, 5: 60, 6: 120, 7: 150}
HEALTH_VALUES = {"HealthBonus": 1, "Stimpack": 10, "Medikit": 25, "Soulsphere": 100, "Megasphere": 200}
FIGHT_INSTRUCTIONS = ("Choose how to handle the visible enemies so the player survives and finishes the level. "
                      "Fight when the fight can be won, back off or heal when health would run out first, "
                      "and continue the objective when the enemies are not a threat.")
INSTRUCTIONS = ("Choose the next objective to finish the level. Prefer objectives that are reachable now and "
                "open new areas toward missing keys and the exit. Avoid objectives already done, or tried "
                "repeatedly without opening anything new.")


class Planner:
    def __init__(self, endpoint, log_path=None, fight=False):
        self.endpoint = endpoint
        self.log_path = log_path
        self.fight_enabled = fight
        self.reset()

    def reset(self):
        self.controller = None
        self.state = None
        self.tick = 0
        self.objective = None  # key, kind, target, since_tick, signature
        self.memory = {}  # key -> chosen, failed, done
        self.signature = None
        self.opens_cache = {}  # (signature, area) -> {option: opens}
        self.rode = False  # the current lift objective has reached its ride phase
        self.fight = None  # choice, tick, hp, item
        self.hp_history = []  # (tick, hp) over the last few seconds
        self.pickup_progress = {}  # item id -> best distance, ticks targeted since it, last tick targeted
        self.objective_set_aside = {}  # option -> tick until which it is not offered
        self.set_aside = {}  # item id -> tick until which it is not offered

    # ----- what-if reachability -------------------------------------------------------------
    def _reach(self, nav, origin, unlock=(), open_sectors=(), raise_lift=None, lift_height=None):
        """Cells reachable from origin on a copy of the navigator with the given changes."""
        trial = copy.copy(nav)
        trial.blocked = set()
        locked = set()
        for key, cells in nav.key_blocks.items():
            if key not in set(self.state.get("keys", ())) | set(unlock):
                locked |= cells
        for sector in self.state.get("closed_remote_doors", ()):
            if sector not in open_sectors:
                locked |= nav.sector_blocks.get(sector, set())
        trial.locked = locked
        if raise_lift is not None:
            trial.sector_heights = list(nav.sector_heights)
            trial.floors = dict(nav.floors)
            affected = set()
            for sector in raise_lift["sectors"]:
                trial.sector_heights[sector] = raise_lift["upper_floor"] if lift_height is None else lift_height
                affected.update(nav.sector_cells.get(sector, ()))
            for node in affected:
                trial.floors[node] = max(trial.sector_heights[i] for i in nav.body_sectors[node])
        return trial.reachable(origin)

    def _ride_down(self, nav, lift):
        """Cells reachable after riding the lift down: it lowers to its lowest neighbouring floor."""
        cells = set().union(*(set(map(tuple, nav.sector_cells.get(i, ()))) for i in lift["sectors"]))
        around = {(x + dx, y + dy) for x, y in cells for dx in (-1, 0, 1) for dy in (-1, 0, 1)} - cells
        floors = [nav.floors[c] for c in around if c in nav.floors]
        on = [c for c in cells if c in nav.floors]
        if not floors or not on:
            return set()
        start = min(on, key=lambda c: math.dist(nav.point(c), (lift["x"], lift["y"])))
        return self._reach(nav, nav.point(start), raise_lift=lift, lift_height=min(floors))

    def _landmarks(self):
        """Points worth reaching: keys not held, the exit, unused mechanisms."""
        s, mission = self.state, self.controller.mission
        held = set(s.get("keys", ()))
        marks = {}
        for key in mission.data.get("key_markers", ()):
            if key["color"] not in held:
                marks.setdefault(f"{key['color']} key", (key["x"], key["y"]))
        if mission.exit:
            marks["the exit"] = mission.objective()
        for m in s.get("switches", ()):
            if not m["activated"]:
                marks[f"{m['name'].lower()} #{m['id']}"] = m["approach"]
        return marks

    def _opens(self, reach_now, reach_after, nav, own=None):
        marks = self._landmarks()
        return [name for name, point in marks.items()
                if name != own and nav.nearest(point) in reach_after and nav.nearest(point) not in reach_now]

    # ----- options ----------------------------------------------------------------------------
    def options(self):
        s, c = self.state, self.controller
        nav, mission = c.navigator, c.mission
        here = (s["x"], s["y"])
        now = nav.reachable(here)

        def reachable(point):
            return nav.nearest(point) in now

        def memory(key):
            m = self.memory.get(key)
            if not m:
                return "Not tried yet."
            parts = []
            if m["done"]:
                parts.append("already done")
            if m["chosen"]:
                parts.append(f"chosen {m['chosen']} time{'s' if m['chosen'] > 1 else ''}")
            if m["failed"]:
                parts.append(f"failed {m['failed']} time{'s' if m['failed'] > 1 else ''}")
            return "History: " + ", ".join(parts) + "."

        area = (self._signature(), min(now) if now else None)
        cache = self.opens_cache.setdefault(area, {})

        def opens_for(name, compute):
            if name not in cache:
                cache[name] = compute()
            return cache[name]

        # Every objective, with whether it is reachable now and the landmarks it would open.
        found = {}  # option -> kind, target, landmark, distance, reachable, opens, extra
        held = set(s.get("keys", ()))
        for key in mission.data.get("key_markers", ()):
            color = key["color"]
            name = f"key_{color}"
            if color in held or name in found:
                continue
            point = (key["x"], key["y"])
            found[name] = dict(kind="key", target=color, landmark=f"{color} key", label=f"Collect the {color} key", point=point,
                               distance=math.dist(here, point) / 32, reachable=reachable(point), extra="",
                               opens=opens_for(name, lambda color=color: self._opens(
                                   now, self._reach(nav, here, unlock=(color,)), nav, own=f"{color} key")))
        for m in s.get("switches", ()):
            if m["locked"] or (m["activated"] and m.get("kind") != "lift"):
                continue
            point = m["board"] if m.get("kind") == "lift" else m["approach"]
            own = f"{m['name'].lower()} #{m['id']}"
            name = f"switch_{m['id']}"

            def compute(m=m, own=own):
                if m.get("kind") == "lift":
                    # ridden up from its boarding point, or down from the top
                    after = self._reach(nav, m["board"], raise_lift=m) | self._ride_down(nav, m) | now
                elif m.get("kind") == "door":
                    after = self._reach(nav, here, open_sectors=m["sectors"])
                else:
                    after = now
                return self._opens(now, after, nav, own=own)
            found[name] = dict(kind="switch", target=str(m["id"]), landmark=own, label=f"Use {own}", point=point,
                               distance=m["distance"], reachable=reachable(point), opens=opens_for(name, compute), lift=m.get("kind") == "lift",
                               extra=" Lowers the exit platform." if m.get("route_exit") else "")
        from doomlib.items import WEAPONS
        inventory = s.get("inventory", {})
        for item_id, item in c.known.items():
            name, slot = item.get("name"), WEAPONS.get(item.get("name"))
            point = (item["x"], item["y"])
            if str(item_id) in self.set_aside or not reachable(point):
                continue
            if slot is not None and not inventory.get(str(slot), {}).get("owned"):
                kind, label, extra = "weapon", f"Collect the {name} (#{item_id})", " A weapon not yet owned."
            elif item.get("category") == "Health" and s.get("hp", 100) < LOW_HEALTH and "Bonus" not in name:
                kind, label, extra = "health", f"Collect the {name} (#{item_id})", f" Health is {s.get('hp', 0):.0f}."
            else:
                continue
            found[f"item_{item_id}"] = dict(kind=kind, target=str(item_id), landmark=f"{name} #{item_id}", label=label, point=point,
                                           distance=math.dist(here, point) / 32, reachable=True, opens=[], extra=extra)
        if mission.exit:
            point = mission.objective()
            found["exit"] = dict(kind="exit", target=None, landmark="the exit", label="Go to the exit", point=point,
                                 distance=math.dist(here, point) / 32, opens=[], extra="",
                                 reachable=reachable(point) and "exit" not in s.get("command_failures", {}))
        by_landmark = {o["landmark"]: o for o in found.values()}

        def chain(o):
            # Code follows the consequences two steps; Laya only chooses the first step.
            parts = []
            for landmark in o["opens"]:
                further = by_landmark.get(landmark, {}).get("opens", [])
                parts.append(landmark + (f" (which opens {', '.join(further)})" if further else ""))
            return ", ".join(parts) or "nothing new found"

        for key, until in list(self.objective_set_aside.items()):
            if self.tick >= until:
                del self.objective_set_aside[key]
        offered = {k: o for k, o in found.items() if o["reachable"] and k not in self.objective_set_aside
                   and str(o.get("target")) not in s.get("target_failures", {})
                   and not (o.get("lift") and not o["opens"])}  # a lift that leads nowhere new is not an objective
        result = {}
        if not offered and not any(o["reachable"] for o in found.values()):
            # nothing reachable at all: describe everything not set aside, so the choice still has content
            offered = {k: o for k, o in found.items() if k not in self.objective_set_aside}
        for name, o in offered.items():
            where = "reachable now" if o["reachable"] else "not reachable from here"
            if o["kind"] == "exit":
                consequence = f"Finishes the level. Keys held: {', '.join(sorted(held)) or 'none'}."
            elif o["kind"] in ("weapon", "health"):
                consequence = o["extra"].strip()
            else:
                consequence = f"Opens: {chain(o)}.{o['extra']}"
            result[name] = dict(kind=o["kind"], target=o["target"], point=o.get("point"), lift=o.get("lift", False),
                                text=f"{o['label']}, {o['distance']:.0f}m away; {where}. {consequence} {memory(name)}")
        result["explore"] = dict(kind="explore", target=None, text=(
            f"No fixed objective: explore accessible areas for another route. {memory('explore')}"))
        return result

    # ----- events -------------------------------------------------------------------------------
    def _done(self):
        o, s = self.objective, self.state
        if o["kind"] == "key":
            return o["target"] in s.get("keys", ())
        if o["kind"] == "switch" and o.get("lift"):
            m = next((m for m in s.get("switches", ()) if str(m["id"]) == o["target"]), None)
            if m is not None and m.get("phase") == "ride":
                self.rode = True
            return m is None or (self.rode and m.get("phase") != "ride")
        if o["kind"] == "switch":
            m = next((m for m in s.get("switches", ()) if str(m["id"]) == o["target"]), None)
            return m is None or m["activated"]
        if o["kind"] in ("weapon", "health"):
            return o["target"] not in {str(k) for k in self.controller.known}
        return False

    def _failed(self):
        o, s = self.objective, self.state
        if o["kind"] in ("weapon", "health") and (o["target"] in s.get("target_failures", {}) or o["target"] in self.set_aside):
            return True
        if o["kind"] == "switch" and o["target"] in s.get("target_failures", {}):
            return True
        if o["kind"] == "exit" and "exit" in s.get("command_failures", {}):
            return True
        if o["kind"] == "key":
            item = next((k for k, i in self.controller.known.items()
                         if i.get("category") == "Key" and o["target"] in i.get("name", "").lower()), None)
            return item is not None and str(item) in s.get("target_failures", {})
        return False

    def _distance(self):
        point = (self.objective or {}).get("point")
        if point is None or self.state is None:
            return None
        return math.dist((self.state["x"], self.state["y"]), point) / 32

    def track(self, s, tick):
        """Set aside items the executor has been 'picking up' for too long without collecting them."""
        self.hp_history = [(t, hp) for t, hp in self.hp_history if tick - t <= 4 * 35] + [(tick, s.get("hp", 0))]
        for item_id, until in list(self.set_aside.items()):
            if tick >= until:
                del self.set_aside[item_id]
        ex = s.get("execution", {})
        item = self.controller.known.get(ex.get("target_id")) if self.controller else None
        if ex.get("action") == "pickup" and ex.get("status") == "executing" and item is not None:
            item_id = str(ex["target_id"])
            distance = math.dist((s["x"], s["y"]), (item["x"], item["y"])) / 32
            best, stuck, last = self.pickup_progress.get(item_id, (distance, 0, tick))
            if tick - last > STUCK_FORGET_SECONDS * 35 or distance < best - 1:
                best, stuck = distance, 0  # not targeted for a while, or 1 m closer
            elif tick - last <= 70:
                stuck += tick - last  # only time spent targeting the item counts
            if stuck >= STUCK_PICKUP_SECONDS * 35:
                self.set_aside[item_id] = tick + SET_ASIDE_SECONDS * 35
                best, stuck = distance, 0
            self.pickup_progress[item_id] = (min(best, distance), stuck, tick)

    def _signature(self):
        s = self.state
        return (tuple(sorted(s.get("keys", ()))), tuple(s.get("closed_remote_doors", ())),
                tuple(sorted(m["id"] for m in s.get("switches", ()) if m["activated"] and m.get("kind") != "lift")))

    def update(self, packet):
        """Re-plan when needed, then narrow the packet's route options to the objective."""
        if self.controller is None or self.state is None or self.controller.mission is None:
            return packet
        signature = self._signature()
        reason = None
        if self.objective is None:
            reason = "no objective"
        elif self._done():
            self.memory[self.objective["key"]]["done"] = True
            if self.objective.get("lift"):
                self.objective_set_aside[self.objective["key"]] = self.tick + LIFT_REST_SECONDS * 35
            reason = "objective done"
        elif self._failed():
            self.memory[self.objective["key"]]["failed"] += 1
            reason = "objective failed"
        elif signature != self.objective["signature"]:
            reason = "keys, doors or mechanisms changed"
        elif self.tick - self.objective["since_tick"] >= REPLAN_SECONDS * 35:
            distance = self._distance()
            if distance is not None and self.objective["distance"] is not None and \
                    self.objective["distance"] - distance >= PROGRESS_METERS:
                self.objective.update(since_tick=self.tick, distance=distance)  # getting closer: keep it
            else:
                self.memory[self.objective["key"]]["failed"] += 1
                reason = f"no progress in {REPLAN_SECONDS} s"
        if reason and self.objective is not None and self.memory[self.objective["key"]]["failed"] >= OBJECTIVE_FAILURES \
                and not self.memory[self.objective["key"]]["done"]:
            self.objective_set_aside[self.objective["key"]] = self.tick + OBJECTIVE_SET_ASIDE_SECONDS * 35
            self.memory[self.objective["key"]]["failed"] = 0
        if reason:
            self.plan(packet["state"], reason, signature)
        narrowed = self.narrow(packet) if self.objective else copy.deepcopy(packet)
        return self.fight_update(packet, narrowed) if self.fight_enabled else narrowed

    def plan(self, state_text, reason, signature):
        started = time.perf_counter()
        options = self.options()
        options_ms = (time.perf_counter() - started) * 1000
        question = {"type": "choice", "instructions": INSTRUCTIONS,
                    "criteria": {k: v["text"] for k, v in options.items()}}
        answer = self._ask(state_text, "objective", question)
        choice = answer["choice"]
        o = options[choice]
        self.objective = dict(key=choice, kind=o["kind"], target=o["target"], since_tick=self.tick, signature=signature,
                              point=o.get("point"), lift=o.get("lift", False))
        self.rode = False
        self.objective["distance"] = self._distance()
        m = self.memory.setdefault(choice, dict(chosen=0, failed=0, done=False))
        m["chosen"] += 1
        self._log(dict(reason=reason, choice=choice, probabilities=answer.get("probabilities"), options=question["criteria"],
                       options_ms=round(options_ms, 1), latency_ms=round((time.perf_counter() - started) * 1000, 1)))

    def _ask(self, state_text, name, question):
        body = json.dumps({"state": state_text, "questions": {name: question}, "model": "doom-adapted"}).encode()
        request = urllib.request.Request(self.endpoint, body, {"content-type": "application/json"})
        with urllib.request.urlopen(request, timeout=15) as response:
            return json.load(response)["answers"][name]

    def _log(self, row):
        if self.log_path:
            with open(self.log_path, "a") as handle:
                handle.write(json.dumps(dict(tick=self.tick, game_seconds=round(self.tick / 35, 2), **row)) + "\n")

    # ----- fights -------------------------------------------------------------------------------
    def fight_options(self, packet):
        """Fight, back off, heal at a named item, or continue the objective, each with code's facts."""
        s, command = self.state, packet["questions"]["command"]["criteria"]
        enemies = [e for e in packet.get("targets", {}).get("enemy", {}).values() if e.get("visible")]
        hp = s.get("hp", 0)
        before = self.hp_history[0][1] if self.hp_history and self.tick - self.hp_history[0][0] >= 2 * 35 else hp
        lost = max(0.0, before - hp)
        span = max(1.0, (self.tick - self.hp_history[0][0]) / 35) if self.hp_history else 1.0
        rate = max(lost / span, sum(ENEMY_DPS.get(e["name"], 6) for e in enemies))  # observed, or what they can deal
        weapon = s.get("inventory", {}).get(str(s.get("weapon_slot")), {})
        names = {}
        for e in enemies:
            names[e["name"]] = names.get(e["name"], 0) + 1
        crowd = ", ".join(f"{n} {name}" for name, n in sorted(names.items()))
        nearest = min(e["distance"] for e in enemies)
        lasts = f"health lasts about {hp / rate:.0f} s under this fire"
        ammo = weapon.get("ammo", 0)
        slot = s.get("weapon_slot")
        dps = WEAPON_DPS.get(slot, 20) if ammo > 0 or slot == 1 else 0
        total = sum(ENEMY_HEALTH.get(e["name"], 100) for e in enemies)
        kill = f"killing them takes about {total / dps:.0f} s with this weapon" if dps else "this weapon is out of ammo"
        # code compares the two clocks; Laya weighs that outcome against backing off, healing or moving on
        winnable = dps and hp / rate >= total / dps
        outcome = ("the player survives and the enemies go down" if winnable else "the player dies before the enemies do")
        options = {"fight": dict(text=(
            f"Fight here. Likely outcome: {outcome}."), facts=f"{kill}; {lasts}; enemies: {crowd}")}
        if "retreat" in command and BACK_OFF:
            # off by default: the executor's retreat walks backwards without firing, and on MAP03 (exec4p6)
            # deaths followed back-off choices in 17 of 22 cases
            options["back_off"] = dict(text=(f"Back off. Likely outcome: the player survives and stops taking damage. "
                                             f"Back away from the enemies without firing to break line of sight; nearest enemy {nearest:.0f} m."))
        if "pickup" in command and hp < 100:
            items = packet.get("targets", {}).get("item", {})
            health = sorted(((k, i) for k, i in items.items() if i.get("category") == "Health" and i.get("reachable")
                             and HEALTH_VALUES.get(i["name"], 10) >= 10
                             and i.get("distance", 999) <= HEAL_METERS and k in packet["questions"].get("item", {}).get("criteria", {})),
                            key=lambda ki: ki[1]["distance"])
            for k, i in health[:2]:
                gain = min(HEALTH_VALUES.get(i["name"], 10), 200 - hp)
                options[f"heal_{k}"] = dict(item=k, text=(
                    f"Heal. Likely outcome: the player survives longer with {gain:.0f} more health (now {hp:.0f}). Go for the {i['name']} "
                    f"#{i['id']}, {i['distance']:.0f} m away and reachable."))
        o = self.objective
        route = dict(switch="use_switch", key="pickup", weapon="pickup", health="pickup", exit="exit").get((o or {}).get("kind"))
        if route in command:
            distance = self._distance()
            options["continue"] = dict(text=(
                f"Move on. Likely outcome: progress on the objective ({o['key'].replace('_', ' ')}"
                + (f", {distance:.0f} m away" if distance is not None else "") + ") while still under fire, firing only on the way."))
        return options

    def fight_update(self, original, packet):
        """Ask the fight question on events and narrow the command question to its answer."""
        command = packet["questions"].get("command", {}).get("criteria", {})
        visible = any(e.get("visible") for e in original.get("targets", {}).get("enemy", {}).values())
        if "attack" not in command or not visible:
            self.fight = None
            return packet
        hp = self.state.get("hp", 0)
        if self.fight is None or self.tick - self.fight["tick"] >= FIGHT_SECONDS * 35 or hp <= self.fight["hp"] - FIGHT_HP_DROP:
            started = time.perf_counter()
            options = self.fight_options(original)
            question = {"type": "choice", "instructions": FIGHT_INSTRUCTIONS, "criteria": {k: v["text"] for k, v in options.items()}}
            answer = self._ask(packet["state"], "fight", question)
            choice = answer["choice"]
            self.fight = dict(choice=choice, tick=self.tick, hp=hp, item=options[choice].get("item"))
            self._log(dict(question="fight", choice=choice, probabilities=answer.get("probabilities"), options=question["criteria"],
                           latency_ms=round((time.perf_counter() - started) * 1000, 1)))
        choice, keep = self.fight["choice"], None
        if choice == "fight":
            keep = {"attack"}
        elif choice == "back_off":
            keep = {"retreat"}
        elif choice.startswith("heal_"):
            items = packet["questions"].get("item", {})
            criterion = original["questions"].get("item", {}).get("criteria", {}).get(self.fight["item"])
            if criterion is not None and "item" in packet["questions"]:
                items["criteria"] = {self.fight["item"]: criterion}
                command.setdefault("pickup", original["questions"]["command"]["criteria"].get("pickup"))
                keep = {"pickup"}
        elif choice == "continue":
            keep = set(command) - {"attack", "retreat", "wait"}
        if keep and keep & set(command):
            for k in set(command) - keep - {"look_back"}:  # the command head requires the look-back option
                command.pop(k)
        packet["fight"] = self.fight["choice"]
        return packet

    def narrow(self, packet):
        packet = copy.deepcopy(packet)
        o = self.objective
        questions = packet["questions"]
        command = questions.get("command", {}).get("criteria", {})
        items = packet.get("targets", {}).get("item", {})
        if "item" in questions:
            # Items the executor could not collect are set aside whatever the objective.
            criteria = questions["item"]["criteria"]
            kept = {k: v for k, v in criteria.items() if k not in self.set_aside}
            if kept:
                questions["item"]["criteria"] = kept
            elif "pickup" in command and len(command) > 1:
                command.pop("pickup")
        door = (packet.get("commands", {}).get("open_door") or {}).get("target") or {}
        if "open_door" in command and len(command) > 1 and str(door.get("id")) in self.state.get("target_failures", {}):
            # the executor already failed to reach this door; doomLaya excludes such commands until the area changes
            command.pop("open_door")
        packet["objective"] = dict(key=o["key"], kind=o["kind"], target=o["target"])
        if o["kind"] == "explore":
            # Exploring is the absence of an objective: nothing else is narrowed.
            return packet
        keep = {"use_switch": o["kind"] == "switch", "exit": o["kind"] == "exit"}
        for action, allowed in keep.items():
            if action in command and not allowed and len(command) > 1:
                command.pop(action)
        if o["kind"] == "switch" and "switch" in questions:
            criteria = questions["switch"]["criteria"]
            if o["target"] in criteria:
                questions["switch"]["criteria"] = {o["target"]: criteria[o["target"]]}
        if "item" in questions and "pickup" in command:
            # While pursuing an objective, item errands are limited to the objective's item, keys,
            # weapons not yet owned within NEAR_WEAPON_METERS, health when low, and anything within NEAR_ITEM_METERS.
            from doomlib.items import WEAPONS
            inventory, hp = self.state.get("inventory", {}), self.state.get("hp", 100)
            criteria = questions["item"]["criteria"]

            def wanted(k, i):
                if o["kind"] in ("weapon", "health") and k == o["target"]:
                    return True
                if i.get("category") == "Key":
                    return o["kind"] != "key" or o["target"] in i.get("name", "").lower()
                slot = WEAPONS.get(i.get("name"))
                if slot is not None and not inventory.get(str(slot), {}).get("owned"):
                    return i.get("distance", 999) <= NEAR_WEAPON_METERS
                if i.get("category") == "Health" and hp < LOW_HEALTH:
                    return True
                return i.get("distance", 999) <= NEAR_ITEM_METERS
            kept = {k: v for k, v in criteria.items() if wanted(k, items.get(k, {}))}
            if kept:
                questions["item"]["criteria"] = kept
            elif len(command) > 1:
                command.pop("pickup")
        return packet


def install(endpoint, log_path):
    sys.path.insert(0, os.getcwd())
    import doomlib.decision_questions as dq
    from doomlib.executor import Executor
    planner = Planner(endpoint, log_path, fight=os.environ.get("PLANNER_FIGHT") == "1")

    original_init, original_observe, original_mask = Executor.__init__, Executor.observe, dq.mask_unreachable_items

    def init(self, *a, **k):
        original_init(self, *a, **k)
        planner.reset()  # a new controller means a new map or a respawn
        planner.controller = self

    def observe(self, s, tick, sectors=None):
        result = original_observe(self, s, tick, sectors)
        planner.state, planner.tick = s, tick
        planner.track(s, tick)
        return result

    def mask(packet, *a, **k):
        return planner.update(original_mask(packet, *a, **k))

    Executor.__init__, Executor.observe, dq.mask_unreachable_items = init, observe, mask
    return planner


def main():
    argv = sys.argv[1:]
    endpoint = argv[argv.index("--endpoint") + 1] if "--endpoint" in argv else "http://127.0.0.1:8003/predict"
    if "--mask-unreachable-items" not in argv:
        raise SystemExit("planner.py hooks --mask-unreachable-items; pass it")
    install(endpoint, os.environ.get("PLANNER_LOG"))
    sys.argv = ["agent.py", *argv]
    runpy.run_path(str(Path.cwd() / "agent.py"), run_name="__main__")


if __name__ == "__main__":
    main()
