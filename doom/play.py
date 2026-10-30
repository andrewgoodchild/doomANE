"""A local decision model plays Doom (ViZDoom) through the Jev-compatible ANE server.

    laya-serve ... --port 8008   # any Jev-compatible /v1/systemone server, in another terminal
    uv run python doom/play.py --scenario defend_the_center --policy laya --episodes 10
    uv run python doom/play.py --scenario all --policy all --episodes 10 # the baseline table
    uv run python doom/play.py --scenario defend_the_center --policy laya --episodes 1 --video out.mp4

Each decision the game state is written as plain English (health, ammo, what is in view, where it
is relative to the crosshair) and sent with two typed questions to `/v1/systemone`; the answers
become button presses held for `--skip` tics (4 tics = 8.75 decisions per second of game time).
The game runs in synchronous mode, so scores do not depend on latency; latency is measured and
reported separately to show whether the loop would keep up in real time (35 tics per second).

Policies: `random` (uniform over the scenario's actions), `scripted` (the rule the questions
describe, applied to the same features: an upper bound for this interface), `laya` and
`laya-typed-decisions` (our two checkpoints on the Neural Engine).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np
import vizdoom as vzd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.client import DecisionClient, answer_probs, argmax  # noqa: E402

ENEMIES = {"Cacodemon", "Demon", "MarineChainsawVzd", "Zombieman", "ShotgunGuy", "DoomImp", "ChaingunGuy",
           "HellKnight", "BaronOfHell", "LostSoul", "Spectre", "Revenant"}
ITEMS = {"Medikit", "Stimpack", "CustomMedikit"}
HFOV = 90.0          # ViZDoom default horizontal field of view
AIM_DEG = 3.0        # "in the crosshair"
STEER_DEG = 6.0

SCENARIOS = {
    "basic": {"targets": ENEMIES, "move": "strafe"},
    "defend_the_center": {"targets": ENEMIES, "move": "turn"},
    "health_gathering": {"targets": ITEMS, "move": "steer"},
}


def bearing(label, width=320):
    cx = label.x + label.width / 2
    return (cx - width / 2) / (width / 2) * (HFOV / 2)   # degrees, negative = left of crosshair


def where(deg):
    a = abs(deg)
    if a <= AIM_DEG:
        return "in the crosshair"
    side = "left" if deg < 0 else "right"
    return (f"slightly {side}" if a <= 12 else side if a <= 30 else f"far {side}") + f" ({a:.0f} degrees)"


def observe(state, game, sc):
    """Plain-English state plus the features the scripted rule uses."""
    gv = {str(v).split(".")[-1]: x for v, x in zip(game.get_available_game_variables(), state.game_variables)}
    px, py = gv.get("POSITION_X", 0.0), gv.get("POSITION_Y", 0.0)
    w = game.get_screen_width()
    seen = []
    for l in state.labels:
        if l.object_name in sc["targets"]:
            d = math.hypot(l.object_position_x - px, l.object_position_y - py)
            seen.append((l.object_name, bearing(l, w), d))
    seen.sort(key=lambda t: t[2])
    kind = "enemies" if sc["targets"] is ENEMIES else "medikits"
    parts = []
    if "HEALTH" in gv:
        parts.append(f"Health {gv['HEALTH']:.0f}.")
    if "AMMO2" in gv:
        parts.append(f"Ammo {gv['AMMO2']:.0f}.")
    if seen:
        desc = "; ".join(f"{n} {where(b)}, {d:.0f} units away" for n, b, d in seen[:4])
        parts.append(f"{kind.capitalize()} in view, nearest first: {desc}.")
        parts.append(f"The nearest is {where(seen[0][1])}.")
    else:
        parts.append(f"No {kind} in view.")
    return " ".join(parts), (seen[0][1] if seen else None)


def questions(sc):
    if sc["move"] == "steer":
        return {"steer": {"type": "choice", "instructions": "Which way should the player go to pick up a medikit?",
                          "criteria": {
                              "left": "Turn left: the nearest medikit is to the left of the crosshair, or no medikit is in view.",
                              "right": "Turn right: the nearest medikit is to the right of the crosshair.",
                              "forward": "Walk forward: the nearest medikit is in the crosshair or slightly off it."}}}
    verb = "Strafe" if sc["move"] == "strafe" else "Turn"
    return {
        "aim": {"type": "choice", "instructions": f"Which way should the player {verb.lower()} to aim at the nearest enemy?",
                "criteria": {
                    "left": f"{verb} left: the nearest enemy is to the left of the crosshair, or no enemy is in view (turn to look for one).",
                    "right": f"{verb} right: the nearest enemy is to the right of the crosshair.",
                    "hold": "Hold still: the nearest enemy is in the crosshair."}},
        "fire": {"type": "noul", "instructions": "Is an enemy in the crosshair?",
                 "criteria": {"true": "An enemy is in the crosshair, so shooting now would hit it.",
                              "false": "No enemy is in the crosshair, so shooting now would miss."}},
    }


def scripted(sc, b):
    if sc["move"] == "steer":
        if b is None or b < -STEER_DEG:
            return {"steer": "left"}
        return {"steer": "right" if b > STEER_DEG else "forward"}
    if b is None:
        return {"aim": "left", "fire": "false"}
    if abs(b) <= AIM_DEG:
        return {"aim": "hold", "fire": "true"}
    return {"aim": "left" if b < 0 else "right", "fire": "false"}


def to_buttons(game, sc, choice):
    names = [str(x).split(".")[-1] for x in game.get_available_buttons()]
    press = set()
    if sc["move"] == "steer":
        press.add({"left": "TURN_LEFT", "right": "TURN_RIGHT", "forward": "MOVE_FORWARD"}[choice["steer"]])
    else:
        prefix = "MOVE_" if sc["move"] == "strafe" else "TURN_"
        if choice["aim"] in ("left", "right"):
            press.add(prefix + choice["aim"].upper())
        if choice["fire"] == "true":
            press.add("ATTACK")
    return [n in press for n in names]


def make_game(scenario, seed, record=False):
    g = vzd.DoomGame()
    g.load_config(os.path.join(vzd.scenarios_path, scenario + ".cfg"))
    g.set_window_visible(False)
    g.set_labels_buffer_enabled(True)
    g.set_screen_format(vzd.ScreenFormat.RGB24)
    for v in (vzd.GameVariable.POSITION_X, vzd.GameVariable.POSITION_Y, vzd.GameVariable.KILLCOUNT):
        g.add_available_game_variable(v)
    g.set_mode(vzd.Mode.PLAYER)
    g.set_seed(seed)
    g.init()
    return g


def overlay(frame, lines):
    from PIL import Image, ImageDraw
    img = Image.fromarray(frame).resize((640, 480), Image.NEAREST)
    pad = Image.new("RGB", (640, 480 + 18 * len(lines) + 8), (12, 12, 16))
    pad.paste(img, (0, 0))
    d = ImageDraw.Draw(pad)
    for i, t in enumerate(lines):
        d.text((8, 484 + 18 * i), t[:100], fill=(230, 230, 230))
    return np.asarray(pad)


def play(scenario, policy, episodes, skip, seed, client=None, video=None):
    sc = SCENARIOS[scenario]
    qs = questions(sc)
    out, lat, agree, ndec = [], [], [], 0
    agree_q = {q: [] for q in qs}
    writer = None
    if video:
        import imageio.v2 as imageio
        writer = imageio.get_writer(video, fps=35, codec="libx264", quality=7, macro_block_size=8)
    rng = random.Random(seed)
    for ep in range(episodes):
        g = make_game(scenario, seed + ep)
        g.new_episode()
        while not g.is_episode_finished():
            s = g.get_state()
            text, b = observe(s, g, sc)
            rule = scripted(sc, b)
            probs = {}
            if policy == "random":
                choice = {q: rng.choice(list(v["criteria"].keys()) if v["type"] == "choice" else ["true", "false"])
                          for q, v in qs.items()}
            elif policy == "scripted":
                choice = rule
            else:
                ans, dt = client.systemone(text, qs)
                lat.append(dt)
                probs = {q: answer_probs(v, ans["answers"][q]) for q, v in qs.items()}
                choice = {q: argmax(p) for q, p in probs.items()}
            agree.append(all(choice[q] == rule[q] for q in qs))
            for q in qs:
                agree_q[q].append(choice[q] == rule[q])
            ndec += 1
            if writer:  # hold the decision for `skip` tics one tic at a time, so the video is every game frame
                pr = "  ".join(f"{q}={choice[q]}" + (f" ({probs[q][choice[q]]:.2f})" if probs else "") for q in qs)
                lines = [f"{policy} on the Neural Engine | {pr}" + (f" | {1000 * lat[-1]:.0f} ms" if lat else ""),
                         text[:100], text[100:200]]
                g.set_action(to_buttons(g, sc, choice))
                for _ in range(skip):
                    if g.is_episode_finished():
                        break
                    writer.append_data(overlay(g.get_state().screen_buffer, lines))
                    g.advance_action(1)
            else:
                g.make_action(to_buttons(g, sc, choice), skip)
        gv = {str(v).split(".")[-1]: x for v, x in zip(g.get_available_game_variables(), g.get_state().game_variables)} if g.get_state() else {}
        out.append({"reward": g.get_total_reward(), "tics": g.get_episode_time(),
                    "kills": g.get_game_variable(vzd.GameVariable.KILLCOUNT), "dead": g.is_player_dead()})
        g.close()
    if writer:
        writer.close()
    lat.sort()
    return {
        "scenario": scenario, "policy": policy, "episodes": episodes, "skip": skip,
        "reward_mean": round(st.mean(o["reward"] for o in out), 2),
        "reward_sd": round(st.pstdev(o["reward"] for o in out), 2),
        "kills_mean": round(st.mean(o["kills"] for o in out), 2),
        "survival_s_mean": round(st.mean(o["tics"] for o in out) / 35, 2),
        "deaths": sum(o["dead"] for o in out),
        "decisions": ndec, "agreement_with_rule": round(sum(agree) / len(agree), 4),
        "agreement_by_question": {q: round(sum(v) / len(v), 4) for q, v in agree_q.items()},
        "latency_ms_p50": round(1000 * lat[len(lat) // 2], 1) if lat else None,
        "latency_ms_p95": round(1000 * lat[int(len(lat) * 0.95)], 1) if lat else None,
        "episodes_detail": out,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="defend_the_center", help="name or 'all'")
    ap.add_argument("--policy", default="laya", help="random, scripted, laya, laya-typed-decisions or 'all'")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--skip", type=int, default=4, help="tics per decision (4 = 8.75 decisions per game second)")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--base-url", default="http://127.0.0.1:8008")
    ap.add_argument("--video", default=None)
    ap.add_argument("--out", default=str(ROOT / "results" / "doom"))
    args = ap.parse_args()
    scenarios = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    policies = ["random", "scripted", "laya", "laya-typed-decisions"] if args.policy == "all" else args.policy.split(",")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    for s in scenarios:
        for p in policies:
            client = DecisionClient(args.base_url, p) if p.startswith("laya") else None
            t0 = time.time()
            r = play(s, p, args.episodes, args.skip, args.seed, client, args.video)
            r["wall_s"] = round(time.time() - t0, 1)
            (Path(args.out) / f"{s}_{p}_skip{args.skip}.json").write_text(json.dumps(r, indent=1))
            print(json.dumps({k: v for k, v in r.items() if k != "episodes_detail"}), flush=True)


if __name__ == "__main__":
    main()
