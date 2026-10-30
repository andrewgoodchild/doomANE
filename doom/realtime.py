"""Real-time check: the game runs at 35 tics per second (ViZDoom ASYNC_PLAYER) whether or not the
model keeps up; each decision is applied as soon as it lands.

    uv run python doom/realtime.py --scenario defend_the_center --policy laya
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import vizdoom as vzd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.client import DecisionClient, answer_probs, argmax  # noqa: E402
from doom.play import SCENARIOS, observe, questions, to_buttons  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="defend_the_center")
    ap.add_argument("--policy", default="laya")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--base-url", default="http://127.0.0.1:8008")
    args = ap.parse_args()
    sc = SCENARIOS[args.scenario]
    qs = questions(sc)
    c = DecisionClient(args.base_url, args.policy)
    eps = []
    for ep in range(args.episodes):
        g = vzd.DoomGame()
        g.load_config(os.path.join(vzd.scenarios_path, args.scenario + ".cfg"))
        g.set_window_visible(False)
        g.set_labels_buffer_enabled(True)
        for v in (vzd.GameVariable.POSITION_X, vzd.GameVariable.POSITION_Y, vzd.GameVariable.KILLCOUNT):
            g.add_available_game_variable(v)
        g.set_mode(vzd.Mode.ASYNC_PLAYER)
        g.set_ticrate(35)
        g.set_seed(args.seed + ep)
        g.init()
        g.new_episode()
        n, lat, t0 = 0, [], time.perf_counter()
        while not g.is_episode_finished():
            s = g.get_state()
            if s is None:
                break
            text, _ = observe(s, g, sc)
            a, dt = c.systemone(text, qs)
            lat.append(dt)
            g.make_action(to_buttons(g, sc, {q: argmax(answer_probs(v, a["answers"][q])) for q, v in qs.items()}), 1)
            n += 1
        wall = time.perf_counter() - t0
        lat.sort()
        eps.append({"decisions": n, "wall_s": round(wall, 2), "game_s": round(g.get_episode_time() / 35, 2),
                    "decisions_per_s": round(n / wall, 1), "latency_ms_p50": round(1000 * lat[len(lat) // 2], 1),
                    "latency_ms_p95": round(1000 * lat[int(0.95 * len(lat))], 1), "reward": g.get_total_reward(),
                    "kills": g.get_game_variable(vzd.GameVariable.KILLCOUNT)})
        g.close()
    out = {"scenario": args.scenario, "policy": args.policy, "mode": "ASYNC_PLAYER 35 tics/s", "episodes": eps}
    (ROOT / "results" / "doom" / f"realtime_{args.scenario}_{args.policy}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out))


if __name__ == "__main__":
    main()
