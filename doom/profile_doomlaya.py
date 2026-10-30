"""Profile doomLaya decisions in-process by replaying recorded request packets.

Run from ext/doomLaya-dev with its venv:

    .venv/bin/python ../../doom/profile_doomlaya.py --encoder ane --decisions RUN/decisions.jsonl \\
        --n 40 [--cprofile out.prof] -- $(cat ../../doom/doomlaya_args.txt)

The server is built exactly as `serve_doom_laya.py` builds it; `uvicorn.run` is intercepted and
the /predict endpoint is called directly, so HTTP is excluded from the timings.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ane_doomlaya  # noqa: E402


def build_app(encoder, compute_units, serve_args, heads="cpu", skip_forced=False, guard=None, memory=None):
    import uvicorn
    captured = {}
    uvicorn.run = lambda app, **kw: captured.setdefault("app", app)
    ane_doomlaya.install(encoder, compute_units=compute_units, heads=heads, skip_forced=skip_forced, residual_guard=guard, memory_path=memory)
    sys.argv = ["serve_doom_laya.py"] + serve_args
    import runpy
    runpy.run_path("serve_doom_laya.py", run_name="__main__")
    route = next(r for r in captured["app"].routes if getattr(r, "path", None) == "/predict")
    request_type = next(iter(route.endpoint.__annotations__.values()))
    return route.endpoint, request_type


def packets(path, n):
    rows = [json.loads(line) for line in open(path)]
    pick = rows[200:200 + n // 2] + rows[1500:1500 + n // 2]  # before and during the MAP03 switch loop
    return [r["packet"] for r in pick]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", choices=["torch", "ane"], default="ane")
    ap.add_argument("--compute-units", default="CPU_AND_NE")
    ap.add_argument("--heads", choices=["cpu", "ane"], default="cpu")
    ap.add_argument("--skip-forced", action="store_true")
    ap.add_argument("--residual-guard", nargs="*", default=[], metavar="QUESTION=TAU")
    ap.add_argument("--command-memory")
    ap.add_argument("--decisions", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--cprofile")
    ap.add_argument("--answers-out", help="write the answers as jsonl for parity checks")
    args, serve_args = ap.parse_known_args()
    serve_args = [a for a in serve_args if a != "--"]
    predict, Request = build_app(args.encoder, args.compute_units, serve_args, args.heads, args.skip_forced,
                                 {q: float(t) for q, t in (g.split("=") for g in args.residual_guard)}, args.command_memory)
    work = packets(args.decisions, args.n)
    for p in work[:3]:
        predict(Request(state=p["state"], questions=p["questions"], model="doom-adapted"))
    stats = getattr(ane_doomlaya, "_ANE_STATS", None) or {}
    before = dict(stats)
    prof = None
    if args.cprofile:
        import cProfile
        prof = cProfile.Profile()
        prof.enable()
    lat, outs = [], []
    for p in work:
        t = time.perf_counter()
        out = predict(Request(state=p["state"], questions=p["questions"], model="doom-adapted"))
        lat.append((time.perf_counter() - t) * 1000)
        outs.append({k: v.get("choice") for k, v in out["answers"].items()})
    if prof:
        prof.disable()
        prof.dump_stats(args.cprofile)
    stats = getattr(ane_doomlaya, "_ANE_STATS", None) or {}
    delta = {k: round((stats[k] - before.get(k, 0)) / len(work), 1) for k in stats}
    print("PROFILE", json.dumps(dict(encoder=args.encoder, heads=args.heads, skip_forced=args.skip_forced, n=len(work), p50=round(statistics.median(lat)),
                                     mean=round(statistics.mean(lat)), per_decision=delta)), flush=True)
    if args.answers_out:
        with open(args.answers_out, "w") as f:
            for o in outs:
                f.write(json.dumps(o) + "\n")


if __name__ == "__main__":
    main()
