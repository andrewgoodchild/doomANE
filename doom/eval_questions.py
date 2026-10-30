"""Zero-shot test of new System 1 questions on recorded doomLaya games.

Each question is written from the recorded packet state plus facts taken from per-tick telemetry,
asked through a doomLaya /predict server, and scored against a label that code or the game's
outcome supplies:

  progress   noul  "The player is making progress toward the current command's target."
             label: the target got > 1 m closer in the last 5 s (yes), or neither the target
             distance nor the player moved more than 0.3 m / 1 m (no). Lifts being boarded or
             ridden are skipped.
  retreat    noul  "Pull back now to heal or take cover."
             label: the player died within the next 6 s (yes) or survived the next 20 s (no),
             at decisions with an enemy visible.
  keep       noul  "Keep going with the current command instead of switching."
             label: decisions where Laya abandoned an executing route command. Yes if the bot
             then looped (the abandoned target was chosen again within 30 s and not reached),
             no if the switch paid off within 10 s (an item, key or door gained) and the old
             target was not chosen again.

Run from ext/doomLaya-dev:
  .venv/bin/python ../../doom/eval_questions.py --endpoint http://127.0.0.1:8004/predict --per-class 120
Writes results/doom/question_eval.json.
"""
import argparse
import glob
import json
import math
import random
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
QUESTIONS = {
    "progress": dict(type="noul", instructions="The player is making progress toward the current command's target."),
    "retreat": dict(type="noul", instructions="Pull back now to heal or take cover instead of continuing the fight."),
    "keep": dict(type="noul", instructions="Keep going with the current command instead of switching to the new one."),
}
# The same questions as two-option choices (doomLaya's heads were trained on choice questions only)
CHOICES = {
    "progress": dict(type="choice", instructions="Is the player getting closer to the current command's target?",
                     criteria={"yes": "Yes: the target is getting closer.", "no": "No: the player is not getting closer."}),
    "retreat": dict(type="choice", instructions="Choose how to handle this fight.",
                    criteria={"yes": "Pull back now to heal or take cover.", "no": "Keep fighting here."}),
    "keep": dict(type="choice", instructions="Choose between the current command and the proposed new command.",
                 criteria={"yes": "Keep going with the current command.", "no": "Switch to the proposed new command."}),
}
ROUTE = ("pickup", "use_switch", "exit", "open_door")


def load(run):
    telemetry = {}
    for line in open(run + "/telemetry.jsonl"):
        t = json.loads(line)
        telemetry[t["tick"]] = t
    decisions = [json.loads(line) for line in open(run + "/decisions.jsonl")]
    events = [json.loads(line) for line in open(run + "/events.jsonl")]
    return telemetry, decisions, events


def target_distance(t):
    """Distance in metres from the player to the executing command's target, if observable."""
    ex = t["execution"]
    action, target = ex.get("action"), ex.get("target_id")
    if action == "use_switch":
        m = next((m for m in t["switches"] if m["id"] == target), None)
        if m is None or m.get("phase") in ("board", "ride"):
            return None
        return m["distance"]
    if action == "pickup":
        i = next((i for i in t["items"] if str(i["id"]) == str(target)), None)
        return i["distance"] if i else None
    if action == "exit" and t.get("mission") and t["mission"].get("exit"):
        x, y = t["mission"]["exit"]["approach"]
        return math.dist((t["x"], t["y"]), (x, y)) / 32
    if action == "open_door" and t.get("door") and t["door"]["id"] == target:
        return t["door"]["distance"]
    return None


def progress_cases(run):
    telemetry, decisions, _ = load(run)
    for d in decisions:
        tick = d["tick"]
        now, before = telemetry.get(tick), telemetry.get(tick - 175)
        if not now or not before or now["execution"].get("status") != "executing":
            continue
        ex, ex0 = now["execution"], before["execution"]
        if ex.get("action") not in ROUTE or (ex.get("action"), ex.get("target_id")) != (ex0.get("action"), ex0.get("target_id")):
            continue
        d_now, d_before = target_distance(now), target_distance(before)
        if d_now is None or d_before is None:
            continue
        moved = math.dist((now["x"], now["y"]), (before["x"], before["y"])) / 32
        closer = d_before - d_now
        if closer > 1.0:
            label = True
        elif abs(closer) < 0.3 and moved < 1.0:
            label = False
        else:
            continue
        since = tick
        while telemetry.get(since - 35) and (telemetry[since - 35]["execution"].get("action"), telemetry[since - 35]["execution"].get("target_id")) == (ex.get("action"), ex.get("target_id")):
            since -= 35
        facts = (f"Current command: {ex.get('detail', ex.get('action'))}; executing for {(tick - since) / 35:.0f} s.\n"
                 f"Distance to its target now {d_now:.1f} m; 5 s ago {d_before:.1f} m.\n"
                 f"The player moved {moved:.1f} m in the last 5 s.")
        yield dict(run=run, tick=tick, label=label, state=d["packet"]["state"] + "\n" + facts)


def retreat_cases(run):
    telemetry, decisions, events = load(run)
    deaths = [round(e["game_seconds"] * 35) for e in events if e["event"] == "death"]
    end = max(telemetry)
    for d in decisions:
        tick = d["tick"]
        t = telemetry.get(tick)
        if not t or not any(e.get("visible", True) for e in t["enemies"]):
            continue
        nxt = min((x for x in deaths if x > tick), default=None)
        if nxt is not None and nxt - tick <= 210:
            label = True
        elif (nxt is None or nxt - tick > 700) and end - tick > 700:
            label = False
        else:
            continue
        visible = [e for e in t["enemies"] if e.get("visible", True)]
        health = [i for i in t["items"] if i["category"] == "Health"]
        weapon = t["inventory"][str(t["weapon_slot"])] if str(t.get("weapon_slot")) in t["inventory"] else {}
        before = telemetry.get(tick - 105)
        lost = (before["hp"] - t["hp"]) if before else 0
        facts = (f"Health {t['hp']:.0f}, armor {t['armor']:.0f}; lost {max(lost, 0):.0f} health in the last 3 s.\n"
                 f"Visible enemies: {len(visible)}; nearest {min(e['distance'] for e in visible):.1f} m "
                 f"({', '.join(sorted({e['name'] for e in visible}))}).\n"
                 f"Current weapon ammo: {weapon.get('ammo', 0)}.\n"
                 f"Nearest health item: {min((i['distance'] for i in health), default=float('inf')):.1f} m.")
        yield dict(run=run, tick=tick, label=label, state=d["packet"]["state"] + "\n" + facts.replace("inf m", "none known"))


def keep_cases(run):
    telemetry, decisions, events = load(run)
    gains = [round(e["game_seconds"] * 35) for e in events if e["event"] in ("pickups", "door_opened", "key_pickup", "resource_gain")]
    for i, d in enumerate(decisions):
        t = telemetry.get(d["tick"])
        if not t:
            continue
        ex = t["execution"]
        new = d["answers"]["command"]["choice"]
        if ex.get("status") != "executing" or ex.get("action") not in ROUTE or new == ex.get("action") or new not in ROUTE:
            continue
        old = (ex.get("action"), ex.get("target_id"))
        later = [x for x in decisions[i + 1:] if x["tick"] - d["tick"] <= 1050]
        came_back = any(telemetry.get(x["tick"], {}).get("execution", {}).get("action") == old[0]
                        and telemetry.get(x["tick"], {}).get("execution", {}).get("target_id") == old[1] for x in later[4:])
        finished = any(0 < g - d["tick"] <= 350 for g in gains)
        if came_back and not finished:
            label = True
        elif finished and not came_back:
            label = False
        else:
            continue
        d_now = target_distance(t)
        facts = (f"Current command: {ex.get('detail', ex.get('action'))}; status executing"
                 + (f"; target {d_now:.1f} m away" if d_now is not None else "") + ".\n"
                 f"Proposed new command: {new}.")
        yield dict(run=run, tick=d["tick"], label=label, state=d["packet"]["state"] + "\n" + facts)


def ask(endpoint, name, state, form="noul"):
    question = QUESTIONS[name] if form == "noul" else CHOICES[name]
    body = json.dumps({"state": state, "questions": {name: question}, "model": "doom-adapted"}).encode()
    request = urllib.request.Request(endpoint, body, {"content-type": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        answer = json.load(response)["answers"][name]
    return answer["noul"] if form == "noul" else answer["probabilities"]["yes"]


def auroc(scores, labels):
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--endpoint", default="http://127.0.0.1:8004/predict")
    p.add_argument("--per-class", type=int, default=120)
    p.add_argument("--questions", nargs="+", default=list(QUESTIONS))
    p.add_argument("--form", choices=["noul", "choice"], default="noul")
    a = p.parse_args()
    random.seed(771)
    runs = sorted(r for r in glob.glob("runs/2026101*_repeat-*") if Path(r, "events.jsonl").exists())
    builders = dict(progress=progress_cases, retreat=retreat_cases, keep=keep_cases)
    report = {}
    for name in a.questions:
        cases = [c for r in runs for c in builders[name](r)]
        pos = [c for c in cases if c["label"]]
        neg = [c for c in cases if not c["label"]]
        sample = random.sample(pos, min(a.per_class, len(pos))) + random.sample(neg, min(a.per_class, len(neg)))
        scores = [ask(a.endpoint, name, c["state"], a.form) for c in sample]
        labels = [c["label"] for c in sample]
        accuracy = sum((s >= 0.5) == l for s, l in zip(scores, labels)) / len(sample)
        report[name] = dict(form=a.form, question=QUESTIONS[name] if a.form == "noul" else CHOICES[name], available=dict(yes=len(pos), no=len(neg)), asked=len(sample),
                            accuracy=round(accuracy, 3), auroc=auroc(scores, labels),
                            mean_yes_when_yes=round(sum(s for s, l in zip(scores, labels) if l) / max(1, sum(labels)), 3),
                            mean_yes_when_no=round(sum(s for s, l in zip(scores, labels) if not l) / max(1, len(labels) - sum(labels)), 3),
                            examples=[dict(run=c["run"], tick=c["tick"], label=c["label"], yes=s) for c, s in zip(sample, scores)])
        print(name, {k: v for k, v in report[name].items() if k not in ("examples", "question")}, flush=True)
    out = REPO / f"results/doom/question_eval_{a.form}.json"
    out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
