"""Distil doomLaya's command residual for pixel perception from its own engine-perception choices.

Pixel games recorded with agent.py --shadow-engine-packet store, with every decision, the command question
built from engine perception at the same moment (`shadow_packet`). Two steps, run from ext/doomLaya-dev:

  1. teacher: each recorded engine question goes to a running model server with the served stack (the
     same doomlaya_args.txt as the games); the answers are saved next to the run (shadow-teacher.jsonl).
       .venv/bin/python ../../doom/distill_command.py teacher runs/<run>... --endpoint http://127.0.0.1:8013/predict
  2. train: a copy of the parent residual learns, from the pixel question (projected features and the
     pixel semantic logits recorded in play), the teacher's distribution over the actions offered in both
     questions. Loss: KL(teacher || student) + DRIFT x squared logit drift from the parent. Rows offering
     fewer than two common actions are dropped. The kept epoch has the best validation agreement.
       .venv/bin/python ../../doom/distill_command.py train --train runs/<run>... --valid runs/<run>... \\
           --output ../../results/doom_heads/laya-pixel-command-distill-v1

Labels are Laya's own answers (no hosted model, no heuristic). The look gate's look_back is not a
residual action: the base answer under the look gate is used and look_back is left out, as in
repair_command_residual.py. CPU only.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
PARENT = Path("checkpoints/laya-doom-v031/laya-v031-command-key-retained-v1")
DRIFT = 0.1


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def base(answer):
    return answer.get("look_gate", {}).get("base_answer", answer)


def residual_question(question):
    q = copy.deepcopy(question)
    q.pop("look_observation", None)
    q["criteria"].pop("look_back", None)
    return q


def teacher(a):
    import requests
    session = requests.Session()
    session.trust_env = False
    for run in a.runs:
        out = run / "shadow-teacher.jsonl"
        if out.exists():
            print("exists", out, flush=True)
            continue
        rows = []
        for i, d in enumerate(map(json.loads, (run / "decisions.jsonl").open())):
            shadow = d.get("shadow_packet")
            if not shadow:
                continue
            r = session.post(a.endpoint, json={"state": shadow["state"], "questions": shadow["questions"], "model": "doom-adapted"},
                             timeout=(3, 30))
            r.raise_for_status()
            result = r.json()
            rows.append(dict(index=i, game_seconds=d["game_seconds"], answer=result["answers"]["command"],
                             routing=result.get("routing")))
        out.write_text("".join(json.dumps(row) + "\n" for row in rows))
        print(run.name, len(rows), "teacher answers", flush=True)


def rows_from_run(run, spec):
    """Pixel features and logits, the teacher's distribution and both offered action sets, per decision."""
    from doomlib.command_facts import stable_command_facts_input
    from doomlib.numeric_command import ACTIONS, features, semantic_logits
    answers = {r["index"]: r["answer"] for r in map(json.loads, (run / "shadow-teacher.jsonl").open())}
    out = []
    for i, d in enumerate(map(json.loads, (run / "decisions.jsonl").open())):
        if i not in answers:
            continue
        pixel, engine = base(d["answers"]["command"]), base(answers[i])
        if not pixel.get("numeric_residual") or not engine.get("numeric_residual"):
            continue  # a one-option question answered without the model
        qp = residual_question(d["packet"]["questions"]["command"])
        qe = residual_question(d["shadow_packet"]["questions"]["command"])
        state_p, qp = stable_command_facts_input(d["packet"]["state"], qp)
        state_e, qe = stable_command_facts_input(d["shadow_packet"]["state"], qe)
        both = [act in qp["criteria"] and act in qe["criteria"] for act in ACTIONS]
        teacher_p = [float(engine["probabilities"].get(act, 0)) if ok else 0.0 for act, ok in zip(ACTIONS, both)]
        if sum(both) < 2 or sum(teacher_p) <= 0:
            continue
        out.append(dict(run=run.name, t=d["game_seconds"],
                        x=features(dict(state=state_p, question=qp), spec["item_names"]),
                        b=semantic_logits(pixel["numeric_residual"]["semantic_answer"], spec["probability_floor"]),
                        mask=[act in qp["criteria"] for act in ACTIONS], both=both,
                        teacher=[p / sum(teacher_p) for p in teacher_p],
                        xe=features(dict(state=state_e, question=qe), spec["item_names"]),
                        be=semantic_logits(engine["numeric_residual"]["semantic_answer"], spec["probability_floor"]),
                        mask_e=[act in qe["criteria"] for act in ACTIONS],
                        recorded=pixel["choice"], teacher_choice=engine["choice"]))
    return out


def train(a):
    import torch
    from safetensors.torch import load_file, save_file
    from doomlib.numeric_command import ACTIONS, make_residual_model
    torch.set_num_threads(2)
    torch.manual_seed(1106)
    spec = json.loads((PARENT / "numeric-residual.json").read_text())
    parent = make_residual_model(spec)
    parent.load_state_dict(load_file(str(PARENT / "numeric-residual.safetensors")))
    parent.eval()
    groups = {"train": [r for run in a.train for r in rows_from_run(run, spec)],
              "valid": [r for run in a.valid for r in rows_from_run(run, spec)]}
    t = lambda rows, k, dt=torch.float32: torch.tensor([r[k] for r in rows], dtype=dt)
    data = {}
    for name, rows in groups.items():
        x, b, mask, both = t(rows, "x"), t(rows, "b"), t(rows, "mask", torch.bool), t(rows, "both", torch.bool)
        with torch.no_grad():
            old = parent(x, b)
            # The recordings must replay: the parent on the pixel inputs gives the pixel choice, and on the
            # engine inputs the teacher's choice.
            for side, xs, bs, ms, key in (("pixel", x, b, mask, "recorded"),
                                          ("engine", t(rows, "xe"), t(rows, "be"), t(rows, "mask_e", torch.bool), "teacher_choice")):
                replay = parent(xs, bs).masked_fill(~ms, -1e4).argmax(-1)
                recorded = torch.tensor([ACTIONS.index(r[key]) for r in rows])
                if not torch.equal(replay, recorded):
                    raise SystemExit(f"{name}: parent replay of the {side} side disagrees on {(replay != recorded).sum()} rows")
        data[name] = dict(x=x, b=b, mask=mask, both=both, old=old, teacher=t(rows, "teacher"),
                          teacher_choice=t(rows, "teacher").argmax(-1), recorded=torch.tensor([ACTIONS.index(r["recorded"]) for r in rows]))
    print(json.dumps({k: len(v) for k, v in groups.items()}), flush=True)

    model = copy.deepcopy(parent)
    for p in model.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.learning_rate, weight_decay=0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs, eta_min=a.learning_rate / 100)
    tr = data["train"]

    def evaluate(net):
        res = {}
        with torch.no_grad():
            for name, d in data.items():
                choice = net(d["x"], d["b"]).masked_fill(~d["mask"], -1e4).argmax(-1)
                agree = choice == d["teacher_choice"]
                res[name] = dict(n=len(choice), agree=round(float(agree.float().mean()), 4),
                                 by_teacher_action={act: [int(agree[d["teacher_choice"] == i].sum()), int((d["teacher_choice"] == i).sum())]
                                                    for i, act in enumerate(ACTIONS) if (d["teacher_choice"] == i).any()},
                                 choices={act: int((choice == i).sum()) for i, act in enumerate(ACTIONS) if (choice == i).any()},
                                 attack_not_teacher=int(((choice == ACTIONS.index("attack")) & ~agree).sum()),
                                 pickup_not_teacher=int(((choice == ACTIONS.index("pickup")) & ~agree).sum()))
        return res

    initial = evaluate(parent)
    print("parent", json.dumps({k: v["agree"] for k, v in initial.items()}), flush=True)
    best, state, best_epoch, history = None, None, None, []
    start = time.monotonic()
    for epoch in range(a.epochs):
        model.train()
        for idx in torch.randperm(len(tr["x"])).split(256):
            d = {k: v[idx] for k, v in tr.items()}
            z = model(d["x"], d["b"])
            logp = torch.log_softmax(z.masked_fill(~d["both"], -1e4), -1)
            kl = (d["teacher"] * (torch.log(d["teacher"].clamp_min(1e-8)) - logp)).masked_fill(~d["both"], 0).sum(-1)
            drift = ((z - d["old"]).square() * d["mask"]).sum(-1) / d["mask"].sum(-1)
            loss = (kl + a.drift * drift).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        if (epoch + 1) % 5:
            continue
        model.eval()
        m = evaluate(model)
        history.append(dict(epoch=epoch + 1, train=m["train"]["agree"], valid=m["valid"]["agree"]))
        if best is None or m["valid"]["agree"] > best:
            best, state, best_epoch = m["valid"]["agree"], copy.deepcopy(model.state_dict()), epoch + 1
        if (epoch + 1) % 50 == 0:
            print(json.dumps(history[-1]), "best", best, flush=True)
    model.load_state_dict(state)
    model.eval()
    final = evaluate(model)
    a.output.mkdir(parents=True, exist_ok=False)
    save_file({k: v.contiguous() for k, v in state.items()}, str(a.output / "numeric-residual.safetensors"))
    meta = dict(format="doomlaya-command-residual-distilled-v1", name=a.output.name, parent=PARENT.name,
                parent_weights_sha256=digest(PARENT / "numeric-residual.safetensors"),
                parent_spec_sha256=digest(PARENT / "numeric-residual.json"), drift=a.drift,
                best_epoch=best_epoch, epochs=a.epochs, learning_rate=a.learning_rate,
                train_runs={str(r): digest(r / "decisions.jsonl") for r in a.train},
                valid_runs={str(r): digest(r / "decisions.jsonl") for r in a.valid},
                teacher_sha256={str(r): digest(r / "shadow-teacher.jsonl") for r in a.train + a.valid},
                trainer_sha256=digest(__file__),
                note="Teacher: the served command head on the engine-perception question of the same moment. Serve only with --pixel-perception.")
    (a.output / "meta.json").write_text(json.dumps(meta, indent=1))
    (a.output / "metrics.json").write_text(json.dumps(dict(parent=initial, student=final, best_epoch=best_epoch,
                                                            seconds=round(time.monotonic() - start, 1), history=history), indent=1))
    print("SAVED", a.output, "parent", json.dumps({k: v["agree"] for k, v in initial.items()}),
          "student", json.dumps({k: v["agree"] for k, v in final.items()}), "epoch", best_epoch, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="step", required=True)
    t = sub.add_parser("teacher")
    t.add_argument("runs", type=Path, nargs="+")
    t.add_argument("--endpoint", required=True)
    s = sub.add_parser("train")
    s.add_argument("--train", type=Path, nargs="+", required=True)
    s.add_argument("--valid", type=Path, nargs="+", required=True)
    s.add_argument("--output", type=Path, required=True)
    s.add_argument("--epochs", type=int, default=200)
    s.add_argument("--learning-rate", type=float, default=3e-4)
    s.add_argument("--drift", type=float, default=DRIFT, help="weight of the squared logit drift from the parent")
    a = ap.parse_args()
    teacher(a) if a.step == "teacher" else train(a)


if __name__ == "__main__":
    main()
