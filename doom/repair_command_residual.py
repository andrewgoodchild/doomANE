"""Repair doomLaya's command residual on MAP03-style key states, with short memory.

    cd ext/doomLaya && .venv/bin/python ../../doom/repair_command_residual.py \\
        --retain runs/<MAP01 run> --retain runs/<MAP02 run> \\
        --repair-run runs/<failed MAP03 run> --repair-after 296 --output ../../results/doom_heads/command-memory-v1

Data come only from recorded runs: every decision stores the request packet, Laya's own
command scores (`numeric_residual.semantic_answer`) and the residual's final choice, so no
model pass is needed.

* Repair rows: decisions in the repair run after --repair-after seconds where a key a route
  needs is reachable (`command_memory.missing_key_reachable`), no enemy is visible and `pickup`
  is offered. Label: pickup. The last 30 % by time are held out.
* Retention rows: every decision of the --retain runs, and the repair run before
  --repair-after. Target: the parent residual's own logits, recomputed and checked against the
  recorded choice.

The model is `command_memory.MemoryResidual` (parent network plus five zero-initialised memory
columns); every weight trains. The loss follows doomLaya's `training/finetune_numeric_retention.py`:
20 x cross-entropy on repair rows, squared logit drift plus a 100-weighted margin hinge on
retention rows. The saved epoch is the best held-out repair rate among epochs that keep every
retention decision. CPU only.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from command_memory import NAMES, Memory, make_memory_residual, missing_key_reachable  # noqa: E402

PARENT = Path("checkpoints/laya-doom-v031/laya-v031-command-key-retained-v1")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows_from_run(run, spec, until=None, after=None):
    """Projected features, memory, semantic logits and recorded choice for each decision."""
    from doomlib.command_facts import stable_command_facts_input
    from doomlib.numeric_command import ACTIONS, features, semantic_logits
    memory = Memory()
    out = []
    for d in map(json.loads, (run / "decisions.jsonl").open()):
        packet = d["packet"]
        raw = packet["state"]
        mem = memory.features(raw)
        answer = d["answers"]["command"]
        answer = answer.get("look_gate", {}).get("base_answer", answer)
        nr = answer.get("numeric_residual")
        if not nr:
            continue
        q = copy.deepcopy(packet["questions"]["command"])
        q.pop("look_observation", None)
        q["criteria"].pop("look_back", None)
        state, q = stable_command_facts_input(raw, q)
        t = d["game_seconds"]
        if (until is not None and t >= until) or (after is not None and t < after):
            continue
        out.append(dict(run=run.name, t=t, x=features(dict(state=state, question=q), spec["item_names"]), mem=mem,
                        b=semantic_logits(nr["semantic_answer"], spec["probability_floor"]),
                        mask=[a in q["criteria"] for a in ACTIONS], recorded=answer["choice"],
                        key_needed=missing_key_reachable(raw), no_enemy="No enemies visible" in raw))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--retain", type=Path, action="append", required=True)
    ap.add_argument("--repair-run", type=Path, required=True)
    ap.add_argument("--repair-after", type=float, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    a = ap.parse_args()
    import torch
    from safetensors.torch import load_file, save_file
    from doomlib.numeric_command import ACTIONS, make_residual_model
    torch.set_num_threads(2)
    torch.manual_seed(771)
    spec = json.loads((PARENT / "numeric-residual.json").read_text())
    parent = make_residual_model(spec)
    parent.load_state_dict(load_file(str(PARENT / "numeric-residual.safetensors")))
    parent.eval()

    retain = [r for run in a.retain for r in rows_from_run(run, spec)]
    retain += rows_from_run(a.repair_run, spec, until=a.repair_after)
    late = rows_from_run(a.repair_run, spec, after=a.repair_after)
    pick = ACTIONS.index("pickup")
    repair = [r for r in late if r["key_needed"] and r["no_enemy"] and r["mask"][pick]]
    cut = int(len(repair) * 0.7)
    groups = {"retain": retain, "repair_train": repair[:cut], "repair_valid": repair[cut:]}
    t = lambda rows, k, dt=torch.float32: torch.tensor([r[k] for r in rows], dtype=dt)
    data = {}
    for name, rows in groups.items():
        x, b, mem, mask = t(rows, "x"), t(rows, "b"), t(rows, "mem"), t(rows, "mask", torch.bool)
        with torch.no_grad():
            old = parent(x, b)
        old_choice = old.masked_fill(~mask, -1e4).argmax(-1)
        recorded = torch.tensor([ACTIONS.index(r["recorded"]) for r in rows])
        if not torch.equal(old_choice, recorded):
            raise SystemExit(f"{name}: parent replay disagrees with the recording on {(old_choice != recorded).sum()} rows")
        data[name] = dict(x=x, b=b, mem=mem, mask=mask, old=old, old_choice=old_choice, y=torch.full((len(rows),), pick))
    conflicts = sum(1 for r in retain if r["key_needed"] and r["no_enemy"] and r["mask"][pick] and r["recorded"] != "pickup")
    print(json.dumps({k: len(v) for k, v in groups.items()}), "retained key-needed rows not choosing pickup:", conflicts, flush=True)

    model = make_memory_residual(parent)
    opt = torch.optim.AdamW(model.parameters(), lr=a.learning_rate, weight_decay=0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs, eta_min=a.learning_rate / 100)
    train = {k: torch.cat([data["retain"][k], data["repair_train"][k]]) for k in data["retain"]}
    is_repair = torch.cat([torch.zeros(len(retain), dtype=torch.bool), torch.ones(cut, dtype=torch.bool)])

    def evaluate():
        res = {}
        with torch.no_grad():
            for name, d in data.items():
                choice = model(d["x"], d["b"], d["mem"]).masked_fill(~d["mask"], -1e4).argmax(-1)
                res[name] = dict(n=len(choice), same_as_parent=int((choice == d["old_choice"]).sum()),
                                 pickup=int((choice == pick).sum()))
        return res

    initial = evaluate()
    best, state, best_epoch, history = None, None, None, []
    start = time.monotonic()
    for epoch in range(a.epochs):
        model.train()
        for idx in torch.randperm(len(is_repair)).split(512):
            d = {k: v[idx] for k, v in train.items()}
            rep = is_repair[idx]
            z = model(d["x"], d["b"], d["mem"])
            ce = torch.nn.functional.cross_entropy(z.masked_fill(~d["mask"], -1e4), d["y"], reduction="none")
            drift = ((z - d["old"]).square() * d["mask"]).sum(-1) / d["mask"].sum(-1)
            others = d["mask"].clone().scatter_(1, d["old_choice"][:, None], False)
            old_margin = d["old"].gather(1, d["old_choice"][:, None]).squeeze(1) - d["old"].masked_fill(~others, -1e4).max(-1).values
            margin = (z.masked_fill(~others, -1e4).max(-1).values - z.gather(1, d["old_choice"][:, None]).squeeze(1)
                      + (old_margin / 2).clamp(0, .1)).clamp_min(0)
            loss = (20 * ce * rep + (drift + 100 * margin) * ~rep).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        if (epoch + 1) % 5:
            continue
        model.eval()
        m = evaluate()
        eligible = m["retain"]["same_as_parent"] == m["retain"]["n"]
        score = (m["repair_valid"]["pickup"], m["repair_train"]["pickup"])
        history.append(dict(epoch=epoch + 1, eligible=eligible, metrics=m))
        if eligible and (best is None or score > best):
            best, state, best_epoch = score, copy.deepcopy(model.state_dict()), epoch + 1
        if (epoch + 1) % 50 == 0:
            print(json.dumps(history[-1]), "best", best, flush=True)
    if state is None or best[0] == 0:
        raise SystemExit("no epoch repaired held-out rows while keeping every retained decision")
    model.load_state_dict(state)
    model.eval()
    a.output.mkdir(parents=True, exist_ok=False)
    save_file({k: v.contiguous() for k, v in state.items()}, str(a.output / "memory-residual.safetensors"))
    meta = dict(format="doomlaya-command-memory-residual-v1", parent=PARENT.name,
                parent_weights_sha256=digest(PARENT / "numeric-residual.safetensors"),
                parent_spec_sha256=digest(PARENT / "numeric-residual.json"), memory_features=list(NAMES),
                window=Memory().history.maxlen, best_epoch=best_epoch, epochs=a.epochs, learning_rate=a.learning_rate,
                retain_runs={str(r): digest(r / "decisions.jsonl") for r in a.retain},
                repair_run={str(a.repair_run): digest(a.repair_run / "decisions.jsonl")}, repair_after=a.repair_after,
                trainer_sha256=digest(__file__),
                note="Same-map repair: held-out rows come from the same MAP03 run, so this is not generalisation.")
    (a.output / "meta.json").write_text(json.dumps(meta, indent=1))
    (a.output / "metrics.json").write_text(json.dumps(dict(initial=initial, final=evaluate(), best_epoch=best_epoch,
                                                            seconds=round(time.monotonic() - start, 1), history=history), indent=1))
    print("SAVED", a.output, json.dumps(evaluate()), flush=True)


if __name__ == "__main__":
    main()
