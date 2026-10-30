"""Serve azalio/doomLaya v0.3.1 with its shared Laya encoder on the Apple Neural Engine.

Run with doomLaya's own environment from its repo root (it pins laya 0.3.4 / transformers 4.57):

    cd ext/doomLaya-dev
    .venv/bin/python ../../doom/ane_doomlaya.py --encoder ane  [serve_doom_laya.py arguments ...]
    .venv/bin/python ../../doom/ane_doomlaya.py --encoder record --record-file lengths.jsonl [...]
    .venv/bin/python ../../doom/ane_doomlaya.py --encoder torch [...]      # unmodified reference

doomLaya's eight question heads all wrap one frozen encoder object (`doomlib/question_heads.py`,
`load_head` reuses `base.model.encoder`), and Laya 0.3.4's DecisionModel only calls
`self.encoder(input_ids=..., attention_mask=...).last_hidden_state`. So replacing that one module
moves every encoder pass to the Neural Engine; the small heads, the numeric residuals and all of
doomLaya's controller code run unchanged in PyTorch on the CPU.

ANE path: each row runs on the smallest fixed-length Core ML package that holds it
(`results/coreml/doomlaya_enc_L{L}_fp16.mlpackage`, exported by `doom/export_doom_encoder.py`);
longer rows fall back to the original PyTorch encoder and are counted.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import coreml_cache  # noqa: E402
PKG_DIR = ROOT / "results" / "coreml"
_STATS = Counter()  # process-wide counters for the patches below


def _rows(attention_mask):
    return [int(r.sum().item()) for r in attention_mask]


class RecordingEncoder:
    def __init__(self, orig, path):
        import torch
        self.torch = torch
        self.orig, self.path = orig, path

    def wrap(self):
        torch = self.torch
        rec = self

        class Rec(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.inner = rec.orig
                self.config = rec.orig.config

            def forward(self, input_ids, attention_mask, **kw):
                t0 = time.perf_counter()
                out = self.inner(input_ids=input_ids, attention_mask=attention_mask, **kw)
                with open(rec.path, "a") as f:
                    f.write(json.dumps({"batch": int(input_ids.shape[0]), "padded": int(input_ids.shape[1]),
                                        "lengths": _rows(attention_mask), "ms": round(1000 * (time.perf_counter() - t0), 1)}) + "\n")
                return out
        return Rec()


def ane_encoder(orig, pad_id, compute_units="CPU_AND_NE"):
    import coremltools as ct
    import numpy as np
    import torch

    pk = {}
    merged = PKG_DIR / "doomlaya_enc_fp16.mlpackage"  # doom/merge_doom_encoder.py: one weight copy, a function per length
    if merged.exists():
        functions = [f.name for f in ct.utils.load_spec(str(merged)).description.functions]
        pk = {int(name[1:]): (merged, name) for name in functions}
    else:
        for p in PKG_DIR.glob("doomlaya_enc_L*_fp16.mlpackage"):
            pk[int(p.name.split("_L")[1].split("_")[0])] = (p, None)
    if not pk:
        raise SystemExit("no doomlaya_enc packages in " + str(PKG_DIR))
    models, bufs = {}, {}
    for L in sorted(pk):
        t0 = time.perf_counter()
        path, function = pk[L]
        models[L] = coreml_cache.load(path, compute_units, function)
        # long-lived, exactly typed inputs filled in place: Core ML frees zero-copy inputs off the GIL
        # about a second after the last predict, which segfaults if they were the last reference
        bufs[L] = {"input_ids": np.zeros((1, L), np.int32), "attention_mask": np.zeros((1, L), np.int32)}
        print(f"[ane] loaded L={L} in {time.perf_counter() - t0:.1f}s", flush=True)
    stats = Counter()

    class ANE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = orig
            self.config = orig.config
            self.stats = stats
            self.lengths = sorted(models)
            self.outputs = {}

        def encode_row(self, ids, n, L):
            """Encode one contiguous row of n tokens at bucket L; returns the fp16 `hidden` [1,L,d].

            The output array is also kept in self.outputs[L] until the next call at L, so the
            buffer Core ML hands back is never the last reference when it is freed."""
            b = bufs[L]
            b["input_ids"].fill(pad_id)
            b["input_ids"][0, :n] = ids
            b["attention_mask"].fill(0)
            b["attention_mask"][0, :n] = 1
            t0 = time.perf_counter()
            out = models[L].predict(b)["hidden"]
            stats["ane_ms"] += (time.perf_counter() - t0) * 1000
            stats["tokens"] += n
            stats[f"L{L}"] += 1
            self.outputs[L] = out
            return out

        def forward(self, input_ids, attention_mask, **kw):
            t_start = time.perf_counter()
            B, T = input_ids.shape
            d = self.config.hidden_size
            dtype = next(orig.parameters()).dtype
            h = torch.zeros((B, T, d), dtype=dtype, device=input_ids.device)
            ids_c, am_c = input_ids.cpu(), attention_mask.cpu()
            for i in range(B):
                n = int(am_c[i].sum().item())
                contiguous = bool(am_c[i, :n].all().item())
                L = next((L for L in sorted(models) if n <= L), None)
                if L is None or not contiguous:
                    stats["fallback"] += 1
                    t0 = time.perf_counter()
                    h[i:i + 1] = orig(input_ids=input_ids[i:i + 1], attention_mask=attention_mask[i:i + 1]).last_hidden_state
                    stats["fallback_ms"] += (time.perf_counter() - t0) * 1000
                    continue
                hidden = self.encode_row(ids_c[i, :n].numpy(), n, L)
                h[i, :n] = torch.from_numpy(np.asarray(hidden, dtype=np.float32)[0, :n]).to(dtype)
            stats["encoder_ms"] += (time.perf_counter() - t_start) * 1000
            stats["forwards"] += 1
            stats["rows"] += B
            every = int(os.environ.get("ANE_STATS_EVERY", "0"))
            if every and stats["forwards"] % every == 0:
                print("[ane] stats:", json.dumps({k: round(v, 1) for k, v in stats.items()}), flush=True)
            return SimpleNamespace(last_hidden_state=h)
    return ANE()


def ane_heads(compute_units="CPU_AND_NE"):
    """Run each DecisionModel's head (2 transformer layers + scorer) on the Neural Engine too.

    Packages come from `doom/export_doom_heads.py` and are found by hashing the live model's
    head weights. The encoder output goes straight from the encoder package into the head
    package as fp16; only the option-marker logits and the position-0 vector come back to
    PyTorch, where Laya's own confidence features and act head run unchanged. A model with no
    matching package, or a row the encoder packages cannot hold, uses the original forward.
    """
    import coremltools as ct
    import numpy as np
    import torch
    from laya.common import DecisionModel
    from export_doom_heads import head_digest

    orig_forward = DecisionModel.forward
    loaded = {}  # digest -> {L: MLModel}
    mask_bufs, qtype_buf = {}, np.zeros((1, 1), np.int32)
    keep = {}

    def heads_for(model):
        if "_ane_head_digest" not in model.__dict__:
            named = {k: v for k, v in model.state_dict().items() if not k.startswith("encoder.")}
            model.__dict__["_ane_head_digest"] = head_digest(named)
        dig = model.__dict__["_ane_head_digest"]
        if dig not in loaded:
            path = PKG_DIR / f"doomlaya_head_{dig}_fp16.mlpackage"
            if not path.exists():
                print(f"[ane] no head package for {dig}; this head stays on the CPU", flush=True)
                loaded[dig] = None
            else:
                t0 = time.perf_counter()
                loaded[dig] = {L: coreml_cache.load(path, compute_units, f"L{L}") for L in model.encoder.lengths}
                print(f"[ane] loaded head {dig} in {time.perf_counter() - t0:.1f}s", flush=True)
        return loaded[dig]

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype, detach_encoder=False):
        enc = self.encoder
        heads = heads_for(self) if hasattr(enc, "encode_row") else None
        B, K = marker_pos.shape
        am_c = attention_mask.cpu()
        lens = [int(am_c[i].sum().item()) for i in range(B)]
        buckets = [next((L for L in enc.lengths if n <= L), None) for n in lens] if heads else []
        if not heads or None in buckets or any(not bool(am_c[i, :n].all().item()) for i, n in enumerate(lens)):
            enc.stats["head_fallback"] += 1
            return orig_forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype, detach_encoder)
        t_start = time.perf_counter()
        stats = enc.stats
        ids_c, mpos, qt = input_ids.cpu(), marker_pos.cpu(), qtype.cpu()
        logits = torch.full((B, K), -1e4, dtype=torch.float32)
        pooled = torch.zeros((B, enc.config.hidden_size), dtype=torch.float32)
        for i, (n, L) in enumerate(zip(lens, buckets)):
            hidden = enc.encode_row(ids_c[i, :n].numpy(), n, L)
            m = mask_bufs.setdefault(L, np.zeros((1, L), np.int32))
            m.fill(0)
            m[0, :n] = 1
            qtype_buf[0, 0] = int(qt[i])
            t0 = time.perf_counter()
            out = heads[L].predict({"hidden": hidden, "attention_mask": m, "qtype": qtype_buf})
            stats["head_ms"] += (time.perf_counter() - t0) * 1000
            keep[L] = out
            tl = torch.from_numpy(np.asarray(out["token_logits"], np.float32)[0])
            logits[i] = tl[mpos[i].clamp(min=0)]
            pooled[i] = torch.from_numpy(np.asarray(out["cls"], np.float32)[0])
        logits = logits.masked_fill(~marker_mask.cpu(), -1e4)
        # unchanged from laya.common.DecisionModel.forward
        p = torch.softmax(logits.detach(), -1)
        k = marker_mask.sum(-1).clamp(min=2).float()
        ent = -(p * torch.log(p.clamp_min(1e-9))).sum(-1) / torch.log(k)
        top2 = p.topk(2, -1).values
        feats = torch.stack([top2[:, 0], top2[:, 0] - top2[:, 1], ent, k / 255.0], -1)
        act_logits = self.act_head(torch.cat([pooled, feats], -1))
        stats["decision_model_ms"] += (time.perf_counter() - t_start) * 1000
        stats["head_rows"] += B
        return logits, act_logits

    DecisionModel.forward = forward


def skip_single_option():
    """Answer base-routed one-option choice questions without a model pass.

    With one option the softmax is exactly 1.0 for it, so the pass only ever produced
    {"choice": k, "probabilities": {k: 1.0}, "confidence": 1.0}. act_probability is the
    act head's output; Laya's act head is constant (1.0 on every one of the 2,028 recorded
    MAP03 cases), and that is what is returned. Questions routed to a fine-tuned head, which
    may compose a ranking or numeric residual, are always sent to the model.
    """
    from doomlib.question_heads import QuestionHeads
    orig_predict = QuestionHeads.predict

    def predict(self, state, questions):
        forced = {name: q for name, q in questions.items()
                  if name not in self.overrides and q.get("type") == "choice" and isinstance(q.get("criteria"), (dict, list))
                  and len(q["criteria"]) == 1 and "look_observation" not in q}
        rest = {name: q for name, q in questions.items() if name not in forced}
        result = orig_predict(self, state, rest) if rest else dict(model="laya-rl-agent", answers={}, usage={"input_tokens": 0, "output_tokens": 0})
        for name, q in forced.items():
            key = next(iter(q["criteria"]))
            result["answers"][name] = {"type": "choice", "choice": key, "probabilities": {key: 1.0}, "confidence": 1.0,
                                       "action": {"act_probability": 1.0}}
        result["answers"] = {name: result["answers"][name] for name in questions}
        return result

    QuestionHeads.predict = predict


def guard_residuals(thresholds):
    """Stop a numeric residual from overturning a confident Laya answer.

    `thresholds` maps a question ("command", "item", "switch", "movement", "enemy") to tau. When
    the residual's final choice differs from Laya's own (`numeric_residual.semantic_answer`) and
    Laya's top probability is at least tau, the answer becomes Laya's: its choice and its
    probabilities over the same options. The residual's output stays in the record, and
    `numeric_residual.guard` says whether the guard fired. Must run before the heads load,
    because `attach_numeric_residual` binds the predictor functions at load time.
    """
    import importlib
    targets = {"command": ("doomlib.numeric_command", "predict_with_numeric_residual"),
               "item": ("doomlib.numeric_item", "predict"), "movement": ("doomlib.numeric_movement", "predict"),
               "switch": ("doomlib.numeric_switch", "predict"), "enemy": ("doomlib.numeric_enemy", "predict")}
    for question, tau in thresholds.items():
        module_name, attr = targets[question]
        module = importlib.import_module(module_name)
        inner = getattr(module, attr)

        def guarded(agent, state, questions, _inner=inner, _tau=tau, _q=question):
            result = _inner(agent, state, questions)
            for name, answer in result["answers"].items():
                nr = answer.get("numeric_residual") if isinstance(answer, dict) else None
                if not nr or "semantic_answer" not in nr or answer.get("type") != "choice":
                    continue
                sem = nr["semantic_answer"]
                top = max(sem["probabilities"].values())
                fired = sem["choice"] != answer["choice"] and top >= _tau
                nr["guard"] = dict(question=_q, tau=_tau, fired=fired)
                if fired:
                    probs = {k: float(sem["probabilities"].get(k, 0.0)) for k in answer["probabilities"]}
                    total = sum(probs.values()) or 1.0
                    answer["probabilities"] = {k: v / total for k, v in probs.items()}
                    answer["choice"] = sem["choice"]
                    answer["confidence"] = answer["probabilities"][sem["choice"]]
                    _STATS["guard_fired_" + _q] += 1
            return result
        setattr(module, attr, guarded)


def command_memory(path):
    """Use a repaired command residual with short memory (`doom/repair_command_residual.py`).

    Replaces doomLaya's command predictor before the heads load. Everything matches
    `doomlib.numeric_command.predict_with_numeric_residual` except the network, which also takes
    `command_memory.Memory` features computed from the unprojected request state. Memory resets
    when no command request has arrived for 5 s (a new run). Memory
    advances on every command request, including one-option ones skipped by --skip-forced.
    """
    import copy
    import math
    import torch
    from safetensors.torch import load_file
    import doomlib.numeric_command as nc
    from doomlib.question_heads import QuestionHeads
    from command_memory import Memory, make_memory_residual

    path = Path(path)
    meta = json.loads((path / "meta.json").read_text())
    weights = load_file(str(path / "memory-residual.safetensors"))
    # Memory is updated once per request that asks the command question, from the unprojected
    # state, before any one-option skip -- the same sequence repair_command_residual.py trains on.
    current = {"mem": None}
    memory, last = Memory(meta["window"]), [0.0]
    built = {}

    inner_predict = QuestionHeads.predict

    def predict(self, state, questions):
        if "command" in questions:
            now = time.monotonic()
            if now - last[0] > 5:
                memory.reset()
            last[0] = now
            current["mem"] = memory.features(state)
        return inner_predict(self, state, questions)
    QuestionHeads.predict = predict

    def predictor(agent, state, questions):
        if "model" not in built:
            parent_sha = agent.cfg.get("doom_adaptation", {}).get("numeric_residual", {}).get("weights_sha256")
            if parent_sha != meta["parent_weights_sha256"]:
                raise SystemExit(f"--command-memory was trained on parent {meta['parent']}, but the served "
                                 f"command residual has weights {parent_sha}")
            m = make_memory_residual(agent.numeric_residual)
            m.load_state_dict(weights)
            built["model"] = m.to("cpu").eval()
        if set(questions) != {"command"}:
            raise ValueError("Numeric command residual only supports the command question")
        question = questions["command"]
        if question.get("type") != "choice" or not question.get("criteria") or set(question["criteria"]) - set(nc.ACTIONS):
            raise ValueError("Numeric command residual requires explicit known action choices")
        original = agent._semantic_predict(state, questions)
        semantic_answer = original["answers"]["command"]
        spec = agent.numeric_residual_spec
        observed = nc.features(dict(state=state, question=question), spec["item_names"])
        if len(observed) != len(spec["feature_names"]) or not all(math.isfinite(v) for v in observed):
            raise ValueError("Invalid observed numerical features")
        mem = current["mem"]
        base = nc.semantic_logits(semantic_answer, spec["probability_floor"])
        with torch.no_grad():
            x = torch.tensor([observed], dtype=torch.float32)
            b = torch.tensor([base], dtype=torch.float32)
            logits = built["model"](x, b, torch.tensor([mem], dtype=torch.float32))[0]
            allowed = torch.tensor([a in question["criteria"] for a in nc.ACTIONS])
            p = logits.masked_fill(~allowed, -float("inf")).double().softmax(-1).tolist()
            delta = (logits - b[0]).tolist()
        probabilities = {k: p[nc.ACTIONS.index(k)] for k in question["criteria"]}
        choice = max(probabilities, key=probabilities.get)
        answer = dict(type="choice", choice=choice, probabilities=probabilities, confidence=probabilities[choice],
                      action=copy.deepcopy(semantic_answer.get("action", {})),
                      numeric_residual=dict(composition=meta["format"], semantic_answer=semantic_answer,
                                            logit_delta={k: delta[nc.ACTIONS.index(k)] for k in question["criteria"]},
                                            memory=dict(zip(meta["memory_features"], mem))))
        _STATS["command_memory_calls"] += 1
        return dict(original, answers={"command": answer})

    nc.predict_with_numeric_residual = predictor


def commit_route(max_hold=20.0, resume=0.0, floor=0.0, pickups=False, margin=1.0):
    """Hold Laya's last route command until the executor finishes it.

    MAP03 runs fail by flipping between route commands every ~2 s (use_switch #332 -> exit ->
    open_door -> use_switch ...), each abandoned before the executor can finish it. Once Laya
    chooses a route command (use_switch with its target, exit, open_door), later answers keep
    it while the state says "Current command: <it>; status executing", no enemy is visible or
    recently seen, it is still offered, and less than `max_hold` seconds have passed. The
    executor's arrived/completed/blocked/unavailable status, a visible enemy, or the timeout
    releases it; Laya's look_back is never overridden. Laya's own choice and probabilities stay
    in the answer under `commitment`.

    With `resume` > 0: when a held command is released as blocked and Laya answers open_door,
    the blocked command becomes the parent goal. The executor reports an opened door as
    "unavailable" (the closed door it targeted is gone), and on MAP03 Laya then walks away from
    the exit while the open-wait-close door shuts again. For `resume` seconds the next answer
    that is neither open_door nor the parent goes back to the parent, if it is still offered
    and the area is calm.

    With `floor` > 0, a command is held or resumed only while Laya still gives it at least that
    probability: on MAP01 an unfloored hold overrode a pickup Laya chose at 0.99999.

    With `pickups`, pickup and its item target are held the same way: on MAP02 Laya flipped
    between a lift and two far shotguns every 1.5 s and reached none of them.

    `margin` limits target (switch/item) holds to near-ties: the held target is kept only while
    Laya's new target beats it by less than `margin`. Holding MAP01 item targets against 0.75-0.93
    preferences cost 60-90 s of pickups and timed out 3 of 4 runs.
    Patch after the other QuestionHeads wrappers so it sees their final answers.
    """
    import re
    import time
    from doomlib.question_heads import QuestionHeads
    route = ("use_switch", "exit", "open_door") + (("pickup",) if pickups else ())
    target_question = {"use_switch": "switch", "pickup": "item"}
    held = {}  # action, target, since
    parent = {}  # action, since
    inner_predict = QuestionHeads.predict

    def predict(self, state, questions):
        result = inner_predict(self, state, questions)
        answers = result["answers"]
        now = time.monotonic()
        current = re.search(r"^Current command: (\w+)(?: #([^\s;]+))?[^\n]*?; status (\w+)", state, re.M)
        calm = "No enemies visible or seen in the last two seconds." in state
        if held and not (calm and current and current[1] == held["action"] and current[3] == "executing"
                         and (held["target"] is None or current[2] == held["target"])
                         and now - held["since"] < max_hold):
            held.clear()
        command = answers.get("command")
        if parent and now - parent["since"] >= resume:
            parent.clear()
        if command is not None:
            choice = command["choice"]
            if resume and current and current[3] == "blocked" and current[1] in route and choice == "open_door":
                parent.update(action=current[1], since=now)
            probability = command["probabilities"]
            if (held and probability.get(held["action"], 0.0) >= max(floor, 1e-12)
                    and choice not in (held["action"], "look_back")):
                command["commitment"] = dict(held=held["action"], target=held["target"], model_choice=choice,
                                             age_seconds=round(now - held["since"], 2))
                command["choice"] = held["action"]
            elif (not held and parent and calm and probability.get(parent["action"], 0.0) >= max(floor, 1e-12)
                  and choice not in (parent["action"], "open_door", "look_back")):
                command["commitment"] = dict(resumed=parent["action"], model_choice=choice,
                                             age_seconds=round(now - parent["since"], 2))
                command["choice"] = parent["action"]
                parent.clear()
            if not held and calm and command["choice"] in route:
                held.update(action=command["choice"], target=None, since=now)
        name = target_question.get(held.get("action"))
        target = answers.get(name) if name else None
        if target is not None:
            if held["target"] is None:
                held["target"] = target["choice"]
            elif (target["choice"] != held["target"] and target["probabilities"].get(held["target"], 0.0) >= max(floor, 1e-12)
                  and target["probabilities"][target["choice"]] - target["probabilities"][held["target"]] < margin):
                target["commitment"] = dict(held=held["target"], model_choice=target["choice"],
                                            age_seconds=round(now - held["since"], 2))
                target["choice"] = held["target"]
        return result

    QuestionHeads.predict = predict


def random_chooser(seed):
    """Baseline without the model's judgement: every choice answer becomes a uniform random pick.

    The model still runs, so decisions take as long as Laya's; then each choice answer (command,
    targets, weapon, movement, the planner's objective) is replaced by a uniform draw over the options
    that question offered, with uniform probabilities. Laya's own answer stays in the record under
    `random_baseline`. Patch after the other QuestionHeads wrappers so it replaces their final answers.
    """
    import random
    from doomlib.question_heads import QuestionHeads
    inner = QuestionHeads.predict
    rng = random.Random(seed)

    def predict(self, state, questions):
        result = inner(self, state, questions)
        for name, answer in result["answers"].items():
            if not isinstance(answer, dict) or answer.get("type") != "choice" or not answer.get("probabilities"):
                continue
            options = list(answer["probabilities"])
            choice = rng.choice(options)
            answer["random_baseline"] = dict(model_choice=answer["choice"], model_probabilities=answer["probabilities"])
            answer["probabilities"] = {k: 1.0 / len(options) for k in options}
            answer["choice"] = choice
            answer["confidence"] = 1.0 / len(options)
        return result

    QuestionHeads.predict = predict


def install(encoder="ane", record_file="encoder_lengths.jsonl", compute_units="CPU_AND_NE", heads="cpu", skip_forced=False,
            residual_guard=None, memory_path=None, commit=None, commit_resume=0.0, commit_floor=0.0, commit_pickups=False, commit_margin=1.0, random_seed=None):
    """Patch laya.Agent so every Agent built afterwards uses the chosen encoder (and heads)."""
    sys.path.insert(0, os.getcwd())
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import laya
    if heads == "ane":
        if encoder != "ane":
            raise SystemExit("--heads ane needs --encoder ane")
        ane_heads(compute_units)
    if skip_forced:
        skip_single_option()
    if memory_path:
        command_memory(memory_path)
    if residual_guard:
        guard_residuals(residual_guard)
    if commit:
        commit_route(commit, commit_resume, commit_floor, commit_pickups, commit_margin)
    if random_seed is not None:
        random_chooser(random_seed)
    Base = laya.Agent

    class PatchedAgent(Base):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            enc = self.model.encoder
            if encoder == "record":
                self.model.encoder = RecordingEncoder(enc, record_file).wrap()
            elif encoder == "ane":
                self.model.encoder = ane_encoder(enc, self.tok.pad_token_id, compute_units)
                globals()["_ANE_STATS"] = self.model.encoder.stats
            print(f"[ane_doomlaya] encoder = {encoder}", flush=True)

    laya.Agent = PatchedAgent
    if encoder == "ane":
        import atexit
        atexit.register(lambda: print("[ane] encoder calls:", dict(globals().get("_ANE_STATS", {})), flush=True))


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--encoder", choices=["torch", "record", "ane"], default="ane")
    ap.add_argument("--record-file", default="encoder_lengths.jsonl")
    ap.add_argument("--compute-units", default="CPU_AND_NE")
    ap.add_argument("--heads", choices=["cpu", "ane"], default="cpu")
    ap.add_argument("--skip-forced", action="store_true", help="answer one-option base questions without a pass")
    ap.add_argument("--residual-guard", nargs="*", default=[], metavar="QUESTION=TAU",
                    help="keep Laya's answer over a numeric residual when Laya's top probability >= TAU")
    ap.add_argument("--commit-route", type=float, metavar="SECONDS",
                    help="hold Laya's route command until the executor finishes it, at most SECONDS")
    ap.add_argument("--commit-resume", type=float, default=0.0, metavar="SECONDS",
                    help="with --commit-route: return to a door-blocked command for SECONDS after open_door")
    ap.add_argument("--commit-floor", type=float, default=0.0, metavar="P",
                    help="with --commit-route: hold or resume a command only while Laya gives it at least P")
    ap.add_argument("--commit-pickups", action="store_true", help="with --commit-route: also hold pickup and its item")
    ap.add_argument("--commit-margin", type=float, default=1.0, metavar="M",
                    help="with --commit-route: keep a held switch/item target only while Laya's new target beats it by < M")
    ap.add_argument("--random-chooser", type=int, metavar="SEED",
                    help="baseline: replace every choice answer with a uniform random option (the model still runs)")
    ap.add_argument("--command-memory", help="repaired command residual with memory (doom/repair_command_residual.py)")
    ours, rest = ap.parse_known_args()
    guard = {q: float(t) for q, t in (g.split("=") for g in ours.residual_guard)}
    install(ours.encoder, ours.record_file, ours.compute_units, ours.heads, ours.skip_forced, guard, ours.command_memory, ours.commit_route, ours.commit_resume, ours.commit_floor, ours.commit_pickups, ours.commit_margin, ours.random_chooser)
    sys.argv = ["serve_doom_laya.py"] + rest
    import runpy
    runpy.run_path("serve_doom_laya.py", run_name="__main__")


if __name__ == "__main__":
    main()
