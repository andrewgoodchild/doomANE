"""Export doomLaya v0.3.1's question heads (Laya DecisionModel minus the encoder) to Core ML.

    cd ext/doomLaya && .venv/bin/python ../../doom/export_doom_heads.py [--only NAME ...]

Each head checkpoint's two norm-first transformer layers, type embedding and per-token scorer
become one multifunction package `results/coreml/doomlaya_head_<sha12>_fp16.mlpackage`, with one
function `L{L}` per encoder length (weights stored once). Inputs: `hidden` [1,L,1024] fp16 (the
encoder package's output), `attention_mask` [1,L] int32, `qtype` [1,1] int32. Outputs:
`token_logits` [1,L] (the scorer at every position; the runtime gathers the option markers) and
`cls` [1,1024] (position 0 after the head, for the act head). `<sha12>` hashes the head weights,
so the runtime can find a package from a loaded model. A parity report goes to
results/doomlaya_heads.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import coreml_cache  # noqa: E402  compile once, never MLModel(mlpackage)
CKPT = ROOT / "ext" / "doomLaya" / "checkpoints" / "laya-doom-v031"
LENGTHS = (128, 256, 320, 384, 448, 512, 640)
PREFIXES = ("head.", "type_emb.", "scorer.")


def checkpoints():
    """Every head the v0.3.1 server loads: the base bundle plus each --head-checkpoint."""
    args = (ROOT / "doom" / "doomlaya_args.txt").read_text().split()
    paths = [args[args.index("--checkpoint") + 1], args[args.index("--item-head-checkpoint") + 1]]
    paths += [a.split("=", 1)[1] for a in args if "=" in a and a.split("=", 1)[0] in
              ("movement", "command", "look_gate", "switch", "enemy", "weapon")]
    return [ROOT / "ext" / "doomLaya" / p for p in dict.fromkeys(paths)]


def head_state(path):
    from safetensors.torch import load_file
    sd = load_file(str(path / "model.safetensors"))
    return {k: v.float() for k, v in sd.items() if k.startswith(PREFIXES)}


def head_digest(named_tensors):
    """sha256 over the head tensors in name order; runtime computes the same from the live model."""
    h = hashlib.sha256()
    for name in sorted(named_tensors):
        if name.startswith(PREFIXES):
            h.update(name.encode())
            h.update(named_tensors[name].detach().float().contiguous().numpy().tobytes())
    return h.hexdigest()[:12]


class HeadCore(torch.nn.Module):
    """nn.TransformerEncoderLayer(norm_first, relu) x N with an explicit additive key mask."""

    def __init__(self, sd, d):
        super().__init__()
        self.d, self.nh = d, max(1, d // 64)
        self.n_layers = len({k.split(".")[2] for k in sd if k.startswith("head.layers.")})
        for k, v in sd.items():
            self.register_buffer(k.replace(".", "_"), v)

    def w(self, name):
        return getattr(self, name.replace(".", "_"))

    def forward(self, hidden, attention_mask, qtype):
        B, L, d = hidden.shape
        nh, hd = self.nh, d // self.nh
        x = hidden.float() + self.w("type_emb.weight")[qtype[:, 0]][:, None, :]
        bias = (1.0 - attention_mask.float())[:, None, None, :] * -1e4
        for i in range(self.n_layers):
            p = f"head.layers.{i}."
            y = torch.nn.functional.layer_norm(x, (d,), self.w(p + "norm1.weight"), self.w(p + "norm1.bias"), 1e-5)
            qkv = y @ self.w(p + "self_attn.in_proj_weight").t() + self.w(p + "self_attn.in_proj_bias")
            q, k, v = qkv.split(d, -1)
            q = q.reshape(B, L, nh, hd).transpose(1, 2)
            k = k.reshape(B, L, nh, hd).transpose(1, 2)
            v = v.reshape(B, L, nh, hd).transpose(1, 2)
            a = torch.softmax(q @ k.transpose(-1, -2) * (hd ** -0.5) + bias, -1) @ v
            a = a.transpose(1, 2).reshape(B, L, d)
            x = x + a @ self.w(p + "self_attn.out_proj.weight").t() + self.w(p + "self_attn.out_proj.bias")
            y = torch.nn.functional.layer_norm(x, (d,), self.w(p + "norm2.weight"), self.w(p + "norm2.bias"), 1e-5)
            y = torch.relu(y @ self.w(p + "linear1.weight").t() + self.w(p + "linear1.bias"))
            x = x + y @ self.w(p + "linear2.weight").t() + self.w(p + "linear2.bias")
        s = torch.nn.functional.layer_norm(x, (d,), self.w("scorer.0.weight"), self.w("scorer.0.bias"), 1e-5)
        s = torch.nn.functional.gelu(s @ self.w("scorer.1.weight").t() + self.w("scorer.1.bias"))
        token_logits = (s @ self.w("scorer.3.weight").t() + self.w("scorer.3.bias"))[..., 0]
        return token_logits, x[:, 0]


def reference(sd, d, hidden, mask, qtype):
    """The same computation through Laya's own DecisionModel modules."""
    from laya.common import DecisionModel
    enc = torch.nn.Module()
    enc.config = type("C", (), {"hidden_size": d})()
    m = DecisionModel(enc, 2, 2).eval()
    m.load_state_dict({k: v for k, v in sd.items()}, strict=False)
    with torch.no_grad():
        h = hidden + m.type_emb(qtype[:, 0])[:, None, :]
        for layer in m.head.layers:
            h = layer(h, src_key_padding_mask=~mask.bool())
        return m.scorer(h).squeeze(-1), h[:, 0]


def package(sha):
    return ROOT / "results" / "coreml" / f"doomlaya_head_{sha}_fp16.mlpackage"


def export(path, workdir):
    import coremltools as ct
    sd = head_state(path)
    d = sd["type_emb.weight"].shape[1]
    sha = head_digest(sd)
    core = HeadCore(sd, d).eval()
    rng = np.random.default_rng(0)
    funcs, report = {}, {"checkpoint": path.name, "sha12": sha, "lengths": {}}
    for L in LENGTHS:
        n = int(L * 0.8)
        hidden = torch.tensor(rng.standard_normal((1, L, d)).astype(np.float32))
        mask = torch.zeros((1, L), dtype=torch.int32)
        mask[0, :n] = 1
        qtype = torch.zeros((1, 1), dtype=torch.int32)
        with torch.no_grad():
            ours = core(hidden, mask, qtype)
            traced = torch.export.export(core, (hidden.half(), mask, qtype)).run_decompositions({})
        ref = reference(sd, d, hidden, mask, qtype)
        ml = ct.convert(traced, inputs=[ct.TensorType(name="hidden", shape=(1, L, d), dtype=np.float16),
                                        ct.TensorType(name="attention_mask", shape=(1, L), dtype=np.int32),
                                        ct.TensorType(name="qtype", shape=(1, 1), dtype=np.int32)],
                        outputs=[ct.TensorType(name="token_logits", dtype=np.float16), ct.TensorType(name="cls", dtype=np.float16)],
                        compute_precision=ct.precision.FLOAT16, minimum_deployment_target=ct.target.macOS15,
                        convert_to="mlprogram")
        f = workdir / f"{sha}_L{L}.mlpackage"
        ml.save(str(f))
        funcs[f"L{L}"] = f
        m = coreml_cache.load(f, ct.ComputeUnit.CPU_AND_NE)
        feed = {"hidden": hidden.half().numpy(), "attention_mask": mask.numpy(), "qtype": qtype.numpy()}
        m.predict(feed)
        t = time.perf_counter()
        for _ in range(10):
            out = m.predict(feed)
        ms = (time.perf_counter() - t) / 10 * 1000
        tl = np.asarray(out["token_logits"], np.float32)[0, :n]
        report["lengths"][L] = {"ane_ms": round(ms, 2),
                                "rewrite_vs_laya_max_abs": float((ours[0] - ref[0]).abs()[0, :n].max()),
                                "ane_vs_laya_logit_max_abs": float(np.abs(tl - ref[0][0, :n].numpy()).max()),
                                "logit_scale": float(ref[0][0, :n].abs().max())}
        print(path.name, L, report["lengths"][L], flush=True)
    desc = ct.utils.MultiFunctionDescriptor()
    for name, f in funcs.items():
        desc.add_function(str(f), src_function_name="main", target_function_name=name)
    desc.default_function_name = f"L{LENGTHS[-1]}"
    out = package(sha)
    if out.exists():
        shutil.rmtree(out)
    ct.utils.save_multifunction(desc, str(out))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--missing", action="store_true", help="skip heads whose package already exists")
    args = ap.parse_args()
    reports = []
    out = ROOT / "results" / "doomlaya_heads.json"
    old = json.loads(out.read_text()) if out.exists() else []
    for path in checkpoints():
        if args.only and path.name not in args.only:
            continue
        if args.missing and package(head_digest(head_state(path))).exists():
            continue
        with tempfile.TemporaryDirectory() as tmp:
            reports.append(export(path, Path(tmp)))
    keep = [r for r in old if r["checkpoint"] not in {r2["checkpoint"] for r2 in reports}]
    out.write_text(json.dumps(keep + reports, indent=1))


if __name__ == "__main__":
    main()
