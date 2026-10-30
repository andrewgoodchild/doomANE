"""Export doomLaya v0.3.1's shared encoder (ModernBERT-large, fine-tuned) to fixed-length Core ML packages.

    cd ext/doomLaya && .venv/bin/python ../../doom/export_doom_encoder.py --length 384   # one length per process

Uses `doom/coreml_convert.py` (explicit additive masks, torch.export,
fp16, inputs input_ids/attention_mask int32 [1,L], output `hidden` [1,L,1024]). Weights come from the
bundle's `model.safetensors` (keys `encoder.*`) and `encoder/config.json`. Writes
results/coreml/doomlaya_enc_L{L}_fp16.mlpackage and a parity report against PyTorch on a padded input.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "doom"))
sys.path.insert(0, str(ROOT))
import coreml_cache  # noqa: E402  compile once, never MLModel(mlpackage)
from coreml_convert import EncoderWrapper, register_missing_torch_ops  # noqa: E402

BUNDLE = ROOT / "ext" / "doomLaya" / "checkpoints" / "laya-doom-v031" / "laya-map2-explicit-v2"


def load_encoder():
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModel
    cfg = AutoConfig.from_pretrained(str(BUNDLE / "encoder"))
    model = AutoModel.from_config(cfg, attn_implementation="eager")
    sd = {k[len("encoder."):]: v for k, v in load_file(str(BUNDLE / "model.safetensors")).items() if k.startswith("encoder.")}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if unexpected or missing:
        raise SystemExit(f"weight mismatch: missing {missing[:5]} unexpected {unexpected[:5]}")
    return model.float().eval()


def main():
    import coremltools as ct
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=int, required=True)
    args = ap.parse_args()
    L = args.length
    register_missing_torch_ops()
    model = load_encoder()
    rng = np.random.default_rng(0)
    n = max(8, int(L * 0.7))
    ids = np.full((1, L), 50283, np.int32)  # ModernBERT pad id
    ids[0, :n] = rng.integers(1000, 30000, n)
    am = np.zeros((1, L), np.int32)
    am[0, :n] = 1
    with torch.no_grad():
        # the reference first: the wrapper replaces transformers 4.x's mask method on the model
        ref = model(input_ids=torch.tensor(ids).long(), attention_mask=torch.tensor(am).long()).last_hidden_state.numpy()
        wrapper = EncoderWrapper(model, L).eval()
        ours = wrapper(torch.tensor(ids), torch.tensor(am)).numpy()
        traced = torch.export.export(wrapper, (torch.tensor(ids), torch.tensor(am))).run_decompositions({})
    t0 = time.time()
    ml = ct.convert(traced, inputs=[ct.TensorType(name="input_ids", shape=(1, L), dtype=np.int32),
                                    ct.TensorType(name="attention_mask", shape=(1, L), dtype=np.int32)],
                    outputs=[ct.TensorType(name="hidden", dtype=np.float16)],
                    compute_precision=ct.precision.FLOAT16, minimum_deployment_target=ct.target.macOS15,
                    convert_to="mlprogram")
    out = ROOT / "results" / "coreml" / f"doomlaya_enc_L{L}_fp16.mlpackage"
    ml.save(str(out))
    conv_s = time.time() - t0
    m = coreml_cache.load(out, ct.ComputeUnit.CPU_AND_NE)
    feed = {"input_ids": ids.copy(), "attention_mask": am.copy()}
    m.predict(feed)
    t1 = time.perf_counter()
    for _ in range(10):
        hid = np.asarray(m.predict(feed)["hidden"], np.float32)
    ms = (time.perf_counter() - t1) / 10 * 1000
    d = np.abs(hid[0, :n] - ref[0, :n])
    cos = float((hid[0, :n] * ref[0, :n]).sum(-1).mean() / (np.linalg.norm(hid[0, :n], axis=-1) * np.linalg.norm(ref[0, :n], axis=-1)).mean())
    rep = {"length": L, "tokens": n, "wrapper_vs_hf_max_abs": float(np.abs(ours - ref)[0, :n].max()),
           "ane_vs_hf_max_abs": float(d.max()), "ane_vs_hf_mean_abs": float(d.mean()), "cosine": round(cos, 5),
           "convert_s": round(conv_s, 1), "ane_ms": round(ms, 1)}
    (ROOT / "results" / f"doomlaya_enc_L{L}.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep), flush=True)


if __name__ == "__main__":
    main()
