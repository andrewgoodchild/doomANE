"""Merge the fixed-length doomLaya encoder packages into one multifunction package.

    cd ext/doomLaya && .venv/bin/python ../../doom/merge_doom_encoder.py [--delete-sources]

`results/coreml/doomlaya_enc_L{L}_fp16.mlpackage` (one full copy of the weights per length)
become `results/coreml/doomlaya_enc_fp16.mlpackage` with functions `L{L}`; Core ML stores the
shared weights once. Each function is checked against its source package on the same input
before any source is deleted. `doom/ane_doomlaya.py` prefers the merged package.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import coreml_cache  # noqa: E402  compile once, never MLModel(mlpackage)
PKG_DIR = ROOT / "results" / "coreml"


def main():
    import coremltools as ct
    ap = argparse.ArgumentParser()
    ap.add_argument("--delete-sources", action="store_true")
    a = ap.parse_args()
    src = {int(p.name.split("_L")[1].split("_")[0]): p for p in PKG_DIR.glob("doomlaya_enc_L*_fp16.mlpackage")}
    if not src:
        sys.exit("no per-length encoder packages")
    out = PKG_DIR / "doomlaya_enc_fp16.mlpackage"
    desc = ct.utils.MultiFunctionDescriptor()
    for L, p in sorted(src.items()):
        desc.add_function(str(p), src_function_name="main", target_function_name=f"L{L}")
    desc.default_function_name = f"L{max(src)}"
    if out.exists():
        shutil.rmtree(out)
    ct.utils.save_multifunction(desc, str(out))
    rng = np.random.default_rng(0)
    for L, p in sorted(src.items()):
        n = int(L * 0.7)
        feed = {"input_ids": np.full((1, L), 50283, np.int32), "attention_mask": np.zeros((1, L), np.int32)}
        feed["input_ids"][0, :n] = rng.integers(1000, 30000, n)
        feed["attention_mask"][0, :n] = 1
        ref = np.asarray(coreml_cache.load(p, ct.ComputeUnit.CPU_AND_NE).predict(dict(feed))["hidden"], np.float32)
        got = np.asarray(coreml_cache.load(out, ct.ComputeUnit.CPU_AND_NE, f"L{L}").predict(dict(feed))["hidden"], np.float32)
        diff = float(np.abs(ref[0, :n] - got[0, :n]).max())
        print(f"L{L}: max |merged - source| = {diff:.2e}", flush=True)
        if diff > 1e-2:
            sys.exit(f"L{L} differs; sources kept")
    if a.delete_sources:
        for p in src.values():
            shutil.rmtree(p)
        print("deleted", len(src), "per-length packages", flush=True)


if __name__ == "__main__":
    main()
