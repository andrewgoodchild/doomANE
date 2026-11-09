"""Export the trained model to Core ML (fixed shapes, float16, ML Program) and time one autoregressive step per
compute unit, loading through coreml_cache (never ct.models.MLModel on a package: CLAUDE.md).

Input `frames`: [1, 12, 96, 128] float16 in [0,1] (4 frames, oldest first); `action`: [1, 7] float16.
Output `next`: [1, 3, 96, 128] float16 in [0,1]. The timing loop feeds each prediction back as the newest frame,
so the median includes the roll and the numpy/Core ML hand-off, as a playable loop would pay.

    ext/vision-venv/bin/python poc/dream/export_time.py ext/poc-data/dream/run1
"""
import argparse
import json
import shutil
import statistics
import sys
import time
from pathlib import Path

import coremltools as ct
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(ROOT))
from model import Dream, CONTEXT  # noqa: E402
import coreml_cache  # noqa: E402


def load_model(run):
    ckpt = torch.load(run / 'model.pt', map_location='cpu', weights_only=False)
    model = Dream(ckpt['width'])
    model.load_state_dict(ckpt['state'])
    return model.eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('run', type=Path)
    p.add_argument('--calls', type=int, default=300)
    a = p.parse_args()
    model = load_model(a.run)
    frames = torch.rand(1, 3 * CONTEXT, 96, 128)
    action = torch.zeros(1, 7)
    traced = torch.jit.trace(model, (frames, action))
    ml = ct.convert(traced, inputs=[ct.TensorType(name='frames', shape=frames.shape, dtype=np.float16),
                                    ct.TensorType(name='action', shape=action.shape, dtype=np.float16)],
                    outputs=[ct.TensorType(name='next', dtype=np.float16)],
                    minimum_deployment_target=ct.target.macOS15, compute_precision=ct.precision.FLOAT16,
                    convert_to='mlprogram')
    package = a.run / 'dream.mlpackage'
    shutil.rmtree(package, ignore_errors=True)
    ml.save(str(package))
    size_mb = sum(f.stat().st_size for f in package.rglob('*') if f.is_file()) / 1e6
    # reference: torch on the same inputs
    rng = np.random.default_rng(0)
    x0 = rng.random((1, 3 * CONTEXT, 96, 128), dtype=np.float32)
    act = np.zeros((1, 7), np.float32); act[0, 0] = 1
    with torch.no_grad():
        ref = model(torch.from_numpy(x0), torch.from_numpy(act)).numpy()
    report = {'package_mb': round(size_mb, 1), 'parameters': sum(p.numel() for p in model.parameters())}
    for units in ('CPU_AND_NE', 'CPU_AND_GPU', 'CPU_ONLY'):
        cm = coreml_cache.load(package, units)
        x = x0.astype(np.float16)
        out = None
        for _ in range(20):  # warm-up
            out = cm.predict({'frames': x, 'action': act.astype(np.float16)})['next']
        first = cm.predict({'frames': x0.astype(np.float16), 'action': act.astype(np.float16)})['next'].astype(np.float32)
        agreement = float(np.abs(first - ref).mean())
        times = []
        x = x0.astype(np.float16)
        for _ in range(a.calls):
            t0 = time.perf_counter()
            out = cm.predict({'frames': x, 'action': act.astype(np.float16)})['next']
            x = np.concatenate([x[:, 3:], out.astype(np.float16)], 1)  # roll: the prediction becomes the newest frame
            times.append(1000 * (time.perf_counter() - t0))
        report[units] = {'ms_median': round(statistics.median(times), 2), 'ms_p90': round(float(np.percentile(times, 90)), 2),
                         'fps_at_median': round(1000 / statistics.median(times), 1), 'mean_abs_diff_vs_torch': round(agreement, 5)}
        print(units, report[units], flush=True)
    (a.run / 'export.json').write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
