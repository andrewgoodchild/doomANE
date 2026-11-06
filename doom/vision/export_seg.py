"""Export a trained pixel perception model (train_seg.py) to Core ML, compile it once and time it.

Input `view`: [1,200,320,3] uint8-valued float16, the view above the status bar at half resolution
(screen_buffer[:400:2, ::2]), with depth_buffer[:400:2, ::2] as a fourth channel for a pix3 model; the resize to 192x320 and scaling happen inside the model, as in training.
Output `logits`: [1,C,48,80]. The compiled model goes next to the package (a stable path, loaded with
CompiledMLModel, so no temporary copies), with classes.json beside it.

    ext/vision-venv/bin/python doom/vision/export_seg.py results/vision/mobileclip
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
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import train_seg  # noqa: E402


class Runtime(torch.nn.Module):
    def __init__(self, seg):
        super().__init__()
        self.seg = seg

    def forward(self, view):  # 1 x 200 x 320 x 3 (x 4 with depth: the depth buffer last), values 0..255
        x = view.permute(0, 3, 1, 2) / 255
        x = F.interpolate(x, size=(train_seg.H, train_seg.W), mode='bilinear', align_corners=False)
        return self.seg(x)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('run', type=Path)
    p.add_argument('--data', type=Path, default=Path('ext/vision-data/frames'))
    p.add_argument('--monster-threshold', type=float, default=None,
                   help='a cell is a monster when its summed monster-class probability is at least this (operating_point.py); default argmax')
    a = p.parse_args()
    ckpt = torch.load(a.run / 'model.pt', map_location='cpu', weights_only=False)
    seg = train_seg.Seg(ckpt['backbone'], len(ckpt['classes']), ckpt.get('depth', False))
    seg.load_state_dict(ckpt['state'])
    seg.eval()
    if ckpt['backbone'] == 'mobileclip':
        from timm.utils.model import reparameterize_model
        rgb = seg.encoder.rgb if ckpt.get('depth') else seg.encoder
        rgb.trunk = reparameterize_model(rgb.trunk)
        for m in rgb.trunk.modules():
            if hasattr(m, 'fused_attn'):
                m.fused_attn = False  # plain matmul attention converts; the fused op does not
    model = Runtime(seg).eval()
    channels = 4 if ckpt.get('depth') else 3
    example = torch.randint(0, 256, (1, 200, 320, channels)).float()
    traced = torch.export.export(model, (example,)).run_decompositions({})  # torch.jit.trace breaks on FastViT's attention reshapes
    ml = ct.convert(traced, inputs=[ct.TensorType(name='view', shape=example.shape, dtype=np.float16)],
                    outputs=[ct.TensorType(name='logits', dtype=np.float16)],
                    minimum_deployment_target=ct.target.macOS15, compute_precision=ct.precision.FLOAT16)
    package = a.run / 'seg.mlpackage'
    compiled = a.run / 'seg.mlmodelc'
    shutil.rmtree(package, ignore_errors=True); shutil.rmtree(compiled, ignore_errors=True)
    ml.save(str(package))
    ct.utils.compile_model(str(package), str(compiled))
    (a.run / 'classes.json').write_text(json.dumps({'classes': ckpt['classes'], 'category': ckpt['category'], 'depth': channels == 4,
                                                   'monster_threshold': a.monster_threshold}, indent=1))
    report = {}
    data = a.data
    frames = np.load(data / f"{ckpt['val_maps'][0]}.npz")
    views = (np.concatenate([frames['images'], frames['depth'][..., None]], -1) if channels == 4 else frames['images'])[:40]
    with torch.no_grad():
        ref = torch.cat([model(torch.from_numpy(v[None]).float()) for v in views]).argmax(1).numpy()
    for units in ('CPU_AND_NE', 'CPU_AND_GPU', 'CPU_ONLY'):
        cm = ct.models.CompiledMLModel(str(compiled), compute_units=getattr(ct.ComputeUnit, units))
        for v in views[:3]:
            cm.predict({'view': v[None].astype(np.float16)})
        times, agree = [], []
        for i, v in enumerate(views):
            t0 = time.perf_counter()
            out = cm.predict({'view': v[None].astype(np.float16)})['logits']
            times.append(1000 * (time.perf_counter() - t0))
            agree.append(float((out.argmax(1)[0] == ref[i]).mean()))
        report[units] = {'ms_median': round(statistics.median(times), 2), 'cell_agreement_with_torch': round(float(np.mean(agree)), 4)}
        print(units, report[units], flush=True)
    (a.run / 'export.json').write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
