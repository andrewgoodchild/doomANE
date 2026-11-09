"""Evaluate the next-frame model on the held-out split: one-step PSNR (against a copy-the-last-frame baseline),
35-step autoregressive rollouts with the recorded actions, a side-by-side GIF (real above, predicted below), a
PNG strip at t+1, t+5, t+10, t+20, t+35, and an action-response test (turn left against turn right from the same
context must shift the view in opposite directions).

    ext/vision-venv/bin/python poc/dream/evaluate.py ext/poc-data/dream/run1 --out results/poc/dream
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from model import Dream, CONTEXT  # noqa: E402
from train import Data, to_float  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TURN_LEFT = np.array([0, 0, 1, 0, 0, 0, 0], np.float32)
TURN_RIGHT = np.array([0, 0, 0, 1, 0, 0, 0], np.float32)


def psnr_np(a, b):
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    return float(10 * np.log10(1.0 / max(mse, 1e-10)))


def horizontal_shift(ref, img, max_shift=12):
    """Shift (pixels) that best aligns img to ref along x, by column-profile correlation; positive = content moved right."""
    r = ref.mean(axis=(0, 2)); r = r - r.mean()
    g = img.mean(axis=(0, 2)); g = g - g.mean()
    best, best_s = -1e9, 0
    for s in range(-max_shift, max_shift + 1):
        if s >= 0:
            c = float(np.dot(r[:len(r) - s], g[s:]))
        else:
            c = float(np.dot(r[-s:], g[:len(g) + s]))
        if c > best:
            best, best_s = c, s
    return best_s


def main():
    p = argparse.ArgumentParser()
    p.add_argument('run', type=Path)
    p.add_argument('--data', type=Path, default=ROOT / 'ext/poc-data/dream')
    p.add_argument('--maps', nargs='+', default=['MAP01', 'MAP02', 'MAP03'])
    p.add_argument('--out', type=Path, default=ROOT / 'results/poc/dream')
    p.add_argument('--one-step-samples', type=int, default=3000)
    p.add_argument('--rollouts', type=int, default=24)
    p.add_argument('--steps', type=int, default=35)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    ckpt = torch.load(a.run / 'model.pt', map_location='cpu', weights_only=False)
    model = Dream(ckpt['width']); model.load_state_dict(ckpt['state']); model = model.to(device).eval()
    test = Data(a.data, a.maps, 'test')
    rng = np.random.default_rng(1)
    report = {'checkpoint_step': ckpt.get('step'), 'maps': a.maps}

    # one-step PSNR, by map, against copying the last frame
    rows = test.index[rng.choice(len(test), min(a.one_step_samples, len(test)), replace=False)]
    per_map = {m: {'model': [], 'copy_last': []} for m in a.maps}
    with torch.no_grad():
        for i in range(0, len(rows), 64):
            chunk = rows[i:i + 64]
            x, act, y = test.batch(chunk)
            xf, yf, af = to_float(x, device), to_float(y, device), act.to(device).float()
            pred = model(xf, af).cpu().numpy()
            yn = yf.cpu().numpy(); last = xf[:, -3:].cpu().numpy()
            for j, (k, t) in enumerate(chunk):
                per_map[a.maps[k]]['model'].append(psnr_np(pred[j], yn[j]))
                per_map[a.maps[k]]['copy_last'].append(psnr_np(last[j], yn[j]))
    report['one_step_psnr'] = {m: {kind: round(float(np.mean(v)), 2) for kind, v in d.items()} for m, d in per_map.items()}
    report['one_step_psnr']['all'] = {kind: round(float(np.mean([v for d in per_map.values() for v in d[kind]])), 2) for kind in ('model', 'copy_last')}
    print('one-step PSNR', report['one_step_psnr'], flush=True)

    # rollouts: a held-out start whose next --steps frames stay in one episode
    starts = []
    episodes = {k: np.load(a.data / f'meta_{m}.npz')['episode'] for k, m in enumerate(a.maps)}
    while len(starts) < a.rollouts:
        k, t = test.index[rng.integers(0, len(test))]
        if t + a.steps < len(test.frames[k]) and episodes[k][t + a.steps] == episodes[k][t - CONTEXT]:
            starts.append((k, t))
    step_psnr = np.zeros((len(starts), a.steps)); copy_psnr = np.zeros_like(step_psnr)
    gif_real, gif_pred = None, None
    with torch.no_grad():
        for r, (k, t) in enumerate(starts):
            ctx = test.frames[k][t - CONTEXT:t]                       # 4 x H x W x 3 uint8
            x = torch.from_numpy(np.ascontiguousarray(ctx)).permute(0, 3, 1, 2).reshape(1, 3 * CONTEXT, 96, 128)
            x = x.to(device).float() / 255
            preds = []
            for s in range(a.steps):
                act = torch.from_numpy(test.actions[k][t - 1 + s][None]).to(device).float()
                pred = model(x, act)
                preds.append(pred[0].cpu().numpy())
                x = torch.cat([x[:, 3:], pred], 1)
                real = test.frames[k][t + s].astype(np.float32) / 255
                step_psnr[r, s] = psnr_np(pred[0].cpu().numpy().transpose(1, 2, 0), real)
                copy_psnr[r, s] = psnr_np(ctx[-1].astype(np.float32) / 255, real)
            if r == 0:
                gif_len = 70 if t + 70 < len(test.frames[k]) and episodes[k][t + 70] == episodes[k][t] else a.steps  # 2 s when the episode allows
                gif_real = [test.frames[k][t + s] for s in range(gif_len)]
                # continue the rollout for the GIF length
                x = torch.from_numpy(np.ascontiguousarray(ctx)).permute(0, 3, 1, 2).reshape(1, 3 * CONTEXT, 96, 128).to(device).float() / 255
                gif_pred = []
                for s in range(len(gif_real)):
                    act = torch.from_numpy(test.actions[k][t - 1 + s][None]).to(device).float()
                    pred = model(x, act); x = torch.cat([x[:, 3:], pred], 1)
                    gif_pred.append((pred[0].cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8))
                strip_steps = [1, 5, 10, 20, 35]
                strip = [np.concatenate([test.frames[k][t + s - 1], (preds[s - 1].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)], 0) for s in strip_steps]
                Image.fromarray(np.concatenate(strip, 1)).resize((128 * len(strip) * 2, 96 * 2 * 2), Image.NEAREST).save(a.out / 'rollout_strip.png')
                report['strip'] = {'steps': strip_steps, 'layout': 'real above, predicted below', 'map': a.maps[k], 'start_frame': int(t)}
    mean_step = step_psnr.mean(0)
    report['rollout_psnr'] = {'steps': {str(s): round(float(mean_step[s - 1]), 2) for s in (1, 5, 10, 20, 35) if s <= a.steps},
                              'copy_last_at_same_steps': {str(s): round(float(copy_psnr.mean(0)[s - 1]), 2) for s in (1, 5, 10, 20, 35) if s <= a.steps},
                              'rollouts': len(starts), 'per_step': [round(float(v), 2) for v in mean_step]}
    print('rollout PSNR', report['rollout_psnr']['steps'], 'copy-last', report['rollout_psnr']['copy_last_at_same_steps'], flush=True)
    frames = [Image.fromarray(np.concatenate([r_, p_], 0)).resize((256, 384), Image.NEAREST) for r_, p_ in zip(gif_real, gif_pred)]
    frames[0].save(a.out / 'rollout.gif', save_all=True, append_images=frames[1:], duration=int(1000 / 35), loop=0)
    report['gif'] = {'frames': len(frames), 'seconds': round(len(frames) / 35, 1), 'layout': 'real above, predicted below, 2x nearest'}

    # action response: turn left vs turn right from the same context
    signs, shifts = [], []
    with torch.no_grad():
        for _ in range(200):
            k, t = test.index[rng.integers(0, len(test))]
            ctx = test.frames[k][t - CONTEXT:t]
            x = torch.from_numpy(np.ascontiguousarray(ctx)).permute(0, 3, 1, 2).reshape(1, 3 * CONTEXT, 96, 128).to(device).float() / 255
            last = ctx[-1].astype(np.float32) / 255
            left = model(x, torch.from_numpy(TURN_LEFT[None]).to(device))[0].cpu().numpy().transpose(1, 2, 0)
            right = model(x, torch.from_numpy(TURN_RIGHT[None]).to(device))[0].cpu().numpy().transpose(1, 2, 0)
            sl, sr = horizontal_shift(last[:60], left[:60]), horizontal_shift(last[:60], right[:60])  # the upper view, not the weapon
            shifts.append((sl, sr)); signs.append(sl > 0 and sr < 0)
    shifts = np.array(shifts)
    report['action_response'] = {'contexts': len(signs), 'opposite_and_correct_sign_share': round(float(np.mean(signs)), 3),
                                 'mean_shift_turn_left_px': round(float(shifts[:, 0].mean()), 2), 'mean_shift_turn_right_px': round(float(shifts[:, 1].mean()), 2),
                                 'note': 'positive shift = view content moved right, expected for turning left'}
    print('action response', report['action_response'], flush=True)
    (a.out / 'evaluation.json').write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
