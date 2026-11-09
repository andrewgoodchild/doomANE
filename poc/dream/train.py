"""Train the next-frame model on the recordings (poc/dream/record.py) for a fixed wall time on the GPU (MPS).

Split: the last 10 % of frames of each map are the test split (never trained on). A sample is 4 consecutive
frames and the action taken after the 4th, all in one episode, predicting the 5th frame. Loss: L1.

    ext/vision-venv/bin/python poc/dream/train.py --minutes 55 --out ext/poc-data/dream/run1
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from model import Dream, CONTEXT, count_parameters  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


class Data:
    """Frames stay in uint8 memmaps; batches are gathered on the fly."""

    def __init__(self, data_dir, maps, split, holdout=0.1):
        self.frames, self.actions, self.index = [], [], []
        for k, m in enumerate(maps):
            f = np.load(data_dir / f'frames_{m}.npy', mmap_mode='r')
            meta = np.load(data_dir / f'meta_{m}.npz')
            ep = meta['episode']
            n = len(f)
            cut = int(n * (1 - holdout))
            lo, hi = (0, cut) if split == 'train' else (cut, n)
            # valid targets t: frames t-4..t in [lo, hi) and all in one episode
            t = np.arange(lo + CONTEXT, hi)
            ok = ep[t] == ep[t - CONTEXT]
            self.frames.append(f); self.actions.append(meta['actions'])
            self.index.append(np.stack([np.full(ok.sum(), k), t[ok]], 1))
        self.index = np.concatenate(self.index)

    def __len__(self):
        return len(self.index)

    def batch(self, rows):
        xs, acts, ys = [], [], []
        for k, t in rows:
            clip = self.frames[k][t - CONTEXT:t + 1]            # 5 x H x W x 3
            xs.append(clip[:CONTEXT]); ys.append(clip[CONTEXT]); acts.append(self.actions[k][t - 1])
        x = torch.from_numpy(np.stack(xs))                       # B x 4 x H x W x 3
        x = x.permute(0, 1, 4, 2, 3).reshape(len(rows), 3 * CONTEXT, x.shape[2], x.shape[3])
        y = torch.from_numpy(np.stack(ys)).permute(0, 3, 1, 2)
        a = torch.from_numpy(np.stack(acts))
        return x, a, y


def to_float(x, device):
    return x.to(device, non_blocking=True).float() / 255


def psnr(pred, target):
    mse = ((pred - target) ** 2).mean(dim=(1, 2, 3)).clamp_min(1e-10)
    return (10 * torch.log10(1.0 / mse)).mean().item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, default=ROOT / 'ext/poc-data/dream')
    p.add_argument('--maps', nargs='+', default=['MAP01', 'MAP02', 'MAP03'])
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--minutes', type=float, default=55)
    p.add_argument('--batch', type=int, default=48)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--width', type=int, default=48)
    p.add_argument('--eval-every', type=int, default=500)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    train, test = Data(a.data, a.maps, 'train'), Data(a.data, a.maps, 'test')
    print(f'train samples {len(train)}, test samples {len(test)}', flush=True)
    model = Dream(a.width).to(device)
    print('parameters', count_parameters(model), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    rng = np.random.default_rng(0)
    test_rows = test.index[rng.choice(len(test), min(1024, len(test)), replace=False)]
    log, t0, step = [], time.time(), 0
    best = -1
    deadline = t0 + 60 * a.minutes
    while time.time() < deadline:
        # linear warmup then cosine on wall time
        frac = (time.time() - t0) / (60 * a.minutes)
        lr = a.lr * min(1.0, step / 300) * (0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * frac)))
        for g in opt.param_groups:
            g['lr'] = lr
        rows = train.index[rng.integers(0, len(train), a.batch)]
        x, act, y = train.batch(rows)
        x, y, act = to_float(x, device), to_float(y, device), act.to(device).float()
        pred = model(x, act)
        loss = (pred - y).abs().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1
        if step % 50 == 0:
            log.append({'step': step, 'minutes': round((time.time() - t0) / 60, 2), 'loss': round(loss.item(), 5), 'lr': lr})
        if step % a.eval_every == 0:
            model.eval()
            with torch.no_grad():
                ps, ls = [], []
                for i in range(0, len(test_rows), 64):
                    x, act, y = test.batch(test_rows[i:i + 64])
                    x, y, act = to_float(x, device), to_float(y, device), act.to(device).float()
                    pred = model(x, act)
                    ps.append(psnr(pred, y)); ls.append((pred - y).abs().mean().item())
            model.train()
            test_psnr = float(np.mean(ps))
            log.append({'step': step, 'minutes': round((time.time() - t0) / 60, 2), 'test_l1': round(float(np.mean(ls)), 5), 'test_psnr': round(test_psnr, 3)})
            print(log[-1], flush=True)
            if test_psnr > best:
                best = test_psnr
                torch.save({'state': model.state_dict(), 'width': a.width, 'step': step, 'test_psnr': test_psnr}, a.out / 'model.pt')
            (a.out / 'log.json').write_text(json.dumps(log))
    torch.save({'state': model.state_dict(), 'width': a.width, 'step': step}, a.out / 'model_last.pt')
    (a.out / 'log.json').write_text(json.dumps(log))
    (a.out / 'train_summary.json').write_text(json.dumps({'steps': step, 'minutes': round((time.time() - t0) / 60, 1), 'batch': a.batch,
                                                          'parameters': count_parameters(model), 'best_test_psnr': best,
                                                          'train_samples': len(train), 'test_samples': len(test), 'maps': a.maps}, indent=1))
    print('done', step, 'steps, best test PSNR', round(best, 2), flush=True)


if __name__ == '__main__':
    main()
