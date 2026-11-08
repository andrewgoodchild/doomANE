"""Train a pixel perception model: per-cell classes (monsters and items by name) from the screen alone.

The view above the status bar (collect_frames.py: 200x320) is resized to 192x320, and the model labels each
4x4 cell of it (48x80). Two encoders with the same small decoder:
- mobileclip: MobileCLIP-S2's image trunk (FastViT, open_clip weights), fine-tuned;
- tiny: a small CNN from scratch, to show what MobileCLIP's pretraining adds.

Validation maps are development maps left out of training; held-out maps are never in the data.

    ext/vision-venv/bin/python doom/vision/train_seg.py --data ext/vision-data/frames --backbone mobileclip --out results/vision/mobileclip

pix2: --data takes several frame directories (ext/vision-data/scenery adds views of empty scenery), and
--background-weight sets the loss weight of "nothing here" cells. In pix1 it was the class-balanced weight
clamped at 0.05, so a false monster cell cost a twentieth of a missed one, and wall screens became monsters.
pix3: --depth adds the depth buffer as a fourth input channel, read by a small CNN of its own whose features
join the encoder's at each scale (the pretrained trunk still sees only RGB).
pix4: --distractors gives sprites that are neither monsters nor items classes of their own (Gore: corpses
and gibs; Hazard: barrels; Decoration: decorations, vegetation and light sources), category "Distractor".
Before, they were trained as "nothing" and, being sprites, were called monsters. The runtime ignores them,
and the metrics count them as nothing.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

TARGET_CATEGORIES = ('Monster', 'Health', 'Armor', 'Ammo', 'Weapon', 'Key', 'Powerup')
VAL_MAPS = ('MAP01', 'MAP02', 'MAP03', 'MAP04')  # the maps the game comparison is played on
H, W = 192, 320
CELL = 4
MIN_PIXELS = 12      # instances smaller than this (at 200x320) are ignored by the loss: too small to name
MIN_NAME_COUNT = 30  # rarer names share their category's "Other<Category>" class
IGNORE = 255
DISTRACTOR_OF = {'Gore': 'Gore', 'Gibs': 'Gore', 'Hazard': 'Hazard',
                 'Decoration': 'Decoration', 'Vegetation': 'Decoration', 'LightSource': 'Decoration'}


def target_name(r):
    if r['category'] not in TARGET_CATEGORIES or r['name'].startswith('Dead'):
        return None
    return r['name']


def category_number(category, c):
    """1 + the index in TARGET_CATEGORIES of class c's category; 0 for 'none' and distractors."""
    return TARGET_CATEGORIES.index(category[c]) + 1 if c != 'none' and category[c] in TARGET_CATEGORIES else 0


def build_classes(tables, distractors=False):
    counts = {}
    for t in tables:
        for r in t:
            n = target_name(r)
            if n and r['pixels'] >= MIN_PIXELS:
                counts[(r['category'], n)] = counts.get((r['category'], n), 0) + 1
    classes = ['none']
    category = {}
    for (cat, n), c in sorted(counts.items()):
        if c >= MIN_NAME_COUNT:
            classes.append(n); category[n] = cat
    for cat in TARGET_CATEGORIES:
        classes.append('Other' + cat); category['Other' + cat] = cat
    if distractors:
        for d in dict.fromkeys(DISTRACTOR_OF.values()):
            classes.append(d); category[d] = 'Distractor'
    return classes, category, counts


def cell_targets(instances, table, index):
    """Per 4x4 cell: the class covering most of it if that covers >= 25%, IGNORE for tiny instances."""
    inst = torch.from_numpy(instances).long()
    inst = F.interpolate(inst[None, None].float(), size=(H, W), mode='nearest')[0, 0].long()
    lut = torch.zeros(256, dtype=torch.long)
    ignore = torch.zeros(256, dtype=torch.bool)
    for k, r in enumerate(table, 1):
        n = target_name(r)
        if n is None:
            d = DISTRACTOR_OF.get(r['category'])
            if d in index:
                lut[k] = index[d]
                ignore[k] = r['pixels'] < MIN_PIXELS
            continue
        lut[k] = index.get(n, index['Other' + r['category']])
        ignore[k] = r['pixels'] < MIN_PIXELS
    cls = lut[inst]
    ign = ignore[inst]
    C = max(index.values()) + 1
    onehot = F.one_hot(cls, C).permute(2, 0, 1).float()
    area = F.avg_pool2d(onehot[None], CELL)[0]           # C x 48 x 80
    area[0] = 0
    best_area, best = area.max(0)
    target = torch.where(best_area >= 0.25, best, torch.zeros_like(best))
    ign_cells = F.max_pool2d(ign[None, None].float(), CELL)[0, 0] > 0
    target[ign_cells & (target == 0)] = IGNORE
    return target.to(torch.uint8)


class Decoder(nn.Module):
    def __init__(self, channels, classes, width=96):
        super().__init__()
        self.lateral = nn.ModuleList(nn.Conv2d(c, width, 1) for c in channels)
        self.smooth = nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(),
                                    nn.Conv2d(width, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU())
        self.out = nn.Conv2d(width, classes, 1)

    def forward(self, feats):
        x = self.lateral[-1](feats[-1])
        for f, lat in zip(reversed(feats[:-1]), reversed(self.lateral[:-1])):
            x = F.interpolate(x, size=f.shape[-2:], mode='nearest') + lat(f)
        return self.out(self.smooth(x))


class Tiny(nn.Module):
    """A small CNN encoder from scratch with strides 4, 8, 16, 32."""
    channels = (32, 64, 128, 256)

    def __init__(self, inputs=3, channels=None):
        super().__init__()
        self.channels = channels or self.channels
        def block(i, o, s):
            return nn.Sequential(nn.Conv2d(i, o, 3, s, 1), nn.BatchNorm2d(o), nn.ReLU(), nn.Conv2d(o, o, 3, 1, 1), nn.BatchNorm2d(o), nn.ReLU())
        c = self.channels
        self.stem = block(inputs, c[0], 2)
        self.stages = nn.ModuleList([block(c[0], c[0], 2), block(c[0], c[1], 2), block(c[1], c[2], 2), block(c[2], c[3], 2)])

    def forward(self, x):
        x = self.stem(x)
        out = []
        for s in self.stages:
            x = s(x); out.append(x)
        return out


class MobileClipTrunk(nn.Module):
    channels = (80, 160, 320, 640)

    def __init__(self):
        super().__init__()
        import open_clip
        model, _, _ = open_clip.create_model_and_transforms('MobileCLIP-S2', pretrained='datacompdr')
        self.trunk = model.visual.trunk

    def forward(self, x):
        return self.trunk.forward_intermediates(x, intermediates_only=True)


class WithDepth(nn.Module):
    """The RGB encoder plus a small depth CNN; their features are concatenated at each scale."""
    def __init__(self, rgb):
        super().__init__()
        self.rgb = rgb
        self.depth = Tiny(inputs=1, channels=(16, 32, 64, 128))
        self.channels = tuple(a + b for a, b in zip(rgb.channels, self.depth.channels))

    def forward(self, x):
        return [torch.cat(pair, 1) for pair in zip(self.rgb(x[:, :3]), self.depth(x[:, 3:]))]


class Seg(nn.Module):
    def __init__(self, backbone, classes, depth=False):
        super().__init__()
        rgb = MobileClipTrunk() if backbone == 'mobileclip' else Tiny()
        self.encoder = WithDepth(rgb) if depth else rgb
        self.decoder = Decoder(self.encoder.channels, classes)

    def forward(self, x):  # x: B x 3 (or 4 with depth) x 192 x 320, RGB (and depth buffer) in 0..1
        return self.decoder(self.encoder(x))


def load(sources, depth=False):
    """sources: (directory, map) pairs. With depth, images get the depth buffer as a fourth channel."""
    images, targets, tables = [], [], []
    for data, m in sources:
        d = np.load(data / f'{m}.npz')
        t = json.loads((data / f'{m}.json').read_text())
        images.append(np.concatenate([d['images'], d['depth'][..., None]], -1) if depth else d['images']); tables += t
        targets.append(d['instances'])
    return np.concatenate(images), np.concatenate(targets), tables


def prepare(images, instances, tables, index):
    x = torch.from_numpy(images).permute(0, 3, 1, 2)  # uint8
    x = F.interpolate(x.float(), size=(H, W), mode='bilinear', align_corners=False).round().clamp(0, 255).to(torch.uint8).contiguous()  # not channels-last: MPS conv backward fails on it
    y = torch.stack([cell_targets(instances[i], tables[i], index) for i in range(len(tables))])
    return x, y


def evaluate(model, x, y, classes, category, device, batch=32):
    """Cell metrics and object metrics: an object is a connected group of cells of one target class."""
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            preds.append(model(x[i:i + batch].to(device).float() / 255).argmax(1).cpu())
    pred = torch.cat(preds)
    distractor = torch.tensor([c != 'none' and category[c] == 'Distractor' for c in classes])
    pred = torch.where(distractor[pred.long()], 0, pred)  # distractors are scored as nothing
    y = torch.where((y != IGNORE) & distractor[y.clamp(max=len(classes) - 1).long()], 0, y)
    valid = y != IGNORE
    fg = (y > 0) & valid
    out = {'cell_fg_recall': float(((pred == y) & fg).sum() / fg.sum()),
           'cell_fg_precision': float(((pred == y) & (pred > 0) & valid).sum() / ((pred > 0) & valid).sum().clamp(min=1)),
           'cell_category_recall': {}}
    cat_of = torch.tensor([category_number(category, c) for c in classes])
    pc, yc = cat_of[pred.long()], cat_of[y.clamp(max=len(classes) - 1).long()]
    for k, cat in enumerate(TARGET_CATEGORIES, 1):
        m = (yc == k) & fg
        if m.sum():
            out['cell_category_recall'][cat] = round(float(((pc == k) & m).sum() / m.sum()), 3)
    out['cell_category_precision'] = {cat: round(float(((pc == k) & (yc == k) & valid).sum() / ((pc == k) & valid).sum().clamp(min=1)), 3)
                                      for k, cat in enumerate(TARGET_CATEGORIES, 1) if ((pc == k) & valid).sum()}
    # Objects: a true object is a connected group of cells of one class; it is found when any of its cells is
    # predicted with the same category, and named right when its cells' most common predicted class is its class.
    yc = torch.where(valid, yc, torch.zeros_like(yc))
    found, total, named, false_blobs, pred_blobs, col_err = {}, {}, {}, 0, 0, []
    for i in range(len(y)):
        yi, pi, yci, pci = y[i].numpy(), pred[i].numpy(), yc[i].numpy(), pc[i].numpy()
        lab, n = blobs(np.where(valid[i].numpy(), yi, 0))
        for j in range(1, n + 1):
            m = lab == j
            k = int(yci[m][0])
            cat = TARGET_CATEGORIES[k - 1]
            hit = (pci[m] == k).any()
            total[cat] = total.get(cat, 0) + 1
            found[cat] = found.get(cat, 0) + int(hit)
            if hit:
                same = m & (pci == k)
                col_err.append(abs(np.where(m)[1].mean() - np.where(same)[1].mean()))
                votes = np.bincount(pi[same], minlength=len(classes))
                named[cat] = named.get(cat, 0) + int(votes.argmax() == yi[m][0])
        lab, n = blobs(pci)
        for j in range(1, n + 1):
            m = lab == j
            pred_blobs += 1
            false_blobs += int(not (yci[m] == pci[m][0]).any() and not (yi[m] == IGNORE).any())
    out['object_recall'] = {c: f'{found[c]}/{total[c]} = {found[c] / total[c]:.3f}' for c in total}
    out['object_named_right_when_found'] = {c: round(named.get(c, 0) / found[c], 3) for c in found if found[c]}
    out['object_false_blobs'] = f'{false_blobs}/{pred_blobs}'
    # 80 cells span the 90-degree field of view; column error converted near the centre of the view.
    out['object_column_error_degrees_median'] = round(float(np.median(col_err)) * 90 / 80, 2) if col_err else None
    model.train()
    return out


def blobs(a):
    """4-connected components of equal nonzero values (no scipy in this venv)."""
    lab = np.zeros(a.shape, np.int32)
    n = 0
    for r, c in zip(*np.nonzero(a)):
        if lab[r, c]:
            continue
        n += 1
        v = a[r, c]
        stack = [(r, c)]
        lab[r, c] = n
        while stack:
            i, j = stack.pop()
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                u, w = i + di, j + dj
                if 0 <= u < a.shape[0] and 0 <= w < a.shape[1] and not lab[u, w] and a[u, w] == v:
                    lab[u, w] = n
                    stack.append((u, w))
    return lab, n


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, nargs='+', required=True)
    p.add_argument('--backbone', choices=('mobileclip', 'tiny'), default='mobileclip')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--distractors', action='store_true', help='classes for sprites that are neither monsters nor items (pix4)')
    p.add_argument('--depth', action='store_true', help='the depth buffer as a fourth input channel (pix3)')
    p.add_argument('--background-weight', type=float, default=None, help='loss weight of empty cells (default: class-balanced, at least 0.05, as in pix1)')
    a = p.parse_args()
    torch.manual_seed(0)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    sources = [(d, f.stem) for d in a.data for f in sorted(d.glob('*.npz'))]
    train = [s for s in sources if s[1] not in VAL_MAPS]
    val = [s for s in sources if s[1] in VAL_MAPS]
    train_maps = sorted({m for _, m in train})
    val_maps = sorted({m for _, m in val})
    xi, xinst, xt = load(train, a.depth)
    vi, vinst, vt = load(val, a.depth)
    classes, category, counts = build_classes(xt, a.distractors)
    index = {c: i for i, c in enumerate(classes)}
    print(len(classes), 'classes:', classes, flush=True)
    x, y = prepare(xi, xinst, xt, index)
    vx, vy = prepare(vi, vinst, vt, index)
    print('train', tuple(x.shape), 'val', tuple(vx.shape), flush=True)
    freq = torch.bincount(y[y != IGNORE].long().flatten(), minlength=len(classes)).float()
    weight = (freq.sum() / freq.clamp(min=1)).pow(0.5)
    weight = (weight / weight[1:].mean()).clamp(max=10).to(device)
    weight[0] = weight[0].clamp(min=0.05) if a.background_weight is None else a.background_weight
    model = Seg(a.backbone, len(classes), a.depth).to(device)
    lr = a.lr or (3e-4 if a.backbone == 'mobileclip' else 1e-3)
    encoder = model.encoder.rgb if a.depth else model.encoder
    new = [*model.decoder.parameters(), *(model.encoder.depth.parameters() if a.depth else [])]  # from scratch: the higher rate
    groups = [{'params': encoder.parameters(), 'lr': lr}, {'params': new, 'lr': lr * 3}]
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    steps = a.epochs * math.ceil(len(x) / a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[lr, lr * 3], total_steps=steps, pct_start=0.1)
    a.out.mkdir(parents=True, exist_ok=True)
    history = []
    t0 = time.time()
    for epoch in range(a.epochs):
        order = torch.randperm(len(x))
        total = 0.0
        for i in range(0, len(x), a.batch):
            b = order[i:i + a.batch]
            xb = x[b].to(device).float() / 255
            yb = y[b].to(device).long()
            if torch.rand(1) < 0.5:
                xb, yb = xb.flip(-1), yb.flip(-1)
            xb[:, :3] = (xb[:, :3] * (0.6 + 0.8 * torch.rand(len(b), 1, 1, 1, device=device))).clamp(0, 1)  # lighting (not depth)
            logits = model(xb)
            loss = F.cross_entropy(logits.permute(0, 2, 3, 1).reshape(-1, logits.shape[1]), yb.reshape(-1), weight=weight, ignore_index=IGNORE)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            total += float(loss) * len(b)
        row = {'epoch': epoch + 1, 'loss': round(total / len(x), 4), 'minutes': round((time.time() - t0) / 60, 1)}
        if (epoch + 1) % 5 == 0 or epoch + 1 == a.epochs:
            row['val'] = evaluate(model, vx, vy, classes, category, device)
        history.append(row)
        print(json.dumps(row), flush=True)
    torch.save({'state': model.state_dict(), 'classes': classes, 'category': category, 'backbone': a.backbone, 'depth': a.depth,
                'train_maps': train_maps, 'val_maps': val_maps}, a.out / 'model.pt')
    (a.out / 'report.json').write_text(json.dumps({'backbone': a.backbone, 'classes': classes, 'name_counts': {f'{k[0]}/{k[1]}': v for k, v in sorted(counts.items())},
                                                   'train_maps': train_maps, 'val_maps': val_maps,
                                                   'data': [str(d) for d in a.data], 'background_weight': float(weight[0]), 'history': history}, indent=1))


if __name__ == '__main__':
    main()
