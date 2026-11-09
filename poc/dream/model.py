"""A small action-conditioned next-frame model: 4 past frames (12 channels) plus the 7-button action, broadcast as
7 channels, through a conv U-Net; the output is the next frame as a residual on the last frame, in [0, 1].
Only ops the Neural Engine runs well: conv, batch norm (folded at export), SiLU, nearest upsample, concat."""
import torch
import torch.nn as nn
import torch.nn.functional as F

CONTEXT = 4
ACTIONS = 7


def block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.SiLU(),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.SiLU())


class Dream(nn.Module):
    def __init__(self, width=48):
        super().__init__()
        w = width
        self.enc1 = block(3 * CONTEXT + ACTIONS, w)        # 96x128
        self.enc2 = block(w, 2 * w)                         # 48x64
        self.enc3 = block(2 * w, 4 * w)                     # 24x32
        self.mid = block(4 * w, 8 * w)                      # 12x16
        self.dec3 = block(8 * w + 4 * w, 4 * w)
        self.dec2 = block(4 * w + 2 * w, 2 * w)
        self.dec1 = block(2 * w + w, w)
        self.out = nn.Conv2d(w, 3, 3, padding=1)
        self.act_emb = nn.Linear(ACTIONS, ACTIONS)         # a learned mixing of the buttons before broadcast

    def forward(self, frames, action):
        """frames: [B, 12, H, W] in [0,1] (oldest first); action: [B, 7] in {0,1}. Returns the next frame [B, 3, H, W]."""
        # broadcast by multiplication, so the trace carries no dynamic shape ints (Core ML export)
        a = self.act_emb(action)[:, :, None, None] * torch.ones_like(frames[:, :1])
        x = torch.cat([frames, a], 1)
        e1 = self.enc1(x)
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        e3 = self.enc3(F.avg_pool2d(e2, 2))
        m = self.mid(F.avg_pool2d(e3, 2))
        d3 = self.dec3(torch.cat([F.interpolate(m, scale_factor=2.0, mode='nearest'), e3], 1))
        d2 = self.dec2(torch.cat([F.interpolate(d3, scale_factor=2.0, mode='nearest'), e2], 1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, scale_factor=2.0, mode='nearest'), e1], 1))
        last = frames[:, -3:]
        return torch.clamp(last + self.out(d1), 0.0, 1.0)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == '__main__':
    m = Dream()
    print('parameters', count_parameters(m))
    y = m(torch.rand(2, 12, 96, 128), torch.zeros(2, 7))
    print(y.shape, y.min().item(), y.max().item())
