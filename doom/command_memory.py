"""Short memory for doomLaya's command decision, as extra inputs to its numeric residual.

The server sees every command request in order. From the request text alone it keeps the
last WINDOW command contexts ("Current command: use_switch #332 ...; status executing") and
derives five observed features:

    missing_key_reachable a key of a colour not in "Collected keys" is in "Reachable items"
                          (the mechanism that needs it is often not listed while the key is
                          reachable, so the route requirement itself cannot be read here)
    current_repeats       share of the window spent on the command target that is current now
    distinct_mechanisms   distinct use_switch targets in the window / 5
    switch_share          share of the window spent on use_switch
    blocked_share         share of the window whose command status was blocked or unavailable

`MemoryResidual` is doomLaya's v2 command residual with these five columns appended to its
first layer. Built from a parent with `from_parent`, the new columns are zero, so it returns
exactly the parent's logits until it is trained (`doom/repair_command_residual.py`).
"""
from __future__ import annotations

import re
from collections import deque

WINDOW = 60  # about 30 s at doomLaya's 0.5 s decision interval
NAMES = ("missing_key_reachable", "current_repeats", "distinct_mechanisms", "switch_share", "blocked_share")
KEY_ITEMS = {"red": ("RedCard", "RedSkull"), "blue": ("BlueCard", "BlueSkull"), "yellow": ("YellowCard", "YellowSkull")}


def command_context(state):
    m = re.search(r"^Current command: (\w+)(?: #([^ ;]+))?.*?; status (\w+)", state, re.M)
    return (m[1], m[2], m[3]) if m else ("none", None, "waiting")


def missing_key_reachable(state):
    keys = re.search(r"^Collected keys: (.*)$", state, re.M)
    reach = re.search(r"^Reachable items: (.*)$", state, re.M)
    if not keys or not reach:
        return 0.0
    missing = [c for c in KEY_ITEMS if c not in keys[1]]
    return float(any(name + "#" in reach[1] for color in missing for name in KEY_ITEMS[color]))


class Memory:
    def __init__(self, window=WINDOW):
        self.history = deque(maxlen=window)

    def reset(self):
        self.history.clear()

    def features(self, state):
        """Features for this request from the requests before it; then remember this one."""
        action, target, status = command_context(state)
        h = list(self.history)
        n = max(1, len(h))
        values = [missing_key_reachable(state),
                  sum(1 for a, t, _ in h if (a, t) == (action, target) and a != "none") / n,
                  len({t for a, t, _ in h if a == "use_switch"}) / 5,
                  sum(1 for a, _, _ in h if a == "use_switch") / n,
                  sum(1 for _, _, s in h if s in ("blocked", "unavailable")) / n]
        self.history.append((action, target, status))
        return values


def make_memory_residual(parent):
    """A copy of a doomLaya NumericResidual whose forward also takes the memory features."""
    import copy
    import torch

    class MemoryResidual(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.parent = copy.deepcopy(parent)
            first = self.parent.network[0]
            self.memory = torch.nn.Linear(len(NAMES), first.out_features, bias=False)
            torch.nn.init.zeros_(self.memory.weight)

        def forward(self, observed, semantic, memory):
            p = self.parent
            basis = (observed[:, p.basis_indices] >= p.basis_thresholds).to(observed.dtype)
            inputs = torch.cat([observed, basis, semantic / 10], dim=-1)
            h = p.network[0](inputs) + self.memory(memory)
            out = p.network[1:](h)
            return semantic * getattr(p, "semantic_scale", 1.0) + out

    return MemoryResidual()
