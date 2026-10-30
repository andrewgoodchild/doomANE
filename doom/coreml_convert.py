"""Core ML conversion helpers for a ModernBERT encoder (from the companion ane-decision-model scouting run).

Fixed-shape export: explicit additive masks instead of transformers' masking utilities, and converters for the two
aten ops coremltools 9.0 lacks. Used by doom/export_doom_encoder.py.
"""
from __future__ import annotations

import torch


MASK_VALUE = -1e4  # fp16-safe additive mask (ANE has no fp32); -inf/finfo.min can NaN on the ANE


class EncoderWrapper(torch.nn.Module):
    """input_ids + attention_mask -> last hidden state, fixed shape.

    Builds the additive attention masks itself instead of letting transformers v5's
    masking utilities do it (those emit new_ones(()) scalars that Core ML cannot convert).
    ModernBERT takes a dict {full_attention, sliding_attention} in transformers v5; in 4.x it takes the 2D
    mask and builds both 4D masks in `_update_attention_mask`, which is replaced here with the same
    arithmetic. BERT-family models take a 4D mask.
    """

    def __init__(self, model, length: int):
        super().__init__()
        self.model = model
        cfg = model.config
        self.is_modernbert = cfg.model_type == "modernbert"
        self.length = length
        if self.is_modernbert:
            half = getattr(cfg, "sliding_window", None) or cfg.local_attention // 2
            idx = torch.arange(length)
            ok = (idx[None, :] - idx[:, None]).abs() <= half
            window_add = torch.where(ok, torch.tensor(0.0), torch.tensor(MASK_VALUE)).to(torch.float32)[None, None]
            self.register_buffer("window_add", window_add, persistent=False)  # [1,1,L,L] additive
            self.legacy_masks = hasattr(model, "_update_attention_mask")  # transformers 4.x
            if self.legacy_masks:
                model._update_attention_mask = lambda mask, output_attentions=False: self.masks(mask)

    def masks(self, attention_mask):
        # arithmetic only (no boolean ops): 0 for real keys, MASK_VALUE for padding keys
        pad = (1.0 - attention_mask.to(torch.float32)) * MASK_VALUE          # [1,L]
        full = pad[:, None, None, :].expand(-1, -1, self.length, -1)          # [1,1,L,L]
        return full, (full + self.window_add if self.is_modernbert else None)

    def forward(self, input_ids, attention_mask):
        full, sliding = self.masks(attention_mask)
        if self.is_modernbert and self.legacy_masks:
            out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        elif self.is_modernbert:
            out = self.model(input_ids=input_ids, attention_mask={"full_attention": full, "sliding_attention": sliding})
        else:
            out = self.model(input_ids=input_ids, attention_mask=full)
        return out.last_hidden_state


def register_missing_torch_ops() -> None:
    """coremltools 9.0 has no converter for aten::new_ones (ModernBERT's mask code uses it)."""
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.frontend.torch.ops import _get_inputs
    from coremltools.converters.mil.frontend.torch.torch_op_registry import (
        _TORCH_OPS_REGISTRY, register_torch_op)

    if "alias" not in _TORCH_OPS_REGISTRY:
        @register_torch_op
        def alias(context, node):  # aten.alias is a view no-op (BERT position-id slicing at full length)
            inputs = _get_inputs(context, node, expected=1)
            context.add(mb.identity(x=inputs[0], name=node.name))

    if "new_ones" in _TORCH_OPS_REGISTRY:
        return

    @register_torch_op
    def new_ones(context, node):
        inputs = _get_inputs(context, node)
        shape = inputs[1]
        if isinstance(shape, list):
            shape = mb.concat(values=shape, axis=0)
        context.add(mb.fill(shape=shape, value=1.0, name=node.name))
