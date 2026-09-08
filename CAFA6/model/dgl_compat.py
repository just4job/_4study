"""Backward-compatible entry; use model.dgl_patch.ensure_dgl_importable()."""
from model.dgl_patch import ensure_dgl_importable

apply_dgl_compat = ensure_dgl_importable
