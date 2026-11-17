"""
RoadAnnoter
===========
Semantic segmentation annotation tool for road scenes.

Developer: Nazmul Haque

Features
  - Polygon annotations stored in YOLO-seg format
  - SAM / MobileSAM click-to-segment and Smart Select
  - Edge Fill annotator (edge detection + colour flood-fill, no model needed)
  - Pretrained suggestions (SegFormer Cityscapes, torchvision COCO models)
  - Light segmentation model training and suggestions from your own model
  - Undo/redo, multi-class, project settings persistence

Dependencies
------------
    pip install PyQt5 opencv-python numpy pillow
    pip install torch torchvision          # training + pretrained suggestions
    pip install transformers               # SegFormer Cityscapes suggestions
    pip install git+https://github.com/facebookresearch/segment-anything.git   # SAM
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
import tempfile
import threading
import time
import traceback
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import List, Optional, Tuple
import warnings

# ── App identity ────────────────────────────────────────────────────────────
APP_NAME      = "RoadAnnoter"
APP_VERSION   = "1.0"
APP_DEVELOPER = "Nazmul Haque"

# Harmless start-up notices from timm / mobile_sam (old import paths, and
# MobileSAM re-registering its TinyViT variants). Silenced so the console
# stays readable; nothing else is filtered.
warnings.filterwarnings("ignore", category=FutureWarning,
                        message=r"Importing from timm\..*is deprecated")
warnings.filterwarnings("ignore", category=UserWarning,
                        message=r"Overwriting tiny_vit_.* in registry")

# ── stdlib XML ──────────────────────────────────────────────────────────────
import xml.etree.ElementTree as ET

# ── numpy / cv2 ─────────────────────────────────────────────────────────────
try:
    import numpy as np
except ImportError:
    print("pip install numpy"); raise SystemExit(1)

try:
    import cv2
except ImportError:
    print("pip install opencv-python"); raise SystemExit(1)

# ── PyQt5 ───────────────────────────────────────────────────────────────────
try:
    from PyQt5.QtCore import (
        QByteArray, QEvent, QPoint, QPointF, QRectF, QSettings,
        Qt, QThread, QTimer, pyqtSignal, QObject,
    )
    from PyQt5.QtGui import (
        QColor, QCursor, QFont, QImage, QKeySequence, QPainter,
        QPainterPath, QPen, QPixmap, QPolygonF,
    )
    from PyQt5.QtWidgets import (
        QAbstractItemView, QAction, QApplication, QCheckBox, QComboBox,
        QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog,
        QFormLayout, QFrame, QGroupBox, QHBoxLayout, QInputDialog,
        QLabel, QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
        QPushButton, QProgressBar, QProgressDialog, QScrollArea,
        QShortcut, QSizePolicy, QSpinBox, QSplitter, QStatusBar,
        QTabWidget, QTextEdit, QVBoxLayout, QWidget,
    )
except ImportError:
    print("pip install PyQt5"); raise SystemExit(1)

# ── torch (optional – needed for training & supervised suggestions) ─────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms.functional as TF
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[INFO] torch not found – training and supervised suggestions disabled.")

# ════════════════════════════════════════════════════════════════════════════
# SAM – self-contained loader
# ════════════════════════════════════════════════════════════════════════════
#
# Design goal: work with EITHER checkpoint type using ONLY what the user has
# already installed, with zero extra pip installs.
#
#   Checkpoint           Required package     Arch
#   ─────────────────    ─────────────────    ──────────────────────────
#   mobile_sam.pt        segment_anything     TinyViT (built-in below)
#   sam_vit_b/l/h.pth    segment_anything     original SAM ViT
#
# MobileSAM TinyViT encoder is implemented here directly in pure PyTorch so
# that segment_anything's Sam shell + SamPredictor can wrap it without any
# extra package.  The weights load cleanly because the key names match the
# official MobileSAM checkpoint exactly.
#
# Discriminator: original SAM checkpoints contain "attn.rel_pos_h" keys;
# MobileSAM TinyViT checkpoints do NOT.
# ════════════════════════════════════════════════════════════════════════════

SAM_AVAILABLE   = False
SAM_REGISTRIES: dict = {}   # lib_name → (registry_dict, SamPredictor_class)

# ── try mobile_sam package first (fastest path if installed from GitHub) ────
try:
    import mobile_sam as _mobile_sam_lib
    SAM_REGISTRIES["mobile_sam"] = (
        _mobile_sam_lib.sam_model_registry,
        _mobile_sam_lib.SamPredictor,
    )
    SAM_AVAILABLE = True
except ImportError:
    pass

# ── try segment_anything (pip install segment-anything) ─────────────────────
try:
    import segment_anything as _sam_lib
    SAM_REGISTRIES["segment_anything"] = (
        _sam_lib.sam_model_registry,
        _sam_lib.SamPredictor,
    )
    SAM_AVAILABLE = True
except ImportError:
    pass

if not SAM_AVAILABLE:
    print("[INFO] SAM not found.\n"
          "       pip install segment-anything   # works for BOTH checkpoint types")


# ════════════════════════════════════════════════════════════════════════════
# Built-in MobileSAM loader  (no mobile_sam package required)
# ════════════════════════════════════════════════════════════════════════════
# Rather than re-implementing TinyViT from scratch with hardcoded dimensions
# (which breaks when the checkpoint differs from the assumed variant), we use
# a "shape-first" strategy:
#
#   1. Load the raw state-dict.
#   2. Read every tensor shape we need directly from the checkpoint.
#   3. Build the nn.Module tree to exactly match those shapes.
#   4. Load with strict=True — zero mismatches guaranteed.
#
# This handles ALL TinyViT variants (5M, 11M, 21M, any future size) because
# we never assume a specific embed_dim value.
# ════════════════════════════════════════════════════════════════════════════

if TORCH_AVAILABLE:

    # ── tiny building blocks ─────────────────────────────────────────────────

    class _CB(nn.Sequential):
        """Conv2d + BN (no bias)."""
        def __init__(self, ic, oc, k=1, s=1, p=0, g=1, w=1):
            super().__init__()
            self.add_module("c", nn.Conv2d(ic, oc, k, s, p, groups=g, bias=False))
            bn = nn.BatchNorm2d(oc)
            nn.init.constant_(bn.weight, w)
            nn.init.constant_(bn.bias, 0)
            self.add_module("bn", bn)

    class _PatchEmbed(nn.Module):
        """Two-strided-conv patch embed: 3 → C0 at stride 4.
        Named 'seq' to match MobileSAM checkpoint keys:
          image_encoder.patch_embed.seq.0.c.weight  (3→C0//2)
          image_encoder.patch_embed.seq.2.c.weight  (C0//2→C0)
        """
        def __init__(self, c0: int):
            super().__init__()
            self.seq = nn.Sequential(
                _CB(3, c0 // 2, 3, 2, 1),   # index 0
                nn.GELU(),                    # index 1
                _CB(c0 // 2, c0, 3, 2, 1),  # index 2
            )
        def forward(self, x): return self.seq(x)   # (B,C0,H/4,W/4)

    class _PatchMerging(nn.Module):
        """Downsample: (B,L,Ci) → (B,L/4,Co).
        H and W are inferred dynamically from the input tensor so that
        the stored res is only used as a hint for out_res computation.
        """
        def __init__(self, res, ci, co):
            super().__init__()
            self.res = res          # hint; actual H,W inferred in forward
            self.out_res = (res[0]//2, res[1]//2)
            self.act = nn.GELU()
            self.conv1 = _CB(ci, co)
            self.conv2 = _CB(co, co, 3, 2, 1, g=co)
            self.conv3 = _CB(co, co, w=0)
        def forward(self, x):
            B, L, C = x.shape
            # Infer H,W from L and the stored aspect ratio (always square for SAM)
            import math
            H = W = int(math.isqrt(L))
            assert H * W == L, f"_PatchMerging: L={L} is not a perfect square"
            x = x.view(B, H, W, C).permute(0,3,1,2)
            x = self.conv3(self.conv2(self.act(self.conv1(x))))
            B, C2, H2, W2 = x.shape
            return x.permute(0,2,3,1).reshape(B, H2*W2, C2)

    class _LWA(nn.Module):
        """Local-window attention (TinyViT style)."""
        def __init__(self, dim, key_dim, num_heads, win):
            super().__init__()
            self.nh = num_heads; self.kd = key_dim; self.scale = key_dim**-0.5
            self.d  = key_dim   # attn_ratio=1
            self.dh = key_dim * num_heads
            self.norm = nn.LayerNorm(dim)
            self.qkv  = nn.Linear(dim, (key_dim*2 + self.d) * num_heads)
            self.proj = nn.Linear(self.dh, dim)
            pts = [(i,j) for i in range(win) for j in range(win)]
            N = len(pts)
            bias = torch.zeros(N, N)
            for a,(r1,c1) in enumerate(pts):
                for b,(r2,c2) in enumerate(pts):
                    bias[a,b] = abs(r1-r2)+abs(c1-c2)
            self.register_buffer("ab", bias)
        def forward(self, x):
            B, N, C = x.shape
            q,k,v = self.qkv(self.norm(x)).split(
                [self.nh*self.kd, self.nh*self.kd, self.dh], dim=2)
            def rsh(t, d):
                return t.reshape(B,N,self.nh,d).permute(0,2,1,3)
            q,k,v = rsh(q,self.kd), rsh(k,self.kd), rsh(v,self.d)
            attn = (q@k.transpose(-2,-1))*self.scale
            attn = (attn + self.ab[:N,:N]).softmax(-1)
            return x + self.proj((attn@v).transpose(1,2).reshape(B,N,self.dh))

    class _MBConv(nn.Module):
        """MBConv block used in early TinyViT stages.

        Some variants also include a local_conv (depthwise conv on the sequence)
        that transformer blocks have — we detect this from checkpoint keys and
        add it when present so key names match exactly.

        shortcut_type:
          'identity'   → nn.Identity()
          'cb'         → _CB(dim, dim)
          'avgpool_cb' → Sequential[AvgPool2d, _CB]
        has_local_conv:
          True  → adds self.local_conv = _CB(dim, dim, 3,1,1,g=dim)
          False → no local_conv submodule
        """
        def __init__(self, in_dim, res, out_dim=None, mid_dim=None,
                     shortcut_type="identity", has_local_conv=False):
            super().__init__()
            self.res = res
            self._has_local_conv = has_local_conv
            out_dim = out_dim if out_dim is not None else in_dim
            self._out_dim = out_dim
            mid = mid_dim if mid_dim is not None else int(in_dim * 4.0)
            self.conv1 = _CB(in_dim, mid)
            self.conv2 = _CB(mid, mid, 3, 1, 1, g=mid)
            self.conv3 = _CB(mid, out_dim, w=0)
            self.act   = nn.GELU()
            # Shortcut / residual connection
            self._use_residual = True
            if shortcut_type == "avgpool_cb":
                self.shortcut = nn.Sequential(nn.AvgPool2d(1, 1), _CB(in_dim, out_dim, w=0))
            elif shortcut_type == "cb":
                self.shortcut = _CB(in_dim, out_dim, w=0)
            elif in_dim == out_dim:
                # Pure identity residual — no extra weights
                self.shortcut = nn.Identity()
            else:
                # Dims differ and no shortcut specified → no residual
                # (the checkpoint has no shortcut keys for this block)
                self._use_residual = False
            if has_local_conv:
                self.local_conv = _CB(out_dim, out_dim, 3, 1, 1, g=out_dim)

        def forward(self, x):
            import math
            B, L, C = x.shape
            H = W = int(math.isqrt(L))
            spatial = x.view(B, H, W, C).permute(0, 3, 1, 2)
            out = self.conv3(self.act(self.conv2(self.act(self.conv1(spatial)))))
            if self._use_residual:
                out = self.act(out + self.shortcut(spatial))
            else:
                out = self.act(out)
            if self._has_local_conv:
                Co = out.shape[1]
                lc = out.permute(0, 2, 3, 1).reshape(B, H * W, Co)
                lc = self.local_conv(lc.transpose(1, 2).reshape(B, Co, H, W))
                out = out + lc
            return out.permute(0, 2, 3, 1).reshape(B, H * W, self._out_dim)

    def _detect_mbconv_props(state: dict, si: int) -> dict:
        """
        Detect MBConv block properties for stage si from checkpoint keys.
        Returns dict with:
          shortcut_type : 'identity' | 'cb' | 'avgpool_cb'
          has_local_conv: bool
        """
        blk_prefix = f"image_encoder.layers.{si}.blocks.0."
        blk_keys   = [k[len(blk_prefix):] for k in state if k.startswith(blk_prefix)]

        # shortcut type
        sc_keys = [k for k in blk_keys if k.startswith("shortcut")]
        if not sc_keys:
            shortcut_type = "identity"
        elif any(".1." in k for k in sc_keys):
            shortcut_type = "avgpool_cb"
        elif any(".c." in k for k in sc_keys):
            shortcut_type = "cb"
        else:
            shortcut_type = "identity"

        # local_conv presence
        has_local_conv = any(k.startswith("local_conv.") for k in blk_keys)

        return {"shortcut_type": shortcut_type, "has_local_conv": has_local_conv}

    def _detect_tvblock_props(state: dict, si: int) -> dict:
        """
        Scan checkpoint keys for stage si's first transformer block and return
        the naming conventions actually used, so our module tree matches exactly.

        Returns:
          mlp_style   : 'sequential' (mlp.0/mlp.2) | 'fc'  (mlp.fc1/mlp.fc2)
          has_norm1   : bool  (separate norm1 / norm2 submodules exist)
          has_local_conv: bool
        """
        pfx  = f"image_encoder.layers.{si}.blocks.0."
        keys = [k[len(pfx):] for k in state if k.startswith(pfx)]

        # MLP naming
        has_fc  = any(k.startswith("mlp.fc") for k in keys)
        mlp_style = "fc" if has_fc else "sequential"

        # norm1/norm2 as separate submodules
        has_norm1 = "norm1.weight" in keys

        # local_conv
        has_local_conv = any(k.startswith("local_conv.") for k in keys)

        return {
            "mlp_style":      mlp_style,
            "has_norm1":      has_norm1,
            "has_local_conv": has_local_conv,
        }

    class _MLP(nn.Module):
        """Two-layer MLP with GELU.  style='sequential' → fc indexed 0/2;
        style='fc' → fc1/fc2 named submodules.  Both produce identical output."""
        def __init__(self, dim: int, mid: int, style: str = "sequential"):
            super().__init__()
            self._style = style
            if style == "fc":
                self.fc1 = nn.Linear(dim, mid)
                self.fc2 = nn.Linear(mid, dim)
                self.act = nn.GELU()
            else:
                # nn.Sequential gives integer-indexed keys: 0, 1 (GELU), 2
                self._seq = nn.Sequential(
                    nn.Linear(dim, mid), nn.GELU(), nn.Linear(mid, dim))
                # expose via nn.ModuleDict with the same attribute name so that
                # state_dict key is "mlp.0" / "mlp.2" when parent is "mlp"
                # We just store as _seq but register with numeric names.
                # Actually: nn.Sequential already gives keys "0","1","2".
                # To get model keys "mlp.0.weight" we need self to BE the Sequential
                # but we can't subclass both.  Instead store directly:
                self.add_module("0", self._seq[0])
                self.add_module("1", self._seq[1])   # GELU has no params
                self.add_module("2", self._seq[2])
                del self._seq

        def forward(self, x):
            if self._style == "fc":
                return self.fc2(self.act(self.fc1(x)))
            else:
                return self._modules["2"](self._modules["1"](self._modules["0"](x)))

    class _TVBlock(nn.Module):
        """
        TinyViT transformer block.  Naming adapts to checkpoint conventions:

        mlp_style='sequential': MLP keys are  mlp.0.weight / mlp.2.weight
        mlp_style='fc':         MLP keys are  mlp.fc1.weight / mlp.fc2.weight
        has_norm1=True:         standalone norm1/norm2 submodules exist
        has_norm1=False:        norms are pre-norm inside attn / mlp; we skip
                                adding norm1/norm2 so no extra keys appear
        has_local_conv:         local_conv depthwise after attention
        """
        def __init__(self, dim, res, num_heads, win, mlp_ratio=4.,
                     mlp_style="sequential", has_norm1=True, has_local_conv=True):
            super().__init__()
            self.res = res; self.win = win
            self._has_norm1      = has_norm1
            self._has_local_conv = has_local_conv
            kd  = dim // max(num_heads, 1)
            mid = int(dim * mlp_ratio)

            if has_norm1:
                self.norm1 = nn.LayerNorm(dim)
                self.norm2 = nn.LayerNorm(dim)
            self.attn      = _LWA(dim, kd, num_heads, win)
            self.mlp       = _MLP(dim, mid, style=mlp_style)
            if has_local_conv:
                self.local_conv = _CB(dim, dim, 3, 1, 1, g=dim)

        def forward(self, x):
            import math
            B, L, C = x.shape
            H = W = int(math.isqrt(L))
            # windowed attention
            res_x = x
            ax = self.norm1(x) if self._has_norm1 else x
            ax = ax.view(B, H, W, C)
            pb = (self.win - H % self.win) % self.win
            pr = (self.win - W % self.win) % self.win
            ax = F.pad(ax, (0,0,0,pr,0,pb))
            Hp, Wp = H+pb, W+pr
            nH, nW = Hp//self.win, Wp//self.win
            ax = ax.view(B,nH,self.win,nW,self.win,C)\
                   .permute(0,1,3,2,4,5).reshape(B*nH*nW, self.win*self.win, C)
            ax = self.attn(ax)
            ax = ax.view(B,nH,nW,self.win,self.win,C)\
                   .permute(0,1,3,2,4,5).reshape(B,Hp,Wp,C)
            if pb or pr: ax = ax[:,:H,:W].contiguous()
            ax = ax.view(B, L, C)
            if self._has_local_conv:
                lc = ax.transpose(1,2).reshape(B,C,H,W)
                lc = self.local_conv(lc).view(B,C,L).transpose(1,2)
                x  = res_x + lc
            else:
                x  = res_x + ax
            mx = self.norm2(x) if self._has_norm1 else x
            return x + self.mlp(mx)

    class _TVStage(nn.Module):
        def __init__(self, ci, co, res, depth, num_heads, win, do_down,
                     block_dim=None, mbconv_out_dim=None, mbconv_mid_dim=None,
                     use_mbconv=False, shortcut_type="identity",
                     has_local_conv=False, tvblock_props=None,
                     down_after_blocks=False):
            """
            down_after_blocks: when True, apply PatchMerging AFTER the blocks
            (used for stage 0 where MBConv blocks run at the patch_embed resolution
            and the downsample follows them, unlike stage 1+ where downsample
            is applied first to change resolution before the transformer blocks).
            """
            super().__init__()
            self._down_after = down_after_blocks
            if do_down:
                # block resolution = res (before downsample) if down_after_blocks,
                # else res//2 (after downsample if placed before blocks)
                block_res = res if down_after_blocks else (res[0]//2, res[1]//2)
                self.downsample = _PatchMerging(res, ci, co)
            else:
                block_res = res
                self.downsample = None
            bdim   = block_dim if block_dim is not None else co
            mb_out = mbconv_out_dim if mbconv_out_dim is not None else bdim
            mb_mid = mbconv_mid_dim
            if use_mbconv:
                self.blocks = nn.ModuleList([
                    _MBConv(bdim, block_res, out_dim=mb_out, mid_dim=mb_mid,
                            shortcut_type=shortcut_type,
                            has_local_conv=has_local_conv)
                    for _ in range(depth)])
            else:
                tp = tvblock_props or {}
                self.blocks = nn.ModuleList([
                    _TVBlock(bdim, block_res, num_heads, win,
                             mlp_style=tp.get("mlp_style", "sequential"),
                             has_norm1=tp.get("has_norm1", True),
                             has_local_conv=tp.get("has_local_conv", True))
                    for _ in range(depth)])

        def forward(self, x):
            if self._down_after:
                for b in self.blocks: x = b(x)
                if self.downsample: x = self.downsample(x)
            else:
                if self.downsample: x = self.downsample(x)
                for b in self.blocks: x = b(x)
            return x

    class _TinyViTEncoder(nn.Module):
        """
        Shape-driven TinyViT encoder for MobileSAM.
        All dimensions are read from the checkpoint, not hardcoded.
        Output: (B, 256, img_size//16, img_size//16)
        """
        def __init__(self, cfg: dict):
            """cfg keys: img_size, embed_dims, depths, num_heads, window_sizes"""
            super().__init__()
            ed   = cfg["embed_dims"]    # e.g. [64,128,160,320]
            deps = cfg["depths"]        # e.g. [2,2,6,2]
            nhs  = cfg["num_heads"]     # e.g. [2,4,5,10]
            wins = cfg["window_sizes"]  # e.g. [7,7,14,7]
            isz  = cfg["img_size"]      # 1024

            H = isz // 4; W = isz // 4

            down_ci         = cfg.get("downsample_ci", {})
            blk_dims        = cfg.get("block_dims", {})
            mb_out_dims     = cfg.get("mbconv_out_dims", {})
            mb_mid_dims     = cfg.get("mbconv_mid_dims", {})
            mbconv_map      = cfg.get("is_mbconv", {})
            shortcut_map    = cfg.get("shortcut_types", {})
            local_conv_map  = cfg.get("has_local_convs", {})
            tvb_props_map   = cfg.get("tvblock_props", {})

            # patch_embed output = stage 0 block input dim = block_dims[0].
            # This differs from embed_dims[0] when stage 0 has its own downsample
            # (e.g. block_dims[0]=64 but embed_dims[0]=128 after the downsample).
            patch_embed_dim = blk_dims.get(0, ed[0])
            self.patch_embed = _PatchEmbed(patch_embed_dim)

            # cur_ch tracks the actual channel count flowing between stages.
            cur_ch = patch_embed_dim

            self.layers = nn.ModuleList()
            for i in range(len(deps)):
                # Gate downsample on checkpoint having downsample keys for this stage
                # (applies to ALL stages including 0 — some variants downsample after stage 0)
                has_down = i in down_ci
                ci     = down_ci.get(i, cur_ch)
                co     = ed[i]
                use_mb = mbconv_map.get(i, False)
                # In this architecture ALL stages run blocks BEFORE the downsample.
                # bdim = what the blocks in this stage actually use = cur_ch
                # (= block_dims[i] = down_ci[i], all the same value).
                bdim   = blk_dims.get(i, cur_ch)
                mb_out = mb_out_dims.get(i, bdim)
                mb_mid = mb_mid_dims.get(i, None)
                stype  = shortcut_map.get(i, "identity")
                hlc    = local_conv_map.get(i, False)
                tp     = tvb_props_map.get(i, {})
                # Downsample always comes AFTER blocks in this checkpoint
                down_after = has_down

                self.layers.append(
                    _TVStage(ci, co, (H,W), deps[i], nhs[i], wins[i],
                             do_down=has_down,
                             block_dim=bdim,
                             mbconv_out_dim=mb_out,
                             mbconv_mid_dim=mb_mid,
                             use_mbconv=use_mb,
                             shortcut_type=stype,
                             has_local_conv=hlc,
                             tvblock_props=tp,
                             down_after_blocks=down_after))
                if has_down:
                    cur_ch = co
                    H //= 2; W //= 2
                elif use_mb:
                    cur_ch = mb_out
                else:
                    cur_ch = bdim

            # neck input = what the last stage's blocks output.
            # cfg["neck_in"] is read directly from neck.0.weight shape[1]
            # in the checkpoint — the most authoritative source.
            neck_in = (cfg.get("neck_in") or
                       blk_dims.get(len(deps)-1) or
                       ed[-1])
            # MobileSAM neck LayerNorm weights are stored as (256,) in the
            # checkpoint — keys: neck.1.weight, neck.1.bias (and neck.3.*).
            # We need a module whose OWN .weight/.bias are those parameters
            # (not nested under a sub-module) so the key paths match exactly.
            class _LN2d(nn.Module):
                """LayerNorm applied channel-wise to (B,C,H,W).
                Exposes .weight and .bias directly so state_dict keys match
                the MobileSAM checkpoint: neck.1.weight / neck.1.bias."""
                def __init__(self, c: int):
                    super().__init__()
                    self.weight = nn.Parameter(torch.ones(c))
                    self.bias   = nn.Parameter(torch.zeros(c))
                    self.eps    = 1e-6
                def forward(self, x):
                    # x: (B,C,H,W)
                    x = x.permute(0,2,3,1)          # (B,H,W,C)
                    x = F.layer_norm(x, (x.shape[-1],), self.weight, self.bias, self.eps)
                    return x.permute(0,3,1,2)        # (B,C,H,W)

            self.neck = nn.Sequential(
                nn.Conv2d(neck_in, 256, 1, bias=False),
                _LN2d(256),
                nn.Conv2d(256, 256, 3, padding=1, bias=False),
                _LN2d(256),
            )
            self._out_H = H; self._out_W = W
            self._last_dim = neck_in   # = last stage block dim = neck input
            # segment_anything's Sam shell accesses image_encoder.img_size
            self.img_size = isz

        def forward(self, x):
            x = self.patch_embed(x)             # (B,C0,H/4,W/4)
            B,C,H,W = x.shape
            x = x.permute(0,2,3,1).reshape(B, H*W, C)
            for layer in self.layers: x = layer(x)
            x = x.reshape(B, self._out_H, self._out_W, self._last_dim).permute(0,3,1,2)
            x = self.neck(x)                    # (B,256,out_H,out_W)
            # segment_anything's prompt encoder expects (B,256,img_size//16,img_size//16)
            # = (B,256,64,64) for img_size=1024.  If our encoder produces a smaller
            # spatial size (e.g. 32×32 due to 3 downsamples after stride-4 patch embed),
            # upsample to match without adding any checkpoint parameters.
            target = self.img_size // 16
            if x.shape[-1] != target:
                x = F.interpolate(x, size=(target, target),
                                  mode="bilinear", align_corners=False)
            return x



        def _read_downsample_ci(state, n_stages):
            """Read each stage's downsample INPUT channels from checkpoint conv1 shape[1]."""
            result = {}
            for si in range(1, n_stages):
                key = f"image_encoder.layers.{si}.downsample.conv1.c.weight"
                if key in state:
                    try:
                        result[si] = tuple(state[key].shape)[1]
                    except Exception:
                        pass
            return result

    def _infer_tinyvit_config(state: dict) -> dict:
        """
        Read all TinyViT hyper-parameters directly from checkpoint tensor shapes.
        Uses the most reliable key available for each quantity and falls back
        gracefully so the error log tells us exactly what went wrong.
        """
        def _shape(key):
            if key not in state:
                return ()
            try:
                return tuple(state[key].shape)
            except Exception:
                return ()

        # ── discover number of stages ─────────────────────────────────────────
        stage_indices: set = set()
        for k in state:
            if k.startswith("image_encoder.layers."):
                parts = k.split(".")
                if len(parts) > 2 and parts[2].isdigit():
                    stage_indices.add(int(parts[2]))
        n_stages = max(stage_indices) + 1 if stage_indices else 4

        # ══════════════════════════════════════════════════════════════════════
        # Key insight: In TinyViT, embed_dims[i] is the DIM OF THE BLOCKS in
        # stage i, but the PatchMerging that creates stage i may output a
        # DIFFERENT dimension than the blocks it feeds into.
        #
        # Concretely (from MobileSAM checkpoints seen in the wild):
        #   patch_embed outputs c0=64
        #   stage 0 blocks have dim=64  (local_conv=[64,1,3,3])
        #   stage 1 PatchMerging: ci=128, co=160  ← ci ≠ stage0 block dim!
        #
        # This happens when stage 0 contains MBConv blocks that expand 64→128.
        # The MBConv blocks' weights are stored under layers.0.blocks.{j}.*
        # but with a different naming pattern than the transformer blocks.
        #
        # Solution: read BOTH the block dim (from transformer block keys) AND
        # the downsample ci/co (from downsample.conv1/conv3 weight shapes).
        # ══════════════════════════════════════════════════════════════════════

        # ── step 1: detect block type and dims per stage ─────────────────────
        # Strategy: scan ALL keys under layers.{si}.blocks.0.* to determine
        # the block type (MBConv vs transformer) and infer the channel dim.
        #
        # MBConv markers  : conv1.c.weight, conv2.c.weight, conv3.c.weight
        # Transformer markers: attn.qkv.weight, norm1.weight, local_conv.c.weight
        # Both are absent : stage has no blocks, or uses a different prefix
        block_dims:      dict = {}   # {si: block INPUT dim}
        mbconv_out_dims: dict = {}   # {si: MBConv block OUTPUT dim} (may differ from input)
        mbconv_mid_dims: dict = {}   # {si: MBConv expansion mid channels}
        is_mbconv:       dict = {}   # {si: bool}
        mid_ch:          int  = None  # local var reset per stage

        # Collect all checkpoint keys for each stage's first block
        stage0_keys: dict = {}   # {si: set_of_subkeys_under_blocks.0}
        for k in state:
            pfx = "image_encoder.layers."
            if not k.startswith(pfx):
                continue
            rest = k[len(pfx):]           # e.g. "0.blocks.0.norm1.weight"
            parts = rest.split(".")
            if len(parts) < 4:
                continue
            try:
                si = int(parts[0])
            except ValueError:
                continue
            if parts[1] == "blocks" and parts[2] == "0":
                subkey = ".".join(parts[3:])  # e.g. "norm1.weight"
                stage0_keys.setdefault(si, set()).add(subkey)

        for si in range(n_stages):
            keys   = stage0_keys.get(si, set())
            mid_ch = None   # reset for each stage

            # ── Definitive classification rule ────────────────────────────────
            # A stage is a TRANSFORMER stage if and only if:
            #   norm1.weight is present in the checkpoint.
            # Everything else (local_conv, conv1, attn) can appear in both types.
            # This is the ONLY reliable discriminator across all TinyViT variants.
            has_norm1 = "norm1.weight" in keys
            has_attn  = any(k.startswith("attn.") for k in keys)
            is_transformer = has_norm1 or has_attn

            is_mbconv[si] = not is_transformer

            if is_mbconv[si]:
                # MBConv block input dim = conv1 input channels = shape[1]
                # MBConv block output dim = conv3 output channels = shape[0]
                # These may DIFFER (non-residual expansion, e.g. 64→128).
                # We store block_dims[si] = INPUT dim (what patch_embed must output).
                # The OUTPUT dim is stored separately as mbconv_out_dims[si].
                mb_in  = None
                mb_out = None
                sh1 = _shape(f"image_encoder.layers.{si}.blocks.0.conv1.c.weight")
                sh2 = _shape(f"image_encoder.layers.{si}.blocks.0.conv2.c.weight")
                sh3 = _shape(f"image_encoder.layers.{si}.blocks.0.conv3.c.weight")
                if sh1 and len(sh1) >= 2:
                    mb_in = sh1[1]   # conv1 input channels = block input
                    mid_ch = sh1[0]  # conv1 output = expansion mid channels
                elif sh2 and len(sh2) >= 1:
                    mid_ch = sh2[0]
                if sh3 and len(sh3) >= 1:
                    mb_out = sh3[0]  # conv3 output = block output
                # Fallback: local_conv dim
                if mb_in is None:
                    sh = _shape(f"image_encoder.layers.{si}.blocks.0.local_conv.c.weight")
                    if sh:
                        mb_in = sh[0]; mb_out = sh[0]; mid_ch = sh[0]
                if mb_in is not None:
                    block_dims[si]      = mb_in
                    mbconv_out_dims[si] = mb_out if mb_out is not None else mb_in
                    # Store the actual mid channels for accurate _MBConv construction
                    # expand_ratio = mid_ch / mb_in (may not be exactly 4.0)
                    if mid_ch is not None and mb_in and mb_in > 0:
                        mbconv_mid_dims[si] = mid_ch
            else:
                # Transformer block: get dim from the most informative key.
                for src_key in [
                    f"image_encoder.layers.{si}.blocks.0.local_conv.c.weight",
                    f"image_encoder.layers.{si}.blocks.0.attn.norm.weight",
                    f"image_encoder.layers.{si}.blocks.0.norm1.weight",
                    f"image_encoder.layers.{si}.blocks.0.norm2.weight",
                ]:
                    sh = _shape(src_key)
                    if sh:
                        block_dims[si] = sh[0]
                        break
                if si not in block_dims:
                    sh = _shape(f"image_encoder.layers.{si}.blocks.0.attn.qkv.weight")
                    if sh and len(sh) >= 2:
                        block_dims[si] = sh[1]

        # Last-resort fallback for stage 0: use patch_embed output dim
        if 0 not in block_dims:
            for pe_key in ("image_encoder.patch_embed.seq.2.c.weight",
                           "image_encoder.patch_embed.seq.0.c.weight"):
                sh = _shape(pe_key)
                if sh:
                    block_dims[0] = sh[0]
                    break

        # ── step 2: downsample ci and co from checkpoint directly ─────────────
        # down_ci[i] = input  channels of stage i's PatchMerging
        # down_co[i] = output channels of stage i's PatchMerging
        # NOTE: stage 0 can also have a downsample (e.g. MobileSAM-21M where
        # MBConv blocks output 64 and then a PatchMerging takes 64→128).
        down_ci: dict = {}
        down_co: dict = {}
        for si in range(n_stages):   # ALL stages, including 0
            sh1 = _shape(f"image_encoder.layers.{si}.downsample.conv1.c.weight")  # (co, ci, 1, 1)
            sh3 = _shape(f"image_encoder.layers.{si}.downsample.conv3.c.weight")  # (co, co, 1, 1)
            if sh1 and len(sh1) >= 2:
                down_co[si] = sh1[0]
                down_ci[si] = sh1[1]
            elif sh3 and len(sh3) >= 1:
                down_co[si] = sh3[0]

        # ── step 3: embed_dims ────────────────────────────────────────────────
        # For stages with downsample (i>0): down_co[i] is the definitive output.
        # For stage 0: use block_dims[0] (from conv3 output or attn input dim).
        # Fallback to 64*2^i if nothing found.
        embed_dims = []
        for si in range(n_stages):
            if si in down_co:
                # Downsample output is the definitive channel count flowing out of this stage
                embed_dims.append(down_co[si])
            elif si in block_dims:
                embed_dims.append(block_dims[si])
            else:
                embed_dims.append(64 * (2 ** si))

        # For MBConv stage 0: also try to get the output dim from conv3.c.weight
        # (the final 1x1 conv output = what feeds the next downsample input)
        if is_mbconv.get(0, False) and 0 in block_dims:
            # block_dims[0] = MBConv input dim (from conv1 shape[1])
            # The output dim of MBConv = same (residual connection keeps dim)
            # BUT the stage-1 downsample ci tells us what actually comes out
            # down_ci[1] is the authoritative value
            pass   # embed_dims[0] stays as block_dims[0] which is correct

        # ── step 4: depths ────────────────────────────────────────────────────
        depths = []
        for si in range(n_stages):
            blk = set()
            prefix = f"image_encoder.layers.{si}.blocks."
            for k in state:
                if k.startswith(prefix):
                    p = k.split(".")
                    if len(p) > 4 and p[4].isdigit():
                        blk.add(int(p[4]))
            depths.append(max(blk) + 1 if blk else 2)

        # ── step 5: num_heads from qkv weight ─────────────────────────────────
        # MBConv stages have no attention → num_heads is irrelevant (use 0)
        num_heads = []
        _pref = {2, 4, 5, 8, 10, 16, 20, 32}
        for si in range(n_stages):
            if is_mbconv.get(si, False):
                num_heads.append(0)   # unused for MBConv stages
                continue
            sh = _shape(f"image_encoder.layers.{si}.blocks.0.attn.qkv.weight")
            if sh and len(sh) >= 2:
                out_dim, blk_dim = sh[0], sh[1]
                cands = [(nh, out_dim // (3*nh))
                         for nh in range(1, 65) if out_dim % (3*nh) == 0]
                pref_c = [c for c in cands if c[0] in _pref]
                pool   = pref_c if pref_c else cands
                target = max(2, blk_dim // 32)
                best   = min(pool, key=lambda c: abs(c[0]-target)) if pool else (2, 1)
                num_heads.append(best[0])
            else:
                num_heads.append(max(2, embed_dims[si] // 32))

        # ── step 6: window_sizes from attn.ab buffer ──────────────────────────
        # MBConv stages have no attention → no attn.ab → use 0 as sentinel
        window_sizes = []
        for si in range(n_stages):
            if is_mbconv.get(si, False):
                window_sizes.append(0)   # unused for MBConv stages
                continue
            sh = _shape(f"image_encoder.layers.{si}.blocks.0.attn.ab")
            if sh and sh[0] > 0:
                window_sizes.append(max(1, int(sh[0] ** 0.5 + 0.5)))
            else:
                window_sizes.append(7)

        # neck_in: input channels of neck Conv2d = last stage's block output dim
        # Read directly from neck.0.weight shape[1] — most reliable source.
        neck_in_from_ckpt = None
        sh = _shape("image_encoder.neck.0.weight")  # (256, neck_in, 1, 1)
        if sh and len(sh) >= 2:
            neck_in_from_ckpt = sh[1]

        # ── MBConv block properties ───────────────────────────────────────────
        shortcut_types:  dict = {}
        has_local_convs: dict = {}
        for si in range(n_stages):
            if is_mbconv.get(si, False):
                props = _detect_mbconv_props(state, si)
                shortcut_types[si]  = props["shortcut_type"]
                has_local_convs[si] = props["has_local_conv"]

        # ── Transformer block naming conventions ──────────────────────────────
        tvblock_props: dict = {}   # {si: {mlp_style, has_norm1, has_local_conv}}
        for si in range(n_stages):
            if not is_mbconv.get(si, False):
                tvblock_props[si] = _detect_tvblock_props(state, si)

        return {
            "img_size":          1024,
            "embed_dims":        embed_dims,
            "depths":            depths,
            "num_heads":         num_heads,
            "window_sizes":      window_sizes,
            "downsample_ci":     down_ci,
            "block_dims":        block_dims,
            "mbconv_out_dims":   mbconv_out_dims,
            "mbconv_mid_dims":   mbconv_mid_dims,
            "is_mbconv":         is_mbconv,
            "shortcut_types":    shortcut_types,
            "has_local_convs":   has_local_convs,
            "tvblock_props":     tvblock_props,
            "neck_in":           neck_in_from_ckpt,
        }


    def _build_mobile_sam_with_segment_anything(checkpoint_path: str):
        """
        Load mobile_sam.pt using segment_anything + built-in TinyViT encoder.
        All tensor shapes are read from the checkpoint — no hardcoded dims.
        """
        from segment_anything import sam_model_registry, SamPredictor

        # 1. Load raw state-dict
        try:
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            state = torch.load(checkpoint_path, map_location="cpu")
        for wk in ("model", "state_dict"):
            if isinstance(state, dict) and wk in state and isinstance(state[wk], dict):
                state = state[wk]; break

        if not isinstance(state, dict):
            raise RuntimeError("Checkpoint is not a state-dict.")

        # 2. Infer TinyViT config from actual tensor shapes
        cfg = _infer_tinyvit_config(state)

        # 3. Build SAM shell (vit_b has same mask-decoder / prompt-encoder)
        sam = sam_model_registry["vit_b"](checkpoint=None)

        # 4. Swap image encoder with our shape-driven TinyViT
        sam.image_encoder = _TinyViTEncoder(cfg)

        # 5. Load weights — strict=False allows minor key mismatches in
        #    buffer names (e.g. attention bias buffers named differently)
        missing, unexpected = sam.load_state_dict(state, strict=False)

        # Only fail if encoder *parameters* (not buffers) are missing
        enc_param_missing = [
            k for k in missing
            if k.startswith("image_encoder")
            and not k.endswith(".ab")          # bias buffers are ok to miss
        ]
        if enc_param_missing:
            raise RuntimeError(
                "TinyViT parameter mismatch after loading.\n"
                "Missing keys (first 10):\n  " +
                "\n  ".join(enc_param_missing[:10]) + "\n\n"
                "Inferred config:\n  " + str(cfg)
            )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam.eval().to(device)
        return sam, SamPredictor(sam)


# ── Checkpoint fingerprinting ────────────────────────────────────────────────

class _SAMFingerprint:
    """Everything we learn by peeking at a raw state-dict before building a model."""
    __slots__ = ("is_mobile_sam", "sam_model_type",
                 "needs_lib", "n_enc_blocks", "has_rel_pos", "load_error")

    def __init__(self):
        self.is_mobile_sam  = False
        self.sam_model_type = "vit_b"
        self.needs_lib      = "segment_anything"
        self.n_enc_blocks   = 0
        self.has_rel_pos    = False
        self.load_error     = ""


def _fingerprint_checkpoint(path: str) -> "_SAMFingerprint":
    fp = _SAMFingerprint()

    if not TORCH_AVAILABLE:
        base = os.path.basename(path).lower()
        if "mobile" in base or "vit_t" in base:
            fp.is_mobile_sam = True; fp.sam_model_type = "vit_t"; fp.needs_lib = "mobile_sam"
        elif "vit_h" in base: fp.sam_model_type = "vit_h"
        elif "vit_l" in base: fp.sam_model_type = "vit_l"
        return fp

    try:
        try:
            state = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            state = torch.load(path, map_location="cpu")
        for wrap in ("model", "state_dict", "net"):
            if isinstance(state, dict) and wrap in state and isinstance(state[wrap], dict):
                state = state[wrap]; break
    except Exception as exc:
        fp.load_error = str(exc)
        base = os.path.basename(path).lower()
        if "mobile" in base or "vit_t" in base:
            fp.is_mobile_sam = True; fp.sam_model_type = "vit_t"; fp.needs_lib = "mobile_sam"
        elif "vit_h" in base: fp.sam_model_type = "vit_h"
        elif "vit_l" in base: fp.sam_model_type = "vit_l"
        return fp

    if not isinstance(state, dict):
        fp.load_error = "Checkpoint is not a state-dict."; return fp

    fp.has_rel_pos = any("attn.rel_pos_h" in k for k in state)

    block_idx: set = set()
    for k in state:
        if k.startswith("image_encoder.blocks."):
            parts = k.split(".")
            if len(parts) > 2 and parts[2].isdigit():
                block_idx.add(int(parts[2]))
    fp.n_enc_blocks = len(block_idx)

    if not fp.has_rel_pos:
        # MobileSAM – no rel_pos, TinyViT backbone
        fp.is_mobile_sam = True; fp.sam_model_type = "vit_t"
        # Can load via mobile_sam package OR our built-in bridge
        fp.needs_lib = "mobile_sam" if "mobile_sam" in SAM_REGISTRIES else "segment_anything_bridge"
    else:
        fp.is_mobile_sam = False
        fp.sam_model_type = "vit_b" if fp.n_enc_blocks <= 12 else "vit_l" if fp.n_enc_blocks <= 24 else "vit_h"
        fp.needs_lib = "segment_anything"

    return fp


def _load_sam_model(path: str, fp: "_SAMFingerprint"):
    """
    Build + load the right SAM model for this checkpoint.
    Returns (sam_model, predictor, label_str).
    """
    device = "cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"

    # ── MobileSAM via dedicated package ─────────────────────────────────────
    if fp.is_mobile_sam and "mobile_sam" in SAM_REGISTRIES:
        registry, PredictorCls = SAM_REGISTRIES["mobile_sam"]
        model = registry["vit_t"](checkpoint=path)
        model.eval().to(device)
        return model, PredictorCls(model), "MobileSAM vit_t [mobile_sam]"

    # ── MobileSAM via built-in TinyViT bridge + segment_anything shell ──────
    if fp.is_mobile_sam and TORCH_AVAILABLE and "segment_anything" in SAM_REGISTRIES:
        sam, predictor = _build_mobile_sam_with_segment_anything(path)
        return sam, predictor, "MobileSAM vit_t [built-in bridge + segment_anything]"

    # ── MobileSAM but nothing useful installed ───────────────────────────────
    if fp.is_mobile_sam:
        raise RuntimeError(
            "This is a MobileSAM checkpoint, but neither mobile_sam nor\n"
            "segment_anything is installed.\n\n"
            "Install one:\n"
            "    pip install segment-anything\n"
            "    pip install git+https://github.com/ChaoningZhang/MobileSAM.git"
        )

    # ── Original SAM (vit_b / vit_l / vit_h) ────────────────────────────────
    if "segment_anything" in SAM_REGISTRIES:
        registry, PredictorCls = SAM_REGISTRIES["segment_anything"]
        mt = fp.sam_model_type
        if mt not in registry:
            raise RuntimeError(f"Model type '{mt}' not in segment_anything registry "
                               f"(available: {list(registry.keys())}).")
        model = registry[mt](checkpoint=path)
        model.eval().to(device)
        return model, PredictorCls(model), f"SAM {mt} [segment_anything]"

    raise RuntimeError(
        "No SAM library installed.\n"
        "    pip install segment-anything"
    )


def _best_lib_for(model_type: str) -> str:
    for lib_name, (registry, _) in SAM_REGISTRIES.items():
        if model_type in registry:
            return lib_name
    return next(iter(SAM_REGISTRIES), "")

# ── YOLO-seg (optional) ─────────────────────────────────────────────────────
try:
    from ultralytics import YOLO as UltralyticsYOLO
    YOLO_SEG_AVAILABLE = True
except ImportError:
    YOLO_SEG_AVAILABLE = False

# ════════════════════════════════════════════════════════════════════════════
# Constants
# ════════════════════════════════════════════════════════════════════════════
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
CLASS_FILE_NAMES  = ("classes.txt", "obj.names")
REVIEW_SUFFIX     = ".reviewed"
SETTINGS_ORG      = "SegAnnotator"
SETTINGS_APP      = "SemanticSegAnnotator"
CRASH_LOG         = "seg_annotator_crash.log"
MIN_POLYGON_PTS   = 3
HANDLE_RADIUS     = 6
APPROX_EPSILON_FRAC = 0.003   # fraction of perimeter for polygon simplification

DEFAULT_SHORTCUTS = {
    "open_folder":          "Ctrl+O",
    "save":                 "Ctrl+S",
    "undo":                 "Ctrl+Z",
    "redo":                 "Ctrl+Y",
    "delete":               "Delete",
    "copy":                 "Ctrl+C",
    "paste":                "Ctrl+V",
    "duplicate":            "Ctrl+D",
    "accept_suggestion":    "Return",
    "accept_all":           "Ctrl+Return",
    "reject_selected":      "Backspace",
    "reject_all":           "Shift+Backspace",
    "fit":                  "F",
    "actual_size":          "0",
    "prev_image":           "A",
    "next_image":           "D",
    "next_unlabeled":       "W",
    "zoom_in":              "Equal",
    "zoom_out":             "Minus",
    "toggle_reviewed":      "R",
    "set_class":            "L",
    "select_all":           "Ctrl+A",
    "escape":               "Escape",
    "sam_mode":             "S",
    "polygon_mode":         "P",
    "edit_mode":            "E",
}

# ════════════════════════════════════════════════════════════════════════════
# Data model
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class PolygonAnnotation:
    """A single polygon annotation.

    points: list of (x, y) in image-pixel space (absolute, NOT normalised).
    class_id: integer index into the project class list.
    score: confidence from model suggestion (-1 = manual).
    is_suggestion: whether this is a pending suggestion.
    """
    points: List[Tuple[float, float]]
    class_id: int = 0
    score: float = -1.0
    is_suggestion: bool = False

    # ── geometry helpers ────────────────────────────────────────────────────
    def bbox(self):
        if not self.points:
            return 0, 0, 0, 0
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        return min(xs), min(ys), max(xs), max(ys)

    def centroid(self):
        if not self.points:
            return 0.0, 0.0
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        return sum(xs) / len(xs), sum(ys) / len(ys)

    def area(self):
        """Shoelace formula."""
        pts = self.points
        n = len(pts)
        if n < 3:
            return 0.0
        s = 0.0
        for i in range(n):
            x0, y0 = pts[i]
            x1, y1 = pts[(i + 1) % n]
            s += x0 * y1 - x1 * y0
        return abs(s) / 2.0

    def contains_point(self, x: float, y: float) -> bool:
        """Ray-casting."""
        pts = self.points
        n = len(pts)
        if n < 3:
            return False
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = pts[i]
            xj, yj = pts[j]
            if ((yi > y) != (yj > y)) and (
                    x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi):
                inside = not inside
            j = i
        return inside

    def move_by(self, dx: float, dy: float):
        self.points = [(x + dx, y + dy) for x, y in self.points]

    def clamp(self, w: int, h: int):
        self.points = [(max(0.0, min(x, w)), max(0.0, min(y, h)))
                       for x, y in self.points]

    def to_yolo_seg(self, w: int, h: int) -> str:
        """Return one YOLO-seg label line."""
        coords = " ".join(
            f"{x / w:.6f} {y / h:.6f}" for x, y in self.points
        )
        return f"{self.class_id} {coords}\n"

    @staticmethod
    def from_yolo_seg(line: str, w: int, h: int) -> Optional["PolygonAnnotation"]:
        parts = line.strip().split()
        if len(parts) < 7 or (len(parts) - 1) % 2 != 0:
            return None
        try:
            class_id = int(float(parts[0]))
            coords = list(map(float, parts[1:]))
            points = [(coords[i] * w, coords[i + 1] * h)
                      for i in range(0, len(coords), 2)]
        except (ValueError, IndexError):
            return None
        if len(points) < MIN_POLYGON_PTS:
            return None
        return PolygonAnnotation(points=points, class_id=class_id)

    def clone(self) -> "PolygonAnnotation":
        return PolygonAnnotation(
            points=list(self.points),
            class_id=self.class_id,
            score=self.score,
            is_suggestion=self.is_suggestion,
        )

    @staticmethod
    def _snap_to_edges(contour: np.ndarray, image_rgb: np.ndarray,
                       mask_u8: np.ndarray,
                       snap_radius: int = 4,
                       canny_low: int = 30, canny_high: int = 100) -> np.ndarray:
        """
        Refine contour vertices by snapping them to nearby image edges,
        with a strict constraint: a vertex can ONLY move to a pixel that
        is still INSIDE the original SAM mask (never outside).

        This prevents any outward spiking — the worst case is the vertex
        stays exactly where it was.
        """
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        gray_smooth = cv2.bilateralFilter(gray, 5, 50, 50)
        edges = cv2.Canny(gray_smooth, canny_low, canny_high)

        # Hard constraint: candidate edge pixels must be INSIDE the mask.
        # This is the key difference — vertices can never move outside the SAM mask.
        edges_inside = cv2.bitwise_and(edges, mask_u8)

        H, W = edges_inside.shape
        edge_pts = np.argwhere(edges_inside > 0)  # (row=y, col=x)
        if len(edge_pts) == 0:
            return contour   # no interior edges → keep original contour

        snapped = contour.copy().astype(np.float32)
        for i, pt in enumerate(contour):
            x, y = float(pt[0][0]), float(pt[0][1])
            x0 = max(0, int(x) - snap_radius)
            x1 = min(W, int(x) + snap_radius + 1)
            y0 = max(0, int(y) - snap_radius)
            y1 = min(H, int(y) + snap_radius + 1)
            window_mask = ((edge_pts[:, 1] >= x0) & (edge_pts[:, 1] < x1) &
                           (edge_pts[:, 0] >= y0) & (edge_pts[:, 0] < y1))
            local = edge_pts[window_mask]
            if len(local) == 0:
                continue   # no edge inside mask nearby → vertex unchanged
            dists = (local[:, 1] - x) ** 2 + (local[:, 0] - y) ** 2
            best = local[np.argmin(dists)]
            snapped[i, 0, 0] = float(best[1])
            snapped[i, 0, 1] = float(best[0])

        return snapped.astype(np.int32)

    @staticmethod
    def from_mask(mask: np.ndarray, class_id: int,
                  score: float = -1.0,
                  epsilon_frac: float = APPROX_EPSILON_FRAC,
                  is_suggestion: bool = False,
                  image_rgb: Optional[np.ndarray] = None,
                  edge_snap: bool = False,
                  snap_radius: int = 4,
                  canny_low: int = 30,
                  canny_high: int = 100) -> Optional["PolygonAnnotation"]:
        """Convert a binary mask (H×W uint8) to a smooth simplified polygon.

        Roboflow-quality pipeline:
          1. Morphological smoothing of the mask (removes holes and rough edges)
          2. Gaussian blur on the mask for a soft boundary
          3. Re-threshold to get a clean binary mask
          4. Find the largest contour
          5. Optional: edge-guided vertex refinement (snap_to_edges)
          6. Douglas-Peucker simplification with adaptive epsilon
        """
        mask_u8 = (mask > 0).astype(np.uint8) * 255

        # ── Step 1: Morphological smoothing ──────────────────────────────
        # Close small holes, then open to remove thin protrusions/spikes
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, k_close)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN,  k_open)

        # ── Step 2: Gaussian blur + re-threshold (softens jagged edges) ──
        blurred = cv2.GaussianBlur(mask_u8, (21, 21), 0)
        _, mask_u8 = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea)

        # ── Step 3: Optional edge-guided refinement ───────────────────────
        if edge_snap and image_rgb is not None:
            contour = PolygonAnnotation._snap_to_edges(
                contour, image_rgb, mask_u8,
                snap_radius=snap_radius,
                canny_low=canny_low, canny_high=canny_high)

        # ── Step 4: Douglas-Peucker simplification ────────────────────────
        perimeter = cv2.arcLength(contour, True)
        epsilon = epsilon_frac * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        pts = [(float(pt[0][0]), float(pt[0][1])) for pt in approx]
        if len(pts) < MIN_POLYGON_PTS:
            return None
        return PolygonAnnotation(points=pts, class_id=class_id,
                                 score=score, is_suggestion=is_suggestion)


# ════════════════════════════════════════════════════════════════════════════
# Tiny segmentation model (MobileNetV3-Small + lightweight decoder)
# ════════════════════════════════════════════════════════════════════════════
if TORCH_AVAILABLE:
    class _ASPP(nn.Module):
        """Atrous Spatial Pyramid Pooling (lightweight)."""
        def __init__(self, in_ch: int, out_ch: int):
            super().__init__()
            self.b0 = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False),
                                    nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
            self.b1 = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=6,  dilation=6,  bias=False),
                                    nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
            self.b2 = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=12, dilation=12, bias=False),
                                    nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
            self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                      nn.Conv2d(in_ch, out_ch, 1, bias=False),
                                      nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
            self.proj = nn.Sequential(nn.Conv2d(out_ch * 4, out_ch, 1, bias=False),
                                      nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                                      nn.Dropout(0.1))

        def forward(self, x):
            h, w = x.shape[-2:]
            p = F.interpolate(self.pool(x), size=(h, w), mode="bilinear", align_corners=False)
            return self.proj(torch.cat([self.b0(x), self.b1(x), self.b2(x), p], dim=1))

    class TinySegNet(nn.Module):
        """MobileNetV3-Small backbone + ASPP + bilinear upsample decoder."""
        def __init__(self, num_classes: int):
            super().__init__()
            import torchvision.models as models
            backbone = models.mobilenet_v3_small(weights="IMAGENET1K_V1")
            # feature layers up to the last inverted residual (stride 16)
            self.encoder = backbone.features          # (B, 576, H/16, W/16)
            self.aspp    = _ASPP(576, 128)
            self.head    = nn.Sequential(
                nn.Conv2d(128, 64, 3, padding=1, bias=False),
                nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.Conv2d(64, num_classes, 1),
            )
            self._num_classes = num_classes

        def forward(self, x):
            h, w = x.shape[-2:]
            feat = self.encoder(x)
            feat = self.aspp(feat)
            feat = self.head(feat)
            return F.interpolate(feat, size=(h, w), mode="bilinear", align_corners=False)

        def predict_mask(self, bgr_img: np.ndarray,
                         device: str = "cpu") -> np.ndarray:
            """Return (H, W) int64 class-label map."""
            img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406])
            std  = np.array([0.229, 0.224, 0.225])
            img = (img - mean) / std
            t = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
            with torch.no_grad():
                logits = self(t)
            return logits.squeeze(0).argmax(0).cpu().numpy().astype(np.int64)


    class SegDataset(Dataset):
        """YOLO-seg dataset loader that rasterises polygons to masks."""
        def __init__(self, folder: str, image_files: List[str],
                     classes: List[str], img_size: int = 512):
            self.folder      = folder
            self.image_files = [f for f in image_files
                                if self._has_label(folder, f)]
            self.classes     = classes
            self.img_size    = img_size

        @staticmethod
        def _has_label(folder, name):
            lp = os.path.splitext(os.path.join(folder, name))[0] + ".txt"
            return os.path.exists(lp) and os.path.getsize(lp) > 0

        def __len__(self):
            return len(self.image_files)

        def __getitem__(self, idx):
            name  = self.image_files[idx]
            img_p = os.path.join(self.folder, name)
            lbl_p = os.path.splitext(img_p)[0] + ".txt"
            img = cv2.imread(img_p)
            if img is None:
                img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            H, W = img.shape[:2]
            # build mask — use int32 (OpenCV 5 no longer accepts int64 for fillPoly)
            mask = np.zeros((H, W), dtype=np.int32)
            if os.path.exists(lbl_p):
                with open(lbl_p, encoding="utf-8") as fh:
                    for line in fh:
                        ann = PolygonAnnotation.from_yolo_seg(line, W, H)
                        if ann is None:
                            continue
                        pts = np.ascontiguousarray(
                            np.array(ann.points, dtype=np.int32).reshape(-1, 1, 2))
                        # np.ascontiguousarray ensures C-contiguous memory layout
                        # required by OpenCV 5's fillPoly
                        mask_draw = np.ascontiguousarray(mask)
                        cv2.fillPoly(mask_draw, [pts], int(ann.class_id))
                        mask = mask_draw
            # resize
            img  = cv2.resize(img,  (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
            # normalise image
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406])
            std  = np.array([0.229, 0.224, 0.225])
            img_rgb = (img_rgb - mean) / std
            t_img  = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float()
            t_mask = torch.from_numpy(mask).long()
            return t_img, t_mask


    class TrainWorker(QObject):
        """Background training worker that emits progress signals."""
        progress     = pyqtSignal(int, int, float, float)   # epoch, total, loss, miou
        finished     = pyqtSignal(str)                       # checkpoint path
        error        = pyqtSignal(str)

        def __init__(self, cfg: dict):
            super().__init__()
            self.cfg  = cfg
            self._stop = False

        def stop(self):
            self._stop = True

        def run(self):
            try:
                self._train()
            except Exception as exc:
                self.error.emit(traceback.format_exc())

        def _train(self):
            cfg   = self.cfg
            dev   = "cuda" if torch.cuda.is_available() else "cpu"
            nc    = cfg["num_classes"]
            model = TinySegNet(nc).to(dev)

            # resume if checkpoint exists
            ckpt_path = cfg.get("checkpoint", "")
            if ckpt_path and os.path.exists(ckpt_path):
                state = torch.load(ckpt_path, map_location=dev)
                try:
                    model.load_state_dict(state["model"])
                except Exception:
                    pass   # architecture mismatch – start fresh

            ds = SegDataset(cfg["folder"], cfg["image_files"],
                            cfg["classes"], cfg.get("img_size", 512))
            if len(ds) == 0:
                self.error.emit("No labeled images found for training.")
                return

            loader = DataLoader(ds, batch_size=cfg.get("batch", 4),
                                shuffle=True, num_workers=0,
                                pin_memory=(dev == "cuda"))
            criterion = nn.CrossEntropyLoss(ignore_index=-1)
            opt = optim.AdamW(model.parameters(),
                              lr=cfg.get("lr", 1e-3),
                              weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=cfg.get("epochs", 20))

            epochs = cfg.get("epochs", 20)
            save_dir = cfg.get("save_dir", ".")
            os.makedirs(save_dir, exist_ok=True)
            best_path = os.path.join(save_dir, "best_seg.pt")
            last_path = os.path.join(save_dir, "last_seg.pt")
            best_miou = -1.0

            for epoch in range(1, epochs + 1):
                if self._stop:
                    break
                model.train()
                total_loss = 0.0
                iou_sum = 0.0
                iou_cnt = 0
                for imgs, masks in loader:
                    if self._stop:
                        break
                    imgs  = imgs.to(dev)
                    masks = masks.to(dev)
                    opt.zero_grad()
                    logits = model(imgs)
                    loss = criterion(logits, masks)
                    loss.backward()
                    opt.step()
                    total_loss += loss.item()
                    # mean IoU (macro, ignore bg)
                    pred = logits.argmax(1)
                    for cls in range(1, nc):
                        inter = ((pred == cls) & (masks == cls)).sum().float()
                        union = ((pred == cls) | (masks == cls)).sum().float()
                        if union > 0:
                            iou_sum += (inter / union).item()
                            iou_cnt += 1

                scheduler.step()
                avg_loss = total_loss / max(len(loader), 1)
                avg_miou = iou_sum / max(iou_cnt, 1)
                # save last
                torch.save({"model": model.state_dict(),
                            "epoch": epoch,
                            "num_classes": nc,
                            "miou": avg_miou}, last_path)
                if avg_miou > best_miou:
                    best_miou = avg_miou
                    torch.save({"model": model.state_dict(),
                                "epoch": epoch,
                                "num_classes": nc,
                                "miou": avg_miou}, best_path)
                self.progress.emit(epoch, epochs, avg_loss, avg_miou)

            self.finished.emit(best_path)


# ════════════════════════════════════════════════════════════════════════════
# Canvas widget
# ════════════════════════════════════════════════════════════════════════════
class Canvas(QWidget):
    annotations_changed  = pyqtSignal()
    selection_changed    = pyqtSignal(int)
    mouse_pos_changed    = pyqtSignal(str)
    suggestions_changed  = pyqtSignal()
    request_navigate     = pyqtSignal(int)   # -1 / +1
    class_edit_requested = pyqtSignal(int)
    edge_click_requested = pyqtSignal(float, float)  # ix, iy in image coords

    # drawing modes
    MODE_IDLE    = "idle"
    MODE_SAM     = "sam"       # single-click SAM prompt
    MODE_POLYGON = "polygon"   # manual polygon drawing
    MODE_EDIT    = "edit"      # select / move / vertex edit
    MODE_PAN     = "pan"
    MODE_EDGE    = "edge"      # edge-detection flood-fill annotator

    def __init__(self):
        super().__init__()
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

        # image state
        self.image:    Optional[np.ndarray]  = None
        self.image_rgb:Optional[np.ndarray]  = None
        self.pixmap:   Optional[QPixmap]     = None
        self.image_path: str                 = ""

        # annotation state
        self.annotations:  List[PolygonAnnotation] = []
        self.suggestions:  List[PolygonAnnotation] = []
        self.classes:      List[str]               = []
        self.current_class_id:   int               = 0
        self.selected_indices:   set               = set()
        self.selected_suggestion_index: int        = -1

        # view
        self.scale    = 1.0
        self.zoom     = 1.0
        self.pan_x    = 0.0
        self.pan_y    = 0.0
        self.base_ox  = 0.0
        self.base_oy  = 0.0

        # interaction
        self.mode         = self.MODE_IDLE
        self.draw_points: List[Tuple[float, float]] = []
        self.preview_pt:  Optional[Tuple[float, float]] = None
        self.drag_start_img: Optional[Tuple[float, float]] = None
        self.drag_start_boxes: dict  = {}
        self.last_mouse:  QPoint     = QPoint()
        self.space_pressed: bool     = False

        # vertex edit
        self.edit_ann_idx:  int  = -1
        self.edit_pt_idx:   int  = -1
        self.edit_dragging: bool = False

        # history
        self.history:      List[dict] = []
        self.redo_stack:   List[dict] = []
        self.max_history   = 100

        # clipboard
        self.clipboard: List[PolygonAnnotation] = []

        # display options
        self.show_annotations = True
        self.show_suggestions = True
        self.alpha_fill       = 80    # 0-255

        # cursor
        self.plus_cursor = self._make_plus_cursor()

    # ── cursor ──────────────────────────────────────────────────────────────
    def _make_plus_cursor(self) -> QCursor:
        sz = 21
        pix = QPixmap(sz, sz)
        pix.fill(Qt.transparent)
        p = QPainter(pix)
        pen = QPen(QColor(20, 20, 20), 2)
        p.setPen(pen)
        c = sz // 2
        p.drawLine(c, 2, c, sz - 3)
        p.drawLine(2, c, sz - 3, c)
        p.end()
        return QCursor(pix, c, c)

    # ── image loading ────────────────────────────────────────────────────────
    def has_image(self) -> bool:
        return self.image is not None

    def load_image(self, path: str):
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Cannot read image: {path}")
        self.image_path = path
        self.image      = img
        self.image_rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w            = img.shape[:2]
        self.pixmap     = QPixmap.fromImage(
            QImage(self.image_rgb.data, w, h, 3 * w, QImage.Format_RGB888))
        self.annotations  = []
        self.suggestions  = []
        self.selected_indices = set()
        self.selected_suggestion_index = -1
        self.draw_points  = []
        self.preview_pt   = None
        self.mode         = self.MODE_IDLE
        self.history      = []
        self.redo_stack   = []
        self.pan_x        = 0.0
        self.pan_y        = 0.0
        self._load_labels()
        self._fit()
        self.annotations_changed.emit()
        self.update()

    def clear_image(self):
        self.image = None
        self.image_rgb = None
        self.pixmap    = None
        self.image_path = ""
        self.annotations  = []
        self.suggestions  = []
        self.selected_indices = set()
        self.selected_suggestion_index = -1
        self.draw_points  = []
        self.preview_pt   = None
        self.mode         = self.MODE_IDLE
        self.history      = []
        self.redo_stack   = []
        self.annotations_changed.emit()
        self.suggestions_changed.emit()
        self.update()

    def _load_labels(self):
        lp = self._label_path()
        if not os.path.exists(lp):
            return
        h, w = self.image.shape[:2]
        with open(lp, encoding="utf-8") as fh:
            for line in fh:
                ann = PolygonAnnotation.from_yolo_seg(line, w, h)
                if ann:
                    self.annotations.append(ann)

    def save_labels(self):
        if not self.has_image():
            return
        lp = self._label_path()
        h, w = self.image.shape[:2]
        if not self.annotations:
            if os.path.exists(lp):
                os.remove(lp)
            return
        lines = [a.to_yolo_seg(w, h) for a in self.annotations]
        _atomic_write(lp, lines)

    def _label_path(self) -> str:
        return os.path.splitext(self.image_path)[0] + ".txt"

    # ── view helpers ─────────────────────────────────────────────────────────
    def _fit(self):
        self.zoom  = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.update()

    def fit_to_window(self):
        self._fit()

    def actual_size(self):
        if not self.has_image():
            return
        bs = self._base_scale()
        self.zoom = 1.0 / max(bs, 1e-9)
        self.pan_x = self.pan_y = 0.0
        self.update()

    def _base_scale(self) -> float:
        if not self.has_image():
            return 1.0
        h, w = self.image.shape[:2]
        return min(self.width() / max(w, 1), self.height() / max(h, 1))

    def _cur_scale(self) -> float:
        return self._base_scale() * self.zoom

    def _update_metrics(self):
        if not self.has_image():
            self.scale = 1.0; self.base_ox = 0.0; self.base_oy = 0.0; return
        h, w = self.image.shape[:2]
        self.scale = self._base_scale()
        sc = self._cur_scale()
        self.base_ox = (self.width()  - w * sc) / 2.0 + self.pan_x
        self.base_oy = (self.height() - h * sc) / 2.0 + self.pan_y

    def img2disp(self, x, y):
        sc = self._cur_scale()
        return x * sc + self.base_ox, y * sc + self.base_oy

    def disp2img(self, x, y):
        sc = self._cur_scale()
        if sc == 0:
            return 0.0, 0.0
        return (x - self.base_ox) / sc, (y - self.base_oy) / sc

    def clamp_img(self, x, y):
        if not self.has_image():
            return x, y
        h, w = self.image.shape[:2]
        return max(0.0, min(float(x), w)), max(0.0, min(float(y), h))

    # ── history ──────────────────────────────────────────────────────────────
    def _snapshot(self) -> dict:
        return {
            "annotations":  [a.clone() for a in self.annotations],
            "suggestions":  [a.clone() for a in self.suggestions],
            "sel":          set(self.selected_indices),
            "sel_sug":      self.selected_suggestion_index,
        }

    def push_history(self):
        self.history.append(self._snapshot())
        self.redo_stack.clear()
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def undo(self):
        if not self.history:
            return
        self.redo_stack.append(self._snapshot())
        s = self.history.pop()
        self._restore(s)

    def redo(self):
        if not self.redo_stack:
            return
        self.history.append(self._snapshot())
        s = self.redo_stack.pop()
        self._restore(s)

    def _restore(self, s: dict):
        self.annotations  = [a.clone() for a in s["annotations"]]
        self.suggestions  = [a.clone() for a in s["suggestions"]]
        self.selected_indices = set(s["sel"])
        self.selected_suggestion_index = s["sel_sug"]
        self.annotations_changed.emit()
        self.suggestions_changed.emit()
        self.update()

    # ── selection ────────────────────────────────────────────────────────────
    def set_selected(self, idx: int):
        self.selected_indices = {idx} if idx >= 0 else set()
        self.selection_changed.emit(idx)
        self.update()

    def select_all(self):
        self.selected_indices = set(range(len(self.annotations)))
        self.selection_changed.emit(max(self.selected_indices) if self.selected_indices else -1)
        self.update()

    def _ann_at(self, ix, iy) -> int:
        for idx in reversed(range(len(self.annotations))):
            if self.annotations[idx].contains_point(ix, iy):
                return idx
        return -1

    def _sug_at(self, ix, iy) -> int:
        for idx in reversed(range(len(self.suggestions))):
            if self.suggestions[idx].contains_point(ix, iy):
                return idx
        return -1

    def _vertex_at(self, ann_idx: int, ix: float, iy: float) -> int:
        """Return vertex index near (ix, iy) for ann_idx, or -1."""
        if not (0 <= ann_idx < len(self.annotations)):
            return -1
        sc = self._cur_scale()
        radius = HANDLE_RADIUS / max(sc, 1e-9)
        for vi, (px, py) in enumerate(self.annotations[ann_idx].points):
            if abs(px - ix) <= radius and abs(py - iy) <= radius:
                return vi
        return -1

    # ── annotation helpers ───────────────────────────────────────────────────
    def set_classes(self, classes: List[str]):
        self.classes = classes
        self.update()

    def class_name(self, cid: int) -> str:
        if 0 <= cid < len(self.classes):
            return self.classes[cid]
        return f"class_{cid}"

    def ann_color(self, cid: int) -> QColor:
        hue = (cid * 47) % 360
        return QColor.fromHsv(hue, 220, 255)

    # ── suggestions management ───────────────────────────────────────────────
    def set_suggestions(self, suggestions: List[PolygonAnnotation]):
        self.suggestions = suggestions
        self.selected_suggestion_index = 0 if suggestions else -1
        self.suggestions_changed.emit()
        self.update()

    def accept_selected_suggestion(self):
        idx = self.selected_suggestion_index
        if not (0 <= idx < len(self.suggestions)):
            return
        self.push_history()
        s = self.suggestions.pop(idx)
        s.is_suggestion = False
        self.annotations.append(s)
        self.set_selected(len(self.annotations) - 1)
        self.selected_suggestion_index = min(idx, len(self.suggestions) - 1)
        self.annotations_changed.emit()
        self.suggestions_changed.emit()
        self.update()

    def accept_all_suggestions(self):
        if not self.suggestions:
            return
        self.push_history()
        for s in self.suggestions:
            s.is_suggestion = False
        self.annotations.extend(self.suggestions)
        self.suggestions.clear()
        self.selected_suggestion_index = -1
        self.set_selected(len(self.annotations) - 1)
        self.annotations_changed.emit()
        self.suggestions_changed.emit()
        self.update()

    def reject_selected_suggestion(self):
        idx = self.selected_suggestion_index
        if not (0 <= idx < len(self.suggestions)):
            return
        self.push_history()
        self.suggestions.pop(idx)
        self.selected_suggestion_index = min(idx, len(self.suggestions) - 1)
        self.suggestions_changed.emit()
        self.update()

    def reject_all_suggestions(self):
        if not self.suggestions:
            return
        self.push_history()
        self.suggestions.clear()
        self.selected_suggestion_index = -1
        self.suggestions_changed.emit()
        self.update()

    def delete_selected(self):
        if not self.selected_indices:
            return
        self.push_history()
        self.annotations = [a for i, a in enumerate(self.annotations)
                            if i not in self.selected_indices]
        self.selected_indices.clear()
        self.annotations_changed.emit()
        self.update()

    def copy_selected(self):
        self.clipboard = [self.annotations[i].clone()
                          for i in sorted(self.selected_indices)
                          if 0 <= i < len(self.annotations)]

    def paste(self):
        if not self.clipboard:
            return
        self.push_history()
        new_idx = []
        for a in self.clipboard:
            c = a.clone()
            c.move_by(10, 10)
            self.annotations.append(c)
            new_idx.append(len(self.annotations) - 1)
        self.selected_indices = set(new_idx)
        self.annotations_changed.emit()
        self.update()

    def duplicate_selected(self):
        self.copy_selected()
        self.paste()

    def set_class_for_selected(self, class_id: int):
        if not self.selected_indices and self.selected_suggestion_index < 0:
            return
        self.push_history()
        for i in self.selected_indices:
            if 0 <= i < len(self.annotations):
                self.annotations[i].class_id = class_id
        if 0 <= self.selected_suggestion_index < len(self.suggestions):
            self.suggestions[self.selected_suggestion_index].class_id = class_id
        self.annotations_changed.emit()
        self.update()

    def reset_annotations(self):
        if not self.annotations:
            return
        self.push_history()
        self.annotations.clear()
        self.selected_indices.clear()
        self.annotations_changed.emit()
        self.update()

    # ── mode control (called from main window) ───────────────────────────────
    def set_mode(self, mode: str):
        if self.mode == self.MODE_POLYGON and mode != self.MODE_POLYGON:
            # discard in-progress polygon
            self.draw_points.clear()
            self.preview_pt = None
        self.mode = mode
        if mode == self.MODE_SAM:
            self.setCursor(self.plus_cursor)
        elif mode == self.MODE_POLYGON:
            self.setCursor(self.plus_cursor)
        elif mode == self.MODE_EDIT:
            self.setCursor(Qt.ArrowCursor)
        else:
            self.setCursor(Qt.ArrowCursor)
        self.update()

    # ── paint ────────────────────────────────────────────────────────────────
    def paintEvent(self, _ev):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(32, 35, 42))
        if not self.has_image():
            painter.setPen(QColor(200, 200, 200))
            painter.setFont(QFont("Segoe UI", 13))
            painter.drawText(self.rect(), Qt.AlignCenter,
                             "Open an image folder to begin")
            return
        self._update_metrics()
        sc = self._cur_scale()
        h, w = self.image.shape[:2]
        painter.drawPixmap(int(self.base_ox), int(self.base_oy),
                           int(w * sc), int(h * sc), self.pixmap)

        if self.show_annotations:
            for idx, ann in enumerate(self.annotations):
                self._draw_polygon(painter, ann,
                                   selected=(idx in self.selected_indices))
        if self.show_suggestions:
            for idx, sug in enumerate(self.suggestions):
                self._draw_suggestion(painter, sug,
                                      selected=(idx == self.selected_suggestion_index))

        # live polygon drawing preview
        if self.mode == self.MODE_POLYGON and self.draw_points:
            self._draw_live_polygon(painter)

        # SAM click indicator
        if self.mode == self.MODE_SAM and self.preview_pt:
            self._draw_sam_cursor(painter, *self.preview_pt)

    def _draw_polygon(self, painter: QPainter, ann: PolygonAnnotation,
                      selected: bool):
        if len(ann.points) < 2:
            return
        color = self.ann_color(ann.class_id)
        pts = [QPointF(*self.img2disp(x, y)) for x, y in ann.points]
        poly = QPolygonF(pts)

        # fill
        fill_color = QColor(color)
        fill_color.setAlpha(self.alpha_fill)
        painter.setBrush(fill_color)
        pen_width = 3 if selected else 2
        painter.setPen(QPen(color, pen_width))
        painter.drawPolygon(poly)

        # label
        cx, cy = self.img2disp(*ann.centroid())
        label = self.class_name(ann.class_id)
        painter.setFont(QFont("Segoe UI", 9))
        fm = painter.fontMetrics()
        tw = fm.horizontalAdvance(label) + 8
        th = fm.height() + 4
        bg = QColor(color); bg.setAlpha(200)
        painter.fillRect(QRectF(cx - tw / 2, cy - th / 2, tw, th), bg)
        painter.setPen(Qt.black)
        painter.drawText(QRectF(cx - tw / 2, cy - th / 2, tw, th),
                         Qt.AlignCenter, label)

        # handles in edit mode
        if selected and self.mode == self.MODE_EDIT:
            painter.setBrush(Qt.white)
            painter.setPen(QPen(Qt.black, 1))
            r = HANDLE_RADIUS
            for pt in pts:
                painter.drawEllipse(pt, r, r)

    def _draw_suggestion(self, painter: QPainter, ann: PolygonAnnotation,
                         selected: bool):
        if len(ann.points) < 2:
            return
        color  = QColor(0, 220, 255)
        pts    = [QPointF(*self.img2disp(x, y)) for x, y in ann.points]
        poly   = QPolygonF(pts)
        fill   = QColor(0, 220, 255, 60)
        painter.setBrush(fill)
        pen = QPen(color, 3 if selected else 2, Qt.DashLine)
        painter.setPen(pen)
        painter.drawPolygon(poly)
        cx, cy = self.img2disp(*ann.centroid())
        label  = self.class_name(ann.class_id)
        score_txt = f" {ann.score:.2f}" if ann.score >= 0 else ""
        full_lbl = f"▶ {label}{score_txt}"
        painter.setFont(QFont("Segoe UI", 9))
        fm = painter.fontMetrics()
        tw = fm.horizontalAdvance(full_lbl) + 8
        th = fm.height() + 4
        painter.fillRect(QRectF(cx - tw / 2, cy - th / 2, tw, th),
                         QColor(0, 180, 220, 200))
        painter.setPen(Qt.black)
        painter.drawText(QRectF(cx - tw / 2, cy - th / 2, tw, th),
                         Qt.AlignCenter, full_lbl)

    def _draw_live_polygon(self, painter: QPainter):
        pts = [QPointF(*self.img2disp(x, y)) for x, y in self.draw_points]
        if self.preview_pt:
            pts.append(QPointF(*self.img2disp(*self.preview_pt)))
        poly = QPolygonF(pts)
        painter.setBrush(QColor(255, 200, 0, 50))
        painter.setPen(QPen(QColor(255, 200, 0), 2))
        painter.drawPolyline(poly)
        # close line
        if len(pts) >= 2:
            painter.drawLine(pts[-1], pts[0])
        # vertex dots
        painter.setBrush(QColor(255, 200, 0))
        for pt in pts[:-1]:
            painter.drawEllipse(pt, 4, 4)

    def _draw_sam_cursor(self, painter: QPainter, ix: float, iy: float):
        dx, dy = self.img2disp(ix, iy)
        painter.setPen(QPen(QColor(0, 255, 100), 2))
        painter.drawEllipse(QPointF(dx, dy), 8, 8)

    # ── mouse ────────────────────────────────────────────────────────────────
    def mousePressEvent(self, event):
        if not self.has_image():
            return
        ix, iy = self.clamp_img(*self.disp2img(event.x(), event.y()))
        btn = event.button()

        # pan
        if btn == Qt.MiddleButton or (btn == Qt.LeftButton and self.space_pressed):
            self.mode = self.MODE_PAN
            self.last_mouse = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            return

        # SAM mode: left-click = add prompt, right-click = negative prompt
        if self.mode == self.MODE_SAM:
            if btn == Qt.LeftButton:
                self.sam_click_requested.emit(ix, iy, True)
            elif btn == Qt.RightButton:
                self.sam_click_requested.emit(ix, iy, False)
            return

        # Edge annotation mode: left-click = segment at point
        if self.mode == self.MODE_EDGE:
            if btn == Qt.LeftButton:
                self.edge_click_requested.emit(ix, iy)
            return

        # polygon drawing mode
        if self.mode == self.MODE_POLYGON:
            if btn == Qt.RightButton:
                # cancel
                self.draw_points.clear()
                self.preview_pt = None
                self.update()
                return
            if btn == Qt.LeftButton:
                # close polygon on double-click handled in mouseDoubleClickEvent
                self.draw_points.append((ix, iy))
                self.update()
            return

        # edit / idle
        if btn == Qt.RightButton:
            # reject suggestion under cursor
            si = self._sug_at(ix, iy)
            if si >= 0:
                self.selected_suggestion_index = si
                self.reject_selected_suggestion()
            return

        if btn == Qt.LeftButton:
            # vertex drag in edit mode?
            if self.mode == self.MODE_EDIT and self.selected_indices:
                for ann_idx in self.selected_indices:
                    vi = self._vertex_at(ann_idx, ix, iy)
                    if vi >= 0:
                        self.push_history()
                        self.edit_ann_idx  = ann_idx
                        self.edit_pt_idx   = vi
                        self.edit_dragging = True
                        return

            # suggestion under cursor?
            si = self._sug_at(ix, iy)
            if si >= 0:
                self.selected_suggestion_index = si
                self.suggestions_changed.emit()
                self.update()
                # start dragging to promote
                self.drag_start_img = (ix, iy)
                self.drag_start_boxes = {}
                return

            # annotation under cursor?
            ai = self._ann_at(ix, iy)
            if ai >= 0:
                if ai not in self.selected_indices:
                    self.set_selected(ai)
                self.push_history()
                self.drag_start_img = (ix, iy)
                self.drag_start_boxes = {
                    i: [list(pt) for pt in self.annotations[i].points]
                    for i in self.selected_indices
                }
                self.setCursor(Qt.ClosedHandCursor)
                return
            # click on empty: deselect
            self.set_selected(-1)

    # expose sam_click_requested as a signal
    sam_click_requested = pyqtSignal(float, float, bool)

    def mouseMoveEvent(self, event):
        if not self.has_image():
            return
        ix, iy = self.clamp_img(*self.disp2img(event.x(), event.y()))
        self.mouse_pos_changed.emit(f"x={ix:.0f}  y={iy:.0f}  zoom={self.zoom:.2f}×")

        # pan
        if self.mode == self.MODE_PAN:
            d = event.pos() - self.last_mouse
            self.pan_x += d.x(); self.pan_y += d.y()
            self.last_mouse = event.pos()
            self.update(); return

        # polygon preview
        if self.mode == self.MODE_POLYGON:
            self.preview_pt = (ix, iy)
            self.update(); return

        # sam hover
        if self.mode == self.MODE_SAM:
            self.preview_pt = (ix, iy)
            self.update(); return

        # vertex drag
        if self.edit_dragging and self.edit_ann_idx >= 0:
            self.annotations[self.edit_ann_idx].points[self.edit_pt_idx] = (ix, iy)
            self.annotations_changed.emit()
            self.update(); return

        # polygon / suggestion drag
        if self.drag_start_img and self.drag_start_boxes is not None:
            dx = ix - self.drag_start_img[0]
            dy = iy - self.drag_start_img[1]
            for ann_idx, orig_pts in self.drag_start_boxes.items():
                if 0 <= ann_idx < len(self.annotations):
                    self.annotations[ann_idx].points = [
                        (p[0] + dx, p[1] + dy) for p in orig_pts]
            self.annotations_changed.emit()
            self.update(); return

    def mouseReleaseEvent(self, event):
        if self.mode == self.MODE_PAN:
            self.mode = self.MODE_EDIT
            self.setCursor(Qt.ArrowCursor)
        self.edit_dragging  = False
        self.edit_ann_idx   = -1
        self.edit_pt_idx    = -1
        self.drag_start_img = None
        self.drag_start_boxes = {}
        self.update()

    def mouseDoubleClickEvent(self, event):
        if not self.has_image():
            return
        ix, iy = self.clamp_img(*self.disp2img(event.x(), event.y()))

        # close polygon
        if self.mode == self.MODE_POLYGON and len(self.draw_points) >= MIN_POLYGON_PTS:
            self.push_history()
            ann = PolygonAnnotation(points=list(self.draw_points),
                                    class_id=self.current_class_id)
            self.annotations.append(ann)
            self.set_selected(len(self.annotations) - 1)
            self.draw_points.clear()
            self.preview_pt = None
            self.annotations_changed.emit()
            self.update(); return

        # accept suggestion
        si = self._sug_at(ix, iy)
        if si >= 0:
            self.selected_suggestion_index = si
            self.accept_selected_suggestion(); return

        # edit annotation class
        ai = self._ann_at(ix, iy)
        if ai >= 0:
            self.set_selected(ai)
            self.class_edit_requested.emit(self.annotations[ai].class_id)

    def wheelEvent(self, event):
        if not self.has_image():
            return
        bx, by = self.disp2img(event.x(), event.y())
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.zoom = max(0.1, min(self.zoom * factor, 25.0))
        self._update_metrics()
        ax, ay = self.img2disp(bx, by)
        self.pan_x += event.x() - ax
        self.pan_y += event.y() - ay
        self.update()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            self.space_pressed = True
            self.setCursor(Qt.OpenHandCursor); return
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            # In SAM/Edge mode: accept the pending suggestion
            if self.mode in (self.MODE_SAM, self.MODE_EDGE) and self.suggestions:
                self.accept_selected_suggestion()
                self.suggestions.clear()
                self.selected_suggestion_index = -1
                self.suggestions_changed.emit()
                self.update()
                return
            # In polygon mode: close the polygon
            if self.mode == self.MODE_POLYGON and len(self.draw_points) >= MIN_POLYGON_PTS:
                self.push_history()
                ann = PolygonAnnotation(points=list(self.draw_points),
                                        class_id=self.current_class_id)
                self.annotations.append(ann)
                self.set_selected(len(self.annotations) - 1)
                self.draw_points.clear()
                self.preview_pt = None
                self.annotations_changed.emit()
                self.update()
            return
        if event.key() == Qt.Key_Space and self.mode in (self.MODE_SAM, self.MODE_EDGE):
            # Space = accept suggestion (don't pan in SAM/Edge mode)
            if self.suggestions:
                self.accept_selected_suggestion()
                self.suggestions.clear()
                self.selected_suggestion_index = -1
                self.suggestions_changed.emit()
                self.update()
            return
        if event.key() == Qt.Key_Escape:
            # Reject any pending suggestion, clear drawing, go to edit
            if self.suggestions:
                self.suggestions.clear()
                self.selected_suggestion_index = -1
                self.suggestions_changed.emit()
            self.draw_points.clear()
            self.preview_pt = None
            self.set_mode(self.MODE_EDIT)
            self.update()
            return
        if event.key() == Qt.Key_Delete:
            self.delete_selected()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Space:
            self.space_pressed = False
            self.unsetCursor(); return
        super().keyReleaseEvent(event)

    def leaveEvent(self, _ev):
        self.mouse_pos_changed.emit("")


# ════════════════════════════════════════════════════════════════════════════
# Utility
# ════════════════════════════════════════════════════════════════════════════
def _atomic_write(path: str, lines: List[str]):
    directory = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(prefix=".seganno_", suffix=".tmp", dir=directory, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.writelines(lines)
        os.replace(tmp, path)
    except Exception:
        try: os.remove(tmp)
        except OSError: pass
        raise


def _natural_key(s: str):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", s)]


# ════════════════════════════════════════════════════════════════════════════
# Training dialog
# ════════════════════════════════════════════════════════════════════════════
class TrainingDialog(QDialog):
    """Non-blocking training dialog with live progress."""
    def __init__(self, cfg: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Train Segmentation Model")
        self.resize(480, 340)
        self._cfg   = cfg
        self._thread: Optional[QThread] = None
        self._worker = None

        layout = QVBoxLayout(self)
        self._status = QLabel("Ready to train.")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        self._progress = QProgressBar()
        self._progress.setRange(0, cfg.get("epochs", 20))
        layout.addWidget(self._progress)

        self._log = QTextEdit(); self._log.setReadOnly(True)
        self._log.setMaximumHeight(160)
        layout.addWidget(self._log)

        btn_row = QHBoxLayout()
        self._start_btn = QPushButton("Start Training")
        self._stop_btn  = QPushButton("Stop")
        self._stop_btn.setEnabled(False)
        self._close_btn = QPushButton("Close")
        self._start_btn.clicked.connect(self._start)
        self._stop_btn.clicked.connect(self._stop)
        self._close_btn.clicked.connect(self.reject)
        btn_row.addWidget(self._start_btn)
        btn_row.addWidget(self._stop_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(self._close_btn)
        layout.addLayout(btn_row)

    def _start(self):
        if not TORCH_AVAILABLE:
            QMessageBox.warning(self, "PyTorch missing",
                                "Install PyTorch: pip install torch torchvision")
            return
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._worker = TrainWorker(self._cfg)
        self._thread = QThread()
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._thread.start()

    def _stop(self):
        if self._worker:
            self._worker.stop()
        self._stop_btn.setEnabled(False)

    def _on_progress(self, epoch, total, loss, miou):
        self._progress.setValue(epoch)
        line = f"Epoch {epoch}/{total}  loss={loss:.4f}  mIoU={miou:.4f}"
        self._status.setText(line)
        self._log.append(line)

    def _on_finished(self, path):
        self._status.setText(f"✅ Training complete. Best model: {path}")
        self._log.append(f"Saved: {path}")
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._cfg["last_checkpoint"] = path
        if self._thread:
            self._thread.quit()

    def _on_error(self, msg):
        self._status.setText("❌ Training error (see log)")
        self._log.append(msg)
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        if self._thread:
            self._thread.quit()


# ════════════════════════════════════════════════════════════════════════════
# Main window
# ════════════════════════════════════════════════════════════════════════════
class SegAnnotator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.resize(1500, 900)

        # ── project state ───────────────────────────────────────────────────
        self.settings = QSettings(SETTINGS_ORG, SETTINGS_APP)
        self.folder:         str  = ""
        self.image_files:    List[str] = []
        self.cur_idx:        int  = -1
        self.label_cache:    dict = {}
        self.reviewed_cache: dict = {}
        self.labeled_count   = 0
        self.reviewed_count  = 0
        self.labels_dirty    = False

        # ── SAM state ───────────────────────────────────────────────────────
        self.sam_model   = None
        self.sam_predictor = None
        self.sam_path:   str  = ""
        self.sam_points: List[Tuple[float, float]] = []
        self.sam_labels: List[int]                 = []  # 1=fg, 0=bg
        self._smart_select_active: bool            = False

        # ── training / supervised model ──────────────────────────────────────
        self.seg_model        = None
        self.seg_model_path:  str  = ""
        self.seg_num_classes: int  = 0
        self.train_cfg:       dict = {}

        # ── build UI ─────────────────────────────────────────────────────────
        self._build_canvas()
        self._build_panels()
        self._build_layout()
        self._build_menus()
        self._build_shortcuts()
        self._apply_style()

        # ── restore session ──────────────────────────────────────────────────
        self._restore_settings()
        if self.folder and os.path.isdir(self.folder):
            self._open_folder_path(self.folder)

    # ════════════════════════════════════════════════════════════════════════
    # UI construction
    # ════════════════════════════════════════════════════════════════════════
    def _build_canvas(self):
        self.canvas = Canvas()
        self.canvas.annotations_changed.connect(self._on_annotations_changed)
        self.canvas.selection_changed.connect(self._on_selection_changed)
        self.canvas.mouse_pos_changed.connect(lambda t: self.status_bar.showMessage(t))
        self.canvas.suggestions_changed.connect(self._refresh_suggestion_list)
        self.canvas.sam_click_requested.connect(self._on_sam_click)
        self.canvas.edge_click_requested.connect(self._on_edge_click)
        self.canvas.class_edit_requested.connect(self._focus_class)

    def _build_panels(self):
        # ── left panel ──────────────────────────────────────────────────────
        self.file_list = QListWidget()
        self.file_list.currentRowChanged.connect(self._load_image_at_row)
        self.file_list.setTextElideMode(Qt.ElideRight)
        self.file_list.setUniformItemSizes(True)

        self.img_info    = QLabel("No folder opened")
        self.img_info.setWordWrap(True)
        self.dataset_info = QLabel("Dataset: 0 images")
        self.dataset_info.setWordWrap(True)

        self.sort_combo  = QComboBox()
        self.sort_combo.addItems(["Name A→Z", "Name natural", "Name Z→A",
                                  "Labeled first", "Unlabeled first"])
        self.sort_combo.currentIndexChanged.connect(self._re_sort)

        self.reviewed_check = QCheckBox("Reviewed")
        self.reviewed_check.toggled.connect(self._toggle_reviewed)

        prev_btn = QPushButton("◀ Prev"); prev_btn.clicked.connect(self._prev_image)
        next_btn = QPushButton("Next ▶"); next_btn.clicked.connect(self._next_image)
        skip_btn = QPushButton("Next empty"); skip_btn.clicked.connect(self._next_unlabeled)

        # ── right panel: tabs ────────────────────────────────────────────────
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_project_tab(),  "Project")
        self.tabs.addTab(self._build_annotate_tab(), "Annotate")
        self.tabs.addTab(self._build_train_tab(),    "Train")

        # ── left panel assembly ──────────────────────────────────────────────
        self.left_panel = QWidget()
        self.left_panel.setMinimumWidth(220)
        self.left_panel.setMaximumWidth(300)
        ll = QVBoxLayout(self.left_panel)
        ll.addWidget(QLabel("Images"))
        ll.addWidget(self.file_list)
        ll.addWidget(self.img_info)
        ll.addWidget(self.dataset_info)
        ll.addWidget(self.sort_combo)
        nav = QHBoxLayout()
        nav.addWidget(prev_btn); nav.addWidget(next_btn)
        ll.addLayout(nav)
        ll.addWidget(skip_btn)
        ll.addWidget(self.reviewed_check)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.addPermanentWidget(
            QLabel(f"{APP_NAME} v{APP_VERSION}  |  Developed by {APP_DEVELOPER}  "))

    def _build_project_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        # classes group
        cg = QGroupBox("Classes")
        cv = QVBoxLayout(cg)
        self.class_list = QListWidget()
        self.class_list.currentRowChanged.connect(self._on_class_selected)
        cv.addWidget(self.class_list)
        cr = QHBoxLayout()
        for label, fn in [("Add", self._add_class), ("Rename", self._rename_class),
                          ("Delete", self._delete_class)]:
            b = QPushButton(label); b.clicked.connect(fn); cr.addWidget(b)
        cv.addLayout(cr)
        v.addWidget(cg)
        # annotations group
        ag = QGroupBox("Annotations")
        av = QVBoxLayout(ag)
        self.ann_list = QListWidget()
        self.ann_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.ann_list.itemSelectionChanged.connect(self._on_ann_list_selection)
        av.addWidget(self.ann_list)
        v.addWidget(ag)
        # suggestions group
        sg = QGroupBox("Suggestions")
        sv = QVBoxLayout(sg)
        self.sug_info = QLabel("Suggestions: 0")
        self.sug_list = QListWidget()
        self.sug_list.currentRowChanged.connect(self._on_sug_list_click)
        sv.addWidget(self.sug_info)
        sv.addWidget(self.sug_list)
        srow = QHBoxLayout()
        for label, fn in [("✔ Accept", self._accept_sel_sug),
                          ("✔ All",    self._accept_all_sug),
                          ("✗ Reject", self._reject_sel_sug),
                          ("✗ All",    self._reject_all_sug)]:
            b = QPushButton(label); b.clicked.connect(fn); srow.addWidget(b)
        sv.addLayout(srow)
        v.addWidget(sg)
        # save button
        save_btn = QPushButton("💾 Save labels")
        save_btn.clicked.connect(self._save_labels)
        v.addWidget(save_btn)
        v.addStretch(1)
        return w

    def _build_annotate_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)

        # drawing mode group
        mg = QGroupBox("Drawing Mode")
        mv = QVBoxLayout(mg)
        row1 = QHBoxLayout()
        row2 = QHBoxLayout()
        self.btn_edit    = QPushButton("✏ Edit (E)");         self.btn_edit.setCheckable(True)
        self.btn_polygon = QPushButton("⬡ Polygon (P)");      self.btn_polygon.setCheckable(True)
        self.btn_sam     = QPushButton("🎯 SAM (S)");          self.btn_sam.setCheckable(True)
        self.btn_smart   = QPushButton("⚡ Smart Select (Q)"); self.btn_smart.setCheckable(True)
        self.btn_smart.setObjectName("primaryButton")
        self.btn_smart.setToolTip(
            "Smart Select: single left-click instantly segments and accepts the object.\n"
            "Like Roboflow's smart polygon tool.\n"
            "Right-click to add a background exclusion point (refine)."
        )
        self.btn_edit.setChecked(True)
        self.btn_edge = QPushButton("🔍 Edge Fill (W)"); self.btn_edge.setCheckable(True)
        self.btn_edge.setToolTip(
            "Edge-detection annotator — no SAM needed.\n"
            "Click anywhere on an object to fill it using edge barriers\n"
            "and color similarity. Works best on objects with clear edges.\n"
            "Adjust Edge Fill settings below.")
        row1.addWidget(self.btn_edit)
        row1.addWidget(self.btn_polygon)
        row2.addWidget(self.btn_sam)
        row2.addWidget(self.btn_smart)
        row3 = QHBoxLayout()
        row3.addWidget(self.btn_edge)
        mv.addLayout(row1)
        mv.addLayout(row2)
        mv.addLayout(row3)
        self.btn_edit.clicked.connect(lambda: self._set_draw_mode("edit"))
        self.btn_polygon.clicked.connect(lambda: self._set_draw_mode("polygon"))
        self.btn_sam.clicked.connect(lambda: self._set_draw_mode("sam"))
        self.btn_smart.clicked.connect(lambda: self._set_draw_mode("smart"))
        self.btn_edge.clicked.connect(lambda: self._set_draw_mode("edge"))
        v.addWidget(mg)

        # SAM group
        sg = QGroupBox("Segment Anything (SAM)")
        sv = QVBoxLayout(sg)
        self.sam_info = QLabel("SAM: not loaded")
        self.sam_info.setWordWrap(True)
        sv.addWidget(self.sam_info)

        load_sam_btn = QPushButton("Load SAM checkpoint (.pth / .pt)")
        load_sam_btn.clicked.connect(self._load_sam)
        sv.addWidget(load_sam_btn)

        # Model-type selector — "Auto-detect" is the first (default) entry.
        # Below it we list every type found across all installed SAM libraries.
        sv.addWidget(QLabel("SAM model type (Auto-detect recommended):"))
        self.sam_type_combo = QComboBox()
        self._populate_sam_type_combo()
        sv.addWidget(self.sam_type_combo)

        # Install hints shown only when nothing is installed
        if not SAM_AVAILABLE:
            hint = QLabel(
                "⚠ No SAM library found.\n"
                "Install one:\n"
                "  pip install mobile-sam\n"
                "  pip install segment-anything"
            )
            hint.setStyleSheet("color: #b45309; font-size: 10px;")
            hint.setWordWrap(True)
            sv.addWidget(hint)

        # Smart Select coverage — controls how large the box hint is
        cov_row = QHBoxLayout()
        cov_row.addWidget(QLabel("Smart coverage:"))
        self.smart_coverage_spin = QSpinBox()
        self.smart_coverage_spin.setRange(10, 100)
        self.smart_coverage_spin.setValue(40)
        self.smart_coverage_spin.setSuffix(" %")
        self.smart_coverage_spin.setToolTip(
            "Controls the search area for Smart Select.\n"
            "Increase if the full object is not captured.\n"
            "Decrease if too much background is included.")
        cov_row.addWidget(self.smart_coverage_spin)
        sv.addLayout(cov_row)

        # ── Edge-snap controls (Roboflow-style boundary snapping) ────────────
        edge_grp = QGroupBox("Edge Snapping  (experimental)")
        edge_grp.setCheckable(True)
        edge_grp.setChecked(False)   # off by default — mask smoothing is enough
        self.edge_snap_group = edge_grp
        eg = QFormLayout(edge_grp)

        self.snap_radius_spin = QSpinBox()
        self.snap_radius_spin.setRange(1, 20)
        self.snap_radius_spin.setValue(4)
        self.snap_radius_spin.setToolTip(
            "Max pixels a vertex will move to snap to a nearby edge.\n"
            "Keep small (3-6) to avoid snapping to unrelated edges.\n"
            "Vertices only snap within the mask boundary band.")
        eg.addRow("Snap radius (px):", self.snap_radius_spin)

        self.canny_low_spin = QSpinBox()
        self.canny_low_spin.setRange(1, 255)
        self.canny_low_spin.setValue(30)
        self.canny_low_spin.setToolTip("Canny lower threshold — lower = more edges detected")
        eg.addRow("Edge sensitivity (low):", self.canny_low_spin)

        self.canny_high_spin = QSpinBox()
        self.canny_high_spin.setRange(1, 255)
        self.canny_high_spin.setValue(100)
        self.canny_high_spin.setToolTip("Canny upper threshold — lower = more edges detected")
        eg.addRow("Edge sensitivity (high):", self.canny_high_spin)

        sv.addWidget(edge_grp)

        # ── Edge Fill settings ────────────────────────────────────────────
        ef_grp = QGroupBox("🔍 Edge Fill Settings")
        ef_grp.setToolTip("Settings for the Edge Fill annotator (W key).\n"
                          "No SAM required — uses edge detection + color flood-fill.")
        efv = QFormLayout(ef_grp)

        self.ef_canny_lo = QSpinBox(); self.ef_canny_lo.setRange(1,255)
        self.ef_canny_lo.setValue(20)
        self.ef_canny_lo.setToolTip("Lower Canny threshold — lower = more edges = tighter fill")
        efv.addRow("Edge detect (low):", self.ef_canny_lo)

        self.ef_canny_hi = QSpinBox(); self.ef_canny_hi.setRange(1,255)
        self.ef_canny_hi.setValue(80)
        self.ef_canny_hi.setToolTip("Upper Canny threshold")
        efv.addRow("Edge detect (high):", self.ef_canny_hi)

        self.ef_dilate = QSpinBox(); self.ef_dilate.setRange(0,10)
        self.ef_dilate.setValue(4)
        self.ef_dilate.setToolTip("Barrier wall thickness — increase to plug edge gaps")
        efv.addRow("Edge gap fill (px):", self.ef_dilate)

        self.ef_color_tol = QSpinBox(); self.ef_color_tol.setRange(1,100)
        self.ef_color_tol.setValue(20)
        self.ef_color_tol.setToolTip(
            "Color tolerance (LAB space).\n"
            "Higher = more pixels included (good for large uniform surfaces).\n"
            "Lower = tighter color match (good for distinguishing similar objects).")
        efv.addRow("Color tolerance:", self.ef_color_tol)

        sv.addWidget(ef_grp)

        clear_pts_btn = QPushButton("Clear SAM points")
        clear_pts_btn.clicked.connect(self._clear_sam_points)
        sv.addWidget(clear_pts_btn)

        confirm_sam_btn = QPushButton("Confirm SAM mask → Suggestion")
        confirm_sam_btn.clicked.connect(self._confirm_sam_mask)
        confirm_sam_btn.setObjectName("primaryButton")
        sv.addWidget(confirm_sam_btn)

        self.sam_alpha_spin = QSpinBox()
        self.sam_alpha_spin.setRange(0, 255)
        self.sam_alpha_spin.setValue(80)
        self.sam_alpha_spin.setPrefix("Fill α: ")
        self.sam_alpha_spin.valueChanged.connect(
            lambda v: setattr(self.canvas, "alpha_fill", v) or self.canvas.update())
        sv.addWidget(self.sam_alpha_spin)
        v.addWidget(sg)

        # polygon help
        ph = QGroupBox("Polygon tips")
        pl = QVBoxLayout(ph)
        pl.addWidget(QLabel("• Click to add vertices\n"
                            "• Double-click or Enter to close\n"
                            "• Right-click or Esc to cancel\n"
                            "• Edit mode: drag vertex handles"))
        v.addWidget(ph)
        v.addStretch(1)
        return w

    def _build_train_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)

        # supervised model group
        mod_g = QGroupBox("Supervised Segmentation Model")
        mod_v = QVBoxLayout(mod_g)
        self.model_info_lbl = QLabel("Model: not loaded")
        self.model_info_lbl.setWordWrap(True)
        mod_v.addWidget(self.model_info_lbl)
        load_mod_btn = QPushButton("Load checkpoint (.pt)")
        load_mod_btn.clicked.connect(self._load_seg_model)
        mod_v.addWidget(load_mod_btn)
        suggest_btn = QPushButton("▶ Generate model suggestions")
        suggest_btn.setObjectName("primaryButton")
        suggest_btn.clicked.connect(self._model_suggest_current)
        mod_v.addWidget(suggest_btn)
        v.addWidget(mod_g)

        # DeepLabV3+ pretrained suggestions group
        dl_g = QGroupBox("🧠 DeepLabV3+ Pretrained Suggestions")
        dl_g.setToolTip(
            "Uses a torchvision DeepLabV3+ model pretrained on COCO/VOC to generate\n"
            "polygon suggestions for the current image — no training needed.\n"
            "Polygons are class-agnostic: you assign classes yourself afterward.")
        dl_v = QVBoxLayout(dl_g)

        self.dl_info_lbl = QLabel(
            "Model: not loaded\n"
            "★ For road/pavement segmentation: use SegFormer Cityscapes\n"
            "   pip install transformers   (then restart app)\n"
            "   Downloads ~14-84 MB from HuggingFace — much faster than 208 MB FCN")
        self.dl_info_lbl.setWordWrap(True)
        dl_v.addWidget(self.dl_info_lbl)

        dl_row1 = QHBoxLayout()
        dl_backbone_lbl = QLabel("Backbone:")
        self.dl_backbone_combo = QComboBox()
        self.dl_backbone_combo.addItems([name for name, *_ in SegAnnotator._DL_MODELS])
        self.dl_backbone_combo.setCurrentIndex(2)   # default: SegFormer-B0 (fastest, best for roads)
        self.dl_backbone_combo.setToolTip(
            "🏙★ SegFormer Cityscapes (TOP 3) — BEST for road scenes:\n"
            "   Has road, sidewalk, car, building, vegetation, sky etc.\n"
            "   Same quality as MATLAB CamVid model.\n"
            "   Requires:  pip install transformers\n"
            "   Downloads ~14-84 MB from HuggingFace on first use.\n\n"
            "🚗 COCO models (bottom 3) — detects vehicles/people only.\n"
            "   NO road class. Use only if transformers is not installed."
        )
        dl_row1.addWidget(dl_backbone_lbl)
        dl_row1.addWidget(self.dl_backbone_combo, 1)
        dl_v.addLayout(dl_row1)

        dl_row2 = QHBoxLayout()
        dl_conf_lbl = QLabel("Min confidence:")
        self.dl_conf_spin = QDoubleSpinBox()
        self.dl_conf_spin.setRange(0.05, 0.99)
        self.dl_conf_spin.setValue(0.4)
        self.dl_conf_spin.setSingleStep(0.05)
        self.dl_conf_spin.setDecimals(2)
        self.dl_conf_spin.setToolTip(
            "Only segments with softmax confidence above this threshold are kept.\n"
            "Lower = more (noisier) suggestions. Higher = fewer but cleaner.")
        dl_row2.addWidget(dl_conf_lbl)
        dl_row2.addWidget(self.dl_conf_spin)

        dl_minarea_lbl = QLabel("Min area (px):")
        self.dl_minarea_spin = QSpinBox()
        self.dl_minarea_spin.setRange(100, 50000)
        self.dl_minarea_spin.setValue(2000)
        self.dl_minarea_spin.setSingleStep(500)
        self.dl_minarea_spin.setToolTip("Ignore segments smaller than this many pixels.")
        dl_row2.addWidget(dl_minarea_lbl)
        dl_row2.addWidget(self.dl_minarea_spin)
        dl_v.addLayout(dl_row2)

        self.dl_suggest_btn = QPushButton("🧠 Generate DeepLab Suggestions")
        self.dl_suggest_btn.setObjectName("primaryButton")
        self.dl_suggest_btn.clicked.connect(self._deeplab_suggest_current)
        dl_v.addWidget(self.dl_suggest_btn)

        self.dl_all_btn = QPushButton("⚡ Generate for ALL images (batch)")
        self.dl_all_btn.setToolTip(
            "Runs DeepLabV3+ on every unlabeled image and saves polygon suggestions.\n"
            "You then review and assign classes image by image.")
        self.dl_all_btn.clicked.connect(self._deeplab_suggest_all)
        dl_v.addWidget(self.dl_all_btn)

        v.addWidget(dl_g)

        # training group
        tr_g = QGroupBox("Train New Model")
        tr_v = QFormLayout(tr_g)
        self.train_epochs  = QSpinBox();        self.train_epochs.setRange(1, 500);  self.train_epochs.setValue(20)
        self.train_batch   = QSpinBox();        self.train_batch.setRange(1, 64);    self.train_batch.setValue(4)
        self.train_lr      = QDoubleSpinBox();  self.train_lr.setRange(1e-5, 0.1);   self.train_lr.setValue(1e-3)
        self.train_lr.setDecimals(5)
        self.train_imgsize = QSpinBox();        self.train_imgsize.setRange(64, 1024); self.train_imgsize.setValue(512)
        self.train_imgsize.setSingleStep(32)
        self.train_savedir = QPushButton("Choose save dir…")
        self.train_savedir.clicked.connect(self._choose_save_dir)
        self._train_save_dir_path = ""
        tr_v.addRow("Epochs:",    self.train_epochs)
        tr_v.addRow("Batch:",     self.train_batch)
        tr_v.addRow("LR:",        self.train_lr)
        tr_v.addRow("Img size:",  self.train_imgsize)
        tr_v.addRow("Save to:",   self.train_savedir)
        start_train_btn = QPushButton("🚀 Start training")
        start_train_btn.setObjectName("primaryButton")
        start_train_btn.clicked.connect(self._start_training)
        tr_v.addRow("", start_train_btn)
        v.addWidget(tr_g)
        v.addStretch(1)
        return w

    def _build_layout(self):
        self.splitter = QSplitter()
        self.splitter.addWidget(self.left_panel)
        self.splitter.addWidget(self.canvas)
        self.splitter.addWidget(self.tabs)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setStretchFactor(2, 0)
        self.splitter.setSizes([250, 1200, 330])
        self.setCentralWidget(self.splitter)

    def _build_menus(self):
        mb = self.menuBar()
        fm = mb.addMenu("File")
        for label, fn in [
            ("Open Folder…",   self._open_folder),
            ("Save Labels",    self._save_labels),
            ("Export classes…",self._export_classes),
        ]:
            a = QAction(label, self); a.triggered.connect(fn); fm.addAction(a)
        vm = mb.addMenu("View")
        self.show_ann_action = QAction("Show Annotations", self, checkable=True, checked=True)
        self.show_sug_action = QAction("Show Suggestions", self, checkable=True, checked=True)
        self.show_ann_action.toggled.connect(lambda v: setattr(self.canvas, "show_annotations", v) or self.canvas.update())
        self.show_sug_action.toggled.connect(lambda v: setattr(self.canvas, "show_suggestions", v) or self.canvas.update())
        vm.addAction(self.show_ann_action)
        vm.addAction(self.show_sug_action)
        hm = mb.addMenu("Help")
        hm.addAction(QAction("About", self, triggered=self._show_about))
        hm.addAction(QAction("Help",  self, triggered=self._show_help))

    def _build_shortcuts(self):
        sc_map = {
            DEFAULT_SHORTCUTS["open_folder"]:       self._open_folder,
            DEFAULT_SHORTCUTS["save"]:              self._save_labels,
            DEFAULT_SHORTCUTS["undo"]:              self.canvas.undo,
            DEFAULT_SHORTCUTS["redo"]:              self.canvas.redo,
            DEFAULT_SHORTCUTS["delete"]:            self.canvas.delete_selected,
            DEFAULT_SHORTCUTS["copy"]:              self.canvas.copy_selected,
            DEFAULT_SHORTCUTS["paste"]:             self.canvas.paste,
            DEFAULT_SHORTCUTS["duplicate"]:         self.canvas.duplicate_selected,
            DEFAULT_SHORTCUTS["accept_suggestion"]: self._accept_sel_sug,
            DEFAULT_SHORTCUTS["accept_all"]:        self._accept_all_sug,
            DEFAULT_SHORTCUTS["reject_selected"]:   self._reject_sel_sug,
            DEFAULT_SHORTCUTS["reject_all"]:        self._reject_all_sug,
            DEFAULT_SHORTCUTS["fit"]:               self.canvas.fit_to_window,
            DEFAULT_SHORTCUTS["actual_size"]:       self.canvas.actual_size,
            DEFAULT_SHORTCUTS["prev_image"]:        self._prev_image,
            DEFAULT_SHORTCUTS["next_image"]:        self._next_image,
            DEFAULT_SHORTCUTS["next_unlabeled"]:    self._next_unlabeled,
            DEFAULT_SHORTCUTS["zoom_in"]:           self.canvas.zoom_in if hasattr(self.canvas, "zoom_in") else lambda: None,
            DEFAULT_SHORTCUTS["zoom_out"]:          self.canvas.zoom_out if hasattr(self.canvas, "zoom_out") else lambda: None,
            DEFAULT_SHORTCUTS["toggle_reviewed"]:   lambda: self.reviewed_check.setChecked(not self.reviewed_check.isChecked()),
            DEFAULT_SHORTCUTS["select_all"]:        self.canvas.select_all,
            DEFAULT_SHORTCUTS["sam_mode"]:          lambda: self._set_draw_mode("sam"),
            DEFAULT_SHORTCUTS["polygon_mode"]:      lambda: self._set_draw_mode("polygon"),
            DEFAULT_SHORTCUTS["edit_mode"]:         lambda: self._set_draw_mode("edit"),
            "Q":                                    lambda: self._set_draw_mode("smart"),
            "W":                                    lambda: self._set_draw_mode("edge"),
        }
        for seq, fn in sc_map.items():
            if not seq:
                continue
            sh = QShortcut(QKeySequence(seq), self)
            sh.setContext(Qt.ApplicationShortcut)
            sh.activated.connect(fn)

        # add zoom helpers if not present on canvas
        if not hasattr(self.canvas, "zoom_in"):
            def _zi(): self.canvas.zoom = min(self.canvas.zoom * 1.15, 25.0); self.canvas.update()
            def _zo(): self.canvas.zoom = max(self.canvas.zoom / 1.15, 0.1);  self.canvas.update()
            self.canvas.zoom_in  = _zi
            self.canvas.zoom_out = _zo

    def _apply_style(self):
        self.setStyleSheet("""
            QMainWindow { background: #f3f4f6; }
            QMenuBar { background:#fff; border-bottom:1px solid #d7dce2; padding:4px 8px; }
            QMenuBar::item { padding:5px 10px; border-radius:5px; }
            QMenuBar::item:selected { background:#e6eef8; }
            QMenu { background:#fff; border:1px solid #d7dce2; padding:4px; }
            QMenu::item { padding:5px 16px; border-radius:4px; }
            QMenu::item:selected { background:#e6eef8; }
            QGroupBox { border:1px solid #d7dce2; border-radius:8px; margin-top:10px;
                        padding-top:10px; background:#fbfbfc; font-weight:600; }
            QGroupBox::title { subcontrol-origin:margin; left:10px; padding:0 4px; color:#374151; }
            QPushButton { background:#fff; border:1px solid #cbd5e1; border-radius:7px;
                          padding:5px 10px; color:#1f2937; }
            QPushButton:hover { background:#f8fafc; border-color:#94a3b8; }
            QPushButton:checked { background:#dbeafe; border-color:#2563eb; color:#1d4ed8; font-weight:700; }
            QPushButton#primaryButton { background:#198754; border:1px solid #157347; color:white; font-weight:700; }
            QPushButton#primaryButton:hover { background:#157347; }
            QTabBar::tab { background:#eef2f7; border:1px solid #d7dce2; padding:6px 12px; margin-right:3px; }
            QTabBar::tab:selected { background:#fff; border-bottom-color:#fff; }
        """)

    # ════════════════════════════════════════════════════════════════════════
    # Folder / image management
    # ════════════════════════════════════════════════════════════════════════
    def _open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Open image folder",
                                                  self.folder or os.getcwd())
        if folder:
            self._open_folder_path(folder)

    def _open_folder_path(self, folder: str):
        self.folder = folder
        self.image_files = sorted(
            [e.name for e in os.scandir(folder)
             if e.is_file() and e.name.lower().endswith(IMAGE_EXTENSIONS)],
            key=_natural_key)
        self._load_classes()
        self._rebuild_cache()
        self._refresh_file_list()
        self.canvas.set_classes(self.canvas.classes)
        self.settings.setValue("last_folder", folder)
        if self.image_files:
            self._load_image_at_row(0)

    def _load_classes(self):
        for name in CLASS_FILE_NAMES:
            path = os.path.join(self.folder, name)
            if os.path.exists(path):
                with open(path, encoding="utf-8") as fh:
                    classes = [l.strip() for l in fh if l.strip()]
                self.canvas.set_classes(classes)
                self._refresh_class_list()
                return
        self.canvas.set_classes([])
        self._refresh_class_list()

    def _save_classes(self):
        if not self.folder:
            return
        path = os.path.join(self.folder, "classes.txt")
        _atomic_write(path, [c + "\n" for c in self.canvas.classes])

    def _rebuild_cache(self):
        self.label_cache    = {}
        self.reviewed_cache = {}
        for name in self.image_files:
            lp = self._label_path(name)
            self.label_cache[name]    = os.path.exists(lp) and os.path.getsize(lp) > 0
            self.reviewed_cache[name] = os.path.exists(self._reviewed_path(name))
        self.labeled_count  = sum(self.label_cache.values())
        self.reviewed_count = sum(self.reviewed_cache.values())

    def _label_path(self, name: str) -> str:
        return os.path.splitext(os.path.join(self.folder, name))[0] + ".txt"

    def _reviewed_path(self, name: str) -> str:
        return os.path.join(self.folder, name + REVIEW_SUFFIX)

    def _decorate(self, name: str) -> str:
        flags = []
        if self.label_cache.get(name):    flags.append("L")
        if self.reviewed_cache.get(name): flags.append("R")
        prefix = f"[{''.join(flags)}] " if flags else ""
        return prefix + name

    def _refresh_file_list(self):
        self.file_list.blockSignals(True)
        self.file_list.clear()
        for name in self.image_files:
            self.file_list.addItem(self._decorate(name))
        self.file_list.blockSignals(False)
        self.dataset_info.setText(
            f"Dataset: {len(self.image_files)} | "
            f"labeled {self.labeled_count} | reviewed {self.reviewed_count}")

    def _re_sort(self, _idx=None):
        mode = self.sort_combo.currentText()
        cur = self.image_files[self.cur_idx] if self.cur_idx >= 0 and self.image_files else None
        if mode.startswith("Name A"):
            self.image_files.sort(key=lambda n: n.lower())
        elif "natural" in mode:
            self.image_files.sort(key=_natural_key)
        elif mode.startswith("Name Z"):
            self.image_files.sort(key=lambda n: n.lower(), reverse=True)
        elif "Labeled first" in mode:
            self.image_files.sort(key=lambda n: not self.label_cache.get(n))
        elif "Unlabeled first" in mode:
            self.image_files.sort(key=lambda n: self.label_cache.get(n))
        self._refresh_file_list()
        if cur and cur in self.image_files:
            self.file_list.setCurrentRow(self.image_files.index(cur))

    def _cur_name(self) -> str:
        if 0 <= self.cur_idx < len(self.image_files):
            return self.image_files[self.cur_idx]
        return ""

    def _cur_path(self) -> str:
        name = self._cur_name()
        return os.path.join(self.folder, name) if name else ""

    def _load_image_at_row(self, row: int):
        if row < 0 or row >= len(self.image_files):
            return
        # auto-save current
        if self.labels_dirty and self.canvas.has_image():
            self._save_labels()
        self.cur_idx = row
        path = os.path.join(self.folder, self.image_files[row])
        try:
            self.canvas.load_image(path)
        except ValueError as exc:
            QMessageBox.warning(self, "Load failed", str(exc))
            return
        # update SAM image embedding
        if self.sam_predictor is not None:
            self._update_sam_image()
        self.img_info.setText(f"{row + 1}/{len(self.image_files)}  {self.image_files[row]}")
        self.reviewed_check.blockSignals(True)
        self.reviewed_check.setChecked(self.reviewed_cache.get(self.image_files[row], False))
        self.reviewed_check.blockSignals(False)
        self.sam_points.clear()
        self.sam_labels.clear()
        self._refresh_ann_list()
        self.labels_dirty = False
        self._update_title()

    def _prev_image(self):
        if self.cur_idx > 0:
            self.file_list.setCurrentRow(self.cur_idx - 1)

    def _next_image(self):
        if self.cur_idx < len(self.image_files) - 1:
            self.file_list.setCurrentRow(self.cur_idx + 1)

    def _next_unlabeled(self):
        start = max(self.cur_idx, -1) + 1
        for offset in range(len(self.image_files)):
            idx = (start + offset) % len(self.image_files)
            if not self.label_cache.get(self.image_files[idx]):
                self.file_list.setCurrentRow(idx)
                return
        self.status_bar.showMessage("All images are labeled.", 3000)

    # ════════════════════════════════════════════════════════════════════════
    # Save / reviewed
    # ════════════════════════════════════════════════════════════════════════
    def _save_labels(self):
        if not self.canvas.has_image():
            return
        self.canvas.save_labels()
        name = self._cur_name()
        if name:
            has = len(self.canvas.annotations) > 0
            old = self.label_cache.get(name, False)
            self.label_cache[name] = has
            if old != has:
                self.labeled_count += (1 if has else -1)
            self._update_file_item(name)
            self._refresh_file_list()
        self.labels_dirty = False
        self._update_title()
        self.status_bar.showMessage("Labels saved.", 2000)

    def _toggle_reviewed(self, checked: bool):
        name = self._cur_name()
        if not name:
            return
        rp = self._reviewed_path(name)
        try:
            if checked:
                _atomic_write(rp, ["reviewed\n"])
            elif os.path.exists(rp):
                os.remove(rp)
        except OSError as exc:
            QMessageBox.warning(self, "Reviewed", str(exc))
            return
        old = self.reviewed_cache.get(name, False)
        self.reviewed_cache[name] = checked
        self.reviewed_count += (1 if checked and not old else
                                -1 if not checked and old else 0)
        self._update_file_item(name)
        self.dataset_info.setText(
            f"Dataset: {len(self.image_files)} | "
            f"labeled {self.labeled_count} | reviewed {self.reviewed_count}")

    def _update_file_item(self, name: str):
        try:
            row = self.image_files.index(name)
        except ValueError:
            return
        item = self.file_list.item(row)
        if item:
            item.setText(self._decorate(name))

    def _update_title(self):
        suffix = " *" if self.labels_dirty else ""
        self.setWindowTitle(f"{APP_NAME}{suffix}")

    # ════════════════════════════════════════════════════════════════════════
    # Classes
    # ════════════════════════════════════════════════════════════════════════
    def _refresh_class_list(self):
        self.class_list.blockSignals(True)
        self.class_list.clear()
        for idx, name in enumerate(self.canvas.classes):
            color = self.canvas.ann_color(idx)
            item = QListWidgetItem(f"[{idx}] {name}")
            item.setBackground(QColor(color.red(), color.green(), color.blue(), 60))
            self.class_list.addItem(item)
        self.class_list.blockSignals(False)

    def _on_class_selected(self, row: int):
        if row >= 0:
            self.canvas.current_class_id = row
            if self.canvas.selected_indices:
                self.canvas.set_class_for_selected(row)

    def _add_class(self):
        name, ok = QInputDialog.getText(self, "Add Class", "Class name:")
        if ok and name.strip():
            classes = list(self.canvas.classes) + [name.strip()]
            self.canvas.set_classes(classes)
            self._save_classes()
            self._refresh_class_list()

    def _rename_class(self):
        row = self.class_list.currentRow()
        if row < 0 or row >= len(self.canvas.classes):
            return
        name, ok = QInputDialog.getText(self, "Rename Class", "New name:",
                                        text=self.canvas.classes[row])
        if ok and name.strip():
            classes = list(self.canvas.classes)
            classes[row] = name.strip()
            self.canvas.set_classes(classes)
            self._save_classes()
            self._refresh_class_list()
            self.canvas.update()

    def _delete_class(self):
        row = self.class_list.currentRow()
        if row < 0 or row >= len(self.canvas.classes):
            return
        if len(self.canvas.classes) <= 1:
            QMessageBox.information(self, "Delete class", "Need at least one class.")
            return
        ans = QMessageBox.question(self, "Delete class",
                                   f"Delete '{self.canvas.classes[row]}'? "
                                   f"All labels with this class will be removed.")
        if ans != QMessageBox.Yes:
            return
        self.canvas.push_history()
        cls_id = row
        self.canvas.annotations = [a for a in self.canvas.annotations if a.class_id != cls_id]
        # shift
        for a in self.canvas.annotations:
            if a.class_id > cls_id:
                a.class_id -= 1
        classes = list(self.canvas.classes)
        classes.pop(cls_id)
        self.canvas.set_classes(classes)
        self._save_classes()
        self._refresh_class_list()
        self.canvas.annotations_changed.emit()

    def _focus_class(self, class_id: int):
        self.tabs.setCurrentIndex(0)
        self.class_list.setCurrentRow(class_id)

    def _export_classes(self):
        if not self.canvas.classes:
            QMessageBox.information(self, "Export", "No classes defined."); return
        path, _ = QFileDialog.getSaveFileName(self, "Export classes",
                                              os.path.join(self.folder or ".", "classes.txt"),
                                              "Text files (*.txt)")
        if path:
            _atomic_write(path, [c + "\n" for c in self.canvas.classes])
            self.status_bar.showMessage(f"Classes exported to {path}", 3000)

    # ════════════════════════════════════════════════════════════════════════
    # Annotation list
    # ════════════════════════════════════════════════════════════════════════
    def _refresh_ann_list(self):
        self.ann_list.blockSignals(True)
        self.ann_list.clear()
        for idx, ann in enumerate(self.canvas.annotations):
            color = self.canvas.ann_color(ann.class_id)
            name  = self.canvas.class_name(ann.class_id)
            n_pts = len(ann.points)
            item  = QListWidgetItem(f"{idx + 1}. {name}  ({n_pts} pts)")
            item.setBackground(QColor(color.red(), color.green(), color.blue(), 50))
            self.ann_list.addItem(item)
            if idx in self.canvas.selected_indices:
                item.setSelected(True)
        self.ann_list.blockSignals(False)
        self._refresh_suggestion_list()

    def _refresh_suggestion_list(self):
        self.sug_list.blockSignals(True)
        self.sug_list.clear()
        for idx, sug in enumerate(self.canvas.suggestions):
            name  = self.canvas.class_name(sug.class_id)
            sc    = f"{sug.score:.2f}" if sug.score >= 0 else "--"
            item  = QListWidgetItem(f"{idx + 1}. {name} (conf {sc}, {len(sug.points)} pts)")
            item.setForeground(QColor(0, 150, 200))
            self.sug_list.addItem(item)
        self.sug_list.blockSignals(False)
        self.sug_info.setText(f"Suggestions: {len(self.canvas.suggestions)}")

    def _on_ann_list_selection(self):
        rows = {self.ann_list.row(i) for i in self.ann_list.selectedItems()}
        self.canvas.selected_indices = rows
        self.canvas.update()

    def _on_sug_list_click(self, row: int):
        self.canvas.selected_suggestion_index = row
        self.canvas.update()

    def _on_selection_changed(self, idx: int):
        self.ann_list.blockSignals(True)
        self.ann_list.clearSelection()
        for i in self.canvas.selected_indices:
            item = self.ann_list.item(i)
            if item:
                item.setSelected(True)
        self.ann_list.blockSignals(False)

    def _on_annotations_changed(self):
        self.labels_dirty = True
        self._update_title()
        self._refresh_ann_list()

    # ════════════════════════════════════════════════════════════════════════
    # Suggestions
    # ════════════════════════════════════════════════════════════════════════
    def _accept_sel_sug(self):
        self.canvas.accept_selected_suggestion()

    def _accept_all_sug(self):
        self.canvas.accept_all_suggestions()

    def _reject_sel_sug(self):
        self.canvas.reject_selected_suggestion()

    def _reject_all_sug(self):
        self.canvas.reject_all_suggestions()

    # ════════════════════════════════════════════════════════════════════════
    # Drawing mode
    # ════════════════════════════════════════════════════════════════════════
    def _set_draw_mode(self, mode: str):
        canvas_mode = {
            "edit":    Canvas.MODE_EDIT,
            "polygon": Canvas.MODE_POLYGON,
            "sam":     Canvas.MODE_SAM,
            "smart":   Canvas.MODE_SAM,
            "edge":    Canvas.MODE_EDGE,
        }.get(mode, Canvas.MODE_EDIT)
        self._smart_select_active = (mode == "smart")
        self.canvas.set_mode(canvas_mode)
        self.btn_edit.setChecked(mode == "edit")
        self.btn_polygon.setChecked(mode == "polygon")
        self.btn_sam.setChecked(mode == "sam")
        self.btn_smart.setChecked(mode == "smart")
        self.btn_edge.setChecked(mode == "edge")
        if mode == "smart":
            self.status_bar.showMessage(
                "Smart Select: left-click = segment & preview  |  "
                "Enter/Space = accept  |  Esc = reject  |  E to exit", 6000)
        elif mode == "edge":
            self.status_bar.showMessage(
                "Edge Fill: left-click on object = auto-segment  |  "
                "No SAM needed  |  Adjust Edge Fill settings on the right  |  E to exit", 6000)

    # ════════════════════════════════════════════════════════════════════════
    # SAM integration
    # ════════════════════════════════════════════════════════════════════════
    def _populate_sam_type_combo(self):
        """Fill the SAM type combo from whatever libraries are installed."""
        self.sam_type_combo.blockSignals(True)
        self.sam_type_combo.clear()
        self.sam_type_combo.addItem("Auto-detect", userData=("auto", "auto"))

        for lib_name, (registry, _) in SAM_REGISTRIES.items():
            for mtype in sorted(registry.keys()):
                label = f"{mtype}  [{lib_name}]"
                self.sam_type_combo.addItem(label, userData=(lib_name, mtype))

        # If nothing is installed, still show the common types as hints
        if not SAM_REGISTRIES:
            for mtype in ("vit_t (MobileSAM)", "vit_b", "vit_l", "vit_h"):
                self.sam_type_combo.addItem(mtype, userData=("", mtype.split()[0]))

        self.sam_type_combo.blockSignals(False)

    def _load_sam(self):
        if not SAM_AVAILABLE:
            QMessageBox.warning(
                self, "SAM not installed",
                "No SAM library is installed.\n\n"
                "For MobileSAM (recommended – CPU-friendly, ~40 MB checkpoint):\n"
                "    pip install mobile-sam\n\n"
                "For original SAM ViT-B/L/H (needs GPU for real-time speed):\n"
                "    pip install segment-anything\n\n"
                "Checkpoint downloads:\n"
                "  MobileSAM : https://github.com/ChaoningZhang/MobileSAM\n"
                "  SAM       : https://github.com/facebookresearch/segment-anything"
            )
            return

        path, _ = QFileDialog.getOpenFileName(
            self, "Load SAM checkpoint",
            self.folder or os.getcwd(),
            "SAM checkpoint (*.pth *.pt);;All files (*)"
        )
        if not path:
            return

        # ── Step 1: fingerprint the checkpoint ──────────────────────────────
        self.sam_info.setText("SAM: inspecting checkpoint…")
        QApplication.processEvents()

        fp = _fingerprint_checkpoint(path)

        if fp.load_error:
            # Could not even read the file
            QMessageBox.warning(
                self, "SAM – cannot read checkpoint",
                f"Could not load '{os.path.basename(path)}':\n\n{fp.load_error}\n\n"
                "The file may be incomplete or corrupted.\n"
                "Re-download it from the official source."
            )
            self.sam_info.setText("SAM: not loaded")
            return

        # ── Step 2: check if the combo overrides auto-detect ────────────────
        combo_data = self.sam_type_combo.currentData()
        if combo_data and combo_data[0] != "auto":
            # User explicitly chose a (lib, type) — respect it
            fp.needs_lib      = combo_data[0]
            fp.sam_model_type = combo_data[1]
            # Recalculate is_mobile_sam for display
            fp.is_mobile_sam  = (fp.sam_model_type == "vit_t")

        # ── Step 3: nothing to block — _load_sam_model handles all cases ───────
        # (MobileSAM loads via built-in bridge when mobile_sam pkg is absent)

        # ── Step 4: actually build and load the model ────────────────────────
        kind = "MobileSAM vit_t" if fp.is_mobile_sam else f"SAM {fp.sam_model_type}"
        self.sam_info.setText(f"SAM: loading {kind} from {os.path.basename(path)}…")
        QApplication.processEvents()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            model, predictor, label = _load_sam_model(path, fp)
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            self.sam_info.setText("SAM: not loaded")
            QMessageBox.warning(self, "SAM load failed", str(exc))
            return

        QApplication.restoreOverrideCursor()

        # ── Step 5: success ──────────────────────────────────────────────────
        self.sam_model     = model
        self.sam_predictor = predictor
        self.sam_path      = path

        self.sam_info.setText(f"SAM ✔  {os.path.basename(path)}\n{label}")
        if self.canvas.has_image():
            self._update_sam_image()

        self.settings.setValue("sam_path", path)
        self.settings.setValue("sam_label", label)

        self.status_bar.showMessage(
            f"SAM loaded: {label} — press S then click objects to segment", 6000
        )

    # ── DeepLabV3+ helpers ───────────────────────────────────────────────────

    # Cityscapes class names (19 classes)
    CITYSCAPES_CLASSES = [
        "road", "sidewalk", "building", "wall", "fence", "pole",
        "traffic light", "traffic sign", "vegetation", "terrain", "sky",
        "person", "rider", "car", "truck", "bus", "train",
        "motorcycle", "bicycle"
    ]

    # ── DeepLabV3+ model registry ────────────────────────────────────────────
    # Each entry: (display_name, source, loader_kwargs, num_classes, bg_class_ids)
    _DL_MODELS = [
        # ── HuggingFace Cityscapes models — ROAD CLASS INCLUDED ★ ───────────
        ("🏙★ SegFormer-B5 Cityscapes (best quality, ~84MB, needs: pip install transformers)",
         "hf", {"model_id": "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"}, 19, set()),
        ("🏙★ SegFormer-B2 Cityscapes (balanced, ~25MB, needs: pip install transformers)",
         "hf", {"model_id": "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"}, 19, set()),
        ("🏙★ SegFormer-B0 Cityscapes (fastest, ~14MB, needs: pip install transformers)",
         "hf", {"model_id": "nvidia/segformer-b0-finetuned-cityscapes-512-1024"}, 19, set()),
        # ── torchvision COCO models (no road class, detects vehicles/people) ─
        ("🚗 COCO — FCN ResNet-101 (vehicles+people, no road class)",
         "tv_cityscapes", {"variant": "fcn_resnet101"}, 21, {0}),
        ("🚗 COCO — DeepLabV3 ResNet-101",
         "tv_cityscapes", {"variant": "deeplabv3_resnet101"}, 21, {0}),
        ("🚗 COCO — DeepLabV3 MobileNetV3 (fastest)",
         "tv_cityscapes", {"variant": "deeplabv3_mobilenet"}, 21, {0}),
    ]

    # Road-relevant COCO/VOC classes (indices to keep)
    _ROAD_RELEVANT_VOC = {2, 6, 7, 14, 15, 19}  # bicycle,bus,car,motorbike,person,train

    def _get_deeplab_model(self):
        """Load (and cache) the selected DeepLabV3+ model."""
        backbone_idx = self.dl_backbone_combo.currentIndex() \
            if hasattr(self, "dl_backbone_combo") else 0
        backbone_idx = min(backbone_idx, len(self._DL_MODELS) - 1)

        device = "cuda" if (hasattr(torch, "cuda") and torch.cuda.is_available()) else "cpu"
        cache_key = (backbone_idx, device)
        if getattr(self, "_dl_model_cache_key", None) == cache_key:
            return self._dl_model_cache

        display_name, source, kwargs, n_cls, bg_ids = self._DL_MODELS[backbone_idx]
        self.dl_info_lbl.setText(f"Downloading / loading:\n{display_name}\nPlease wait…")
        QApplication.processEvents()

        try:
            if source == "hf":
                model, name = self._load_hf_segformer(kwargs, device)
            elif source == "tv_cityscapes":
                model, name = self._load_tv_cityscapes(kwargs, device)
            else:
                model, name = self._load_torchvision_deeplab(kwargs, device)
        except Exception as exc:
            self.dl_info_lbl.setText(f"Load failed: {exc}")
            QMessageBox.warning(self, "DeepLab load failed", str(exc))
            return None

        model.eval().to(device)
        self._dl_model_cache      = model
        self._dl_model_cache_key  = cache_key
        self._dl_model_device     = device
        self._dl_model_name       = name
        self._dl_model_source     = source
        self._dl_n_classes        = n_cls
        self.dl_info_lbl.setText(f"✔ {name}")
        return model

    def _load_hf_segformer(self, kwargs, device):
        """Load a HuggingFace SegFormer model fine-tuned on Cityscapes.
        Has proper road/sidewalk/car/building etc. classes — just like MATLAB CamVid.
        Requires: pip install transformers
        Models cache to ~/.cache/huggingface/hub (~14-84 MB depending on variant).
        """
        try:
            from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
        except ImportError:
            raise RuntimeError(
                "HuggingFace Transformers not installed.\n\n"
                "Install with:\n    pip install transformers\n\n"
                "Then try again — model downloads automatically (~14-84 MB).")

        model_id = kwargs["model_id"]
        self.dl_info_lbl.setText(f"Downloading {model_id}…\n(first run only, cached after)")
        QApplication.processEvents()

        processor = SegformerImageProcessor.from_pretrained(model_id)
        model     = SegformerForSemanticSegmentation.from_pretrained(model_id)
        model.eval().to(device)

        # Store processor alongside model for use in inference
        self._hf_processor = processor
        name = f"SegFormer Cityscapes ({model_id.split('/')[-1]})"
        return model, name

    def _load_tv_cityscapes(self, kwargs, device):
        """
        Load torchvision FCN / DeepLabV3 models that have been
        properly trained on COCO (which includes road-relevant classes:
        car, bus, train, bicycle, motorbike, person) and produce good
        road-scene segmentation out of the box.
        These are the best freely available pretrained weights for road images.
        """
        try:
            import torchvision
        except ImportError:
            raise RuntimeError("pip install torchvision")

        v = kwargs["variant"]
        if v == "fcn_resnet101":
            w = torchvision.models.segmentation.FCN_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1
            m = torchvision.models.segmentation.fcn_resnet101(weights=w)
            n = "FCN ResNet-101 (COCO — road-scene classes)"
        elif v == "deeplabv3_resnet101":
            w = torchvision.models.segmentation.DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1
            m = torchvision.models.segmentation.deeplabv3_resnet101(weights=w)
            n = "DeepLabV3 ResNet-101 (COCO — road-scene classes)"
        else:
            w = torchvision.models.segmentation.DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1
            m = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(weights=w)
            n = "DeepLabV3 MobileNetV3 (COCO — road-scene classes)"
        return m, n

    def _load_torchvision_deeplab(self, kwargs, device):
        try:
            import torchvision
        except ImportError:
            raise RuntimeError("pip install torchvision")
        v = kwargs["variant"]
        if v == "resnet101":
            w = torchvision.models.segmentation.DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1
            m = torchvision.models.segmentation.deeplabv3_resnet101(weights=w)
            n = "DeepLabV3 ResNet-101 (COCO+VOC)"
        elif v == "resnet50":
            w = torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
            m = torchvision.models.segmentation.deeplabv3_resnet50(weights=w)
            n = "DeepLabV3 ResNet-50 (COCO+VOC)"
        else:
            w = torchvision.models.segmentation.DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1
            m = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(weights=w)
            n = "DeepLabV3 MobileNetV3 (COCO+VOC)"
        return m, n

    def _run_deeplab_on_image(self, image_rgb: np.ndarray):
        """Run DeepLabV3+ on image_rgb, return class-agnostic polygon suggestions."""
        model = self._get_deeplab_model()
        if model is None:
            return []

        import torch.nn.functional as TF2

        conf_thr = self.dl_conf_spin.value()    if hasattr(self, "dl_conf_spin")    else 0.4
        min_area = self.dl_minarea_spin.value() if hasattr(self, "dl_minarea_spin") else 2000
        device   = getattr(self, "_dl_model_device", "cpu")
        source   = getattr(self, "_dl_model_source", "torchvision")
        n_cls    = getattr(self, "_dl_n_classes", 21)

        backbone_idx = self.dl_backbone_combo.currentIndex() \
            if hasattr(self, "dl_backbone_combo") else 0
        backbone_idx = min(backbone_idx, len(self._DL_MODELS) - 1)
        _, _, _, _, bg_ids = self._DL_MODELS[backbone_idx]

        H, W = image_rgb.shape[:2]

        if source == "hf":
            # ── HuggingFace SegFormer inference ──────────────────────────────
            processor = getattr(self, "_hf_processor", None)
            if processor is None:
                return []
            from PIL import Image as PILImage
            pil_img = PILImage.fromarray(image_rgb)
            inputs  = processor(images=pil_img, return_tensors="pt")
            inputs  = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            # Upsample logits to original image size
            logits   = outputs.logits   # (1, C, H/4, W/4)
            upsampled = TF2.interpolate(
                logits, size=(H, W), mode="bilinear", align_corners=False)[0]
            probs_np  = torch.softmax(upsampled, dim=0).cpu().numpy()  # (C, H, W)
            pred_cls  = probs_np.argmax(axis=0)
            pred_conf = probs_np.max(axis=0)
        else:
            # ── torchvision inference ─────────────────────────────────────────
            inp = torch.from_numpy(image_rgb.transpose(2, 0, 1)).float() / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            inp  = ((inp - mean) / std).unsqueeze(0).to(device)

            long_side = max(H, W)
            infer_sz  = min(max(int(round(long_side / 32)) * 32, 512), 1024)
            inp_r = TF2.interpolate(inp, size=(infer_sz, infer_sz),
                                    mode="bilinear", align_corners=False)
            with torch.no_grad():
                out = model(inp_r)
                if isinstance(out, dict):
                    out = out["out"]
            probs    = torch.softmax(out, dim=1)
            probs_up = TF2.interpolate(probs, size=(H, W),
                                       mode="bilinear", align_corners=False)[0]
            probs_np  = probs_up.cpu().numpy()
            pred_cls  = probs_np.argmax(axis=0)
            pred_conf = probs_np.max(axis=0)

        # Include ALL non-background classes (not just road-relevant ones)
        # User assigns the class label themselves afterward
        unique_classes = [c for c in np.unique(pred_cls) if c not in bg_ids]

        suggestions = []
        for cls_id in unique_classes:
            bin_mask = ((pred_cls == cls_id) &
                        (pred_conf >= conf_thr)).astype(np.uint8) * 255
            if bin_mask.sum() == 0:
                continue
            num_labels, labels_im = cv2.connectedComponents(bin_mask)
            for lbl in range(1, num_labels):
                comp_mask = ((labels_im == lbl).astype(np.uint8) * 255)
                if comp_mask.sum() < min_area * 255:
                    continue
                ann = PolygonAnnotation.from_mask(
                    comp_mask,
                    class_id=0,
                    score=float(probs_np[cls_id][labels_im == lbl].mean()),
                    is_suggestion=True)
                if ann is not None:
                    suggestions.append(ann)

        # Sort by area descending (largest/most important segments first)
        suggestions.sort(key=lambda a: -cv2.contourArea(
            np.array(a.points, dtype=np.int32).reshape(-1, 1, 2)))
        return suggestions

    def _deeplab_suggest_current(self):
        """Generate DeepLab suggestions for the currently displayed image."""
        if not self.canvas.has_image():
            QMessageBox.information(self, "No image", "Open an image first.")
            return
        self.dl_suggest_btn.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            suggestions = self._run_deeplab_on_image(self.canvas.image_rgb)
        except Exception as exc:
            QMessageBox.warning(self, "DeepLab error", str(exc))
            suggestions = []
        finally:
            QApplication.restoreOverrideCursor()
            self.dl_suggest_btn.setEnabled(True)

        if not suggestions:
            self.status_bar.showMessage(
                "DeepLab: no segments found — try lowering Min confidence.", 4000)
            return

        self.canvas.set_suggestions(suggestions)
        self.canvas.selected_suggestion_index = 0
        self.status_bar.showMessage(
            f"DeepLab: {len(suggestions)} suggestions — "
            "Enter/double-click to accept each, Esc to reject. "
            "Right-click a suggestion to assign class.", 8000)

    def _deeplab_suggest_all(self):
        """Batch: generate DeepLab suggestions for all images and save as label files."""
        if not self.image_paths:
            QMessageBox.information(self, "No project", "Open an image folder first.")
            return

        reply = QMessageBox.question(
            self, "Batch DeepLab suggestions",
            f"Generate DeepLab polygon suggestions for all {len(self.image_paths)} images?\n\n"
            "Existing annotations are NOT overwritten — suggestions are added alongside.\n"
            "This may take several minutes.",
            QMessageBox.Yes | QMessageBox.No)
        if reply != QMessageBox.Yes:
            return

        # Load model once before the loop
        model = self._get_deeplab_model()
        if model is None:
            return

        total   = len(self.image_paths)
        prog    = QProgressDialog("Generating DeepLab suggestions…", "Cancel", 0, total, self)
        prog.setWindowModality(Qt.WindowModal)
        prog.show()
        saved = 0

        for i, img_path in enumerate(self.image_paths):
            if prog.wasCanceled():
                break
            prog.setValue(i)
            prog.setLabelText(f"Processing {i+1}/{total}: {os.path.basename(img_path)}")
            QApplication.processEvents()

            try:
                img_bgr = cv2.imread(img_path)
                if img_bgr is None:
                    continue
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                suggestions = self._run_deeplab_on_image(img_rgb)
                if not suggestions:
                    continue

                # Append suggestions to the label file (class_id=0 = unassigned)
                lbl_path = os.path.splitext(img_path)[0] + ".txt"
                H, W = img_rgb.shape[:2]
                with open(lbl_path, "a", encoding="utf-8") as fh:
                    for ann in suggestions:
                        fh.write(ann.to_yolo_seg(W, H) + "\n")
                saved += 1
            except Exception:
                continue

        prog.setValue(total)
        self.status_bar.showMessage(
            f"DeepLab batch done: suggestions saved for {saved}/{total} images.", 6000)
        # Reload current image labels
        self._load_image_at_row(self.file_list.currentRow())

    def _update_sam_image(self):
        if self.sam_predictor is None or not self.canvas.has_image():
            return
        try:
            self.sam_predictor.set_image(self.canvas.image_rgb)
        except Exception as exc:
            self.status_bar.showMessage(f"SAM image update failed: {exc}", 4000)

    def _on_edge_click(self, ix: float, iy: float):
        """Handle a click in Edge Fill mode — segment using edge+color flood-fill."""
        if not self.canvas.has_image():
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            mask = edge_annotate_at_point(
                self.canvas.image_rgb, ix, iy,
                canny_lo=self.ef_canny_lo.value()   if hasattr(self, "ef_canny_lo")  else 20,
                canny_hi=self.ef_canny_hi.value()   if hasattr(self, "ef_canny_hi")  else 80,
                dilate_k=self.ef_dilate.value()     if hasattr(self, "ef_dilate")    else 3,
                flood_tol=self.ef_color_tol.value() if hasattr(self, "ef_color_tol") else 15,
            )
        finally:
            QApplication.restoreOverrideCursor()

        if mask is None:
            self.status_bar.showMessage(
                "Edge Fill: no region found — try clicking closer to the object "
                "centre, or increase Color tolerance.", 4000)
            return

        # Convert mask to polygon suggestion (same smoothing as SAM)
        ann = PolygonAnnotation.from_mask(
            mask,
            class_id=self.canvas.current_class_id,
            score=-1.0,
            is_suggestion=True)
        if ann is None:
            self.status_bar.showMessage("Edge Fill: region too small or complex.", 3000)
            return

        self.canvas.set_suggestions([ann])
        self.canvas.selected_suggestion_index = 0
        self.status_bar.showMessage(
            "Edge Fill: preview ready  |  "
            "✔ Enter/Space/double-click to accept  |  "
            "✗ Esc/right-click to reject  |  "
            "Click elsewhere for next object", 0)

    def _on_sam_click(self, ix: float, iy: float, is_fg: bool):
        if self._smart_select_active and is_fg:
            # Smart Select: foreground click → predict → instantly accept
            # Reset points so each click is a fresh independent prediction
            self.sam_points.clear()
            self.sam_labels.clear()
        self.sam_points.append((ix, iy))
        self.sam_labels.append(1 if is_fg else 0)
        self._run_sam_prediction(auto_accept=self._smart_select_active and is_fg)

    def _clear_sam_points(self):
        self.sam_points.clear()
        self.sam_labels.clear()
        # Remove ALL pending SAM suggestions (all suggestions added during this
        # SAM session are tagged with is_suggestion=True and score >= 0)
        self.canvas.suggestions.clear()
        self.canvas.selected_suggestion_index = -1
        self.canvas.suggestions_changed.emit()
        self.canvas.update()
        self.status_bar.showMessage("SAM points cleared.", 2000)

    def _run_sam_prediction(self, auto_accept: bool = False):
        if self.sam_predictor is None or not self.sam_points:
            return
        if not self.canvas.has_image():
            self.status_bar.showMessage("Load an image before using SAM.", 3000)
            return

        # Ensure the predictor has the current image embedded.
        # This is needed on first click after loading a new image, or if
        # set_image() failed silently during image load.
        try:
            # segment_anything stores the image in is_image_set; mobile_sam too.
            image_set = getattr(self.sam_predictor, "is_image_set", None)
            if image_set is False or image_set is None:
                self.sam_predictor.set_image(self.canvas.image_rgb)
        except Exception:
            # Fallback: always call set_image (safe but slower)
            try:
                self.sam_predictor.set_image(self.canvas.image_rgb)
            except Exception as exc:
                self.status_bar.showMessage(f"SAM image setup failed: {exc}", 4000)
                return

        pts = np.array(self.sam_points, dtype=np.float32)
        lbs = np.array(self.sam_labels, dtype=np.int32)

        # For Smart Select, also supply a box prompt centred on the fg click.
        # A generous box (40% of image in each direction) gives SAM enough
        # spatial context to find the full object rather than a local patch.
        box_prompt = None
        if auto_accept and len(self.sam_points) == 1:
            H, W = self.canvas.image_rgb.shape[:2]
            cx, cy = self.sam_points[0]
            cov = getattr(self, "smart_coverage_spin", None)
            pct = (cov.value() / 100.0) if cov else 0.40
            pad_x = W * pct
            pad_y = H * pct
            box_prompt = np.array([
                max(0, cx - pad_x), max(0, cy - pad_y),
                min(W, cx + pad_x), min(H, cy + pad_y),
            ], dtype=np.float32)

        predict_kwargs = dict(point_coords=pts, point_labels=lbs,
                              multimask_output=True)
        if box_prompt is not None:
            predict_kwargs["box"] = box_prompt

        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            masks, scores, _ = self.sam_predictor.predict(**predict_kwargs)
            QApplication.restoreOverrideCursor()
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            # If we still get "image must be set", force a re-embed and retry once
            if "set_image" in str(exc).lower() or "image must be" in str(exc).lower():
                try:
                    self.sam_predictor.set_image(self.canvas.image_rgb)
                    QApplication.setOverrideCursor(Qt.WaitCursor)
                    masks, scores, _ = self.sam_predictor.predict(**predict_kwargs)
                    QApplication.restoreOverrideCursor()
                except Exception as exc2:
                    QApplication.restoreOverrideCursor()
                    self.status_bar.showMessage(f"SAM prediction failed: {exc2}", 4000)
                    return
            else:
                self.status_bar.showMessage(f"SAM prediction failed: {exc}", 4000)
                return
        # Pick the best mask:
        # - Smart Select: prefer the LARGEST mask (most complete object coverage)
        # - SAM mode: prefer highest confidence score (most precise)
        mask_areas = [m.sum() for m in masks]
        if auto_accept:
            # Largest mask = most complete segmentation of the clicked region
            best_idx = int(np.argmax(mask_areas))
        else:
            best_idx = int(np.argmax(scores))
        best_mask  = masks[best_idx].astype(np.uint8)
        best_score = float(scores[best_idx])
        # ── Edge-snap settings ────────────────────────────────────────────
        do_snap    = (hasattr(self, "edge_snap_group") and
                      self.edge_snap_group.isChecked())
        snap_r     = self.snap_radius_spin.value()  if hasattr(self, "snap_radius_spin")  else 8
        canny_lo   = self.canny_low_spin.value()    if hasattr(self, "canny_low_spin")    else 30
        canny_hi   = self.canny_high_spin.value()   if hasattr(self, "canny_high_spin")   else 100
        src_image  = self.canvas.image_rgb if do_snap else None

        ann = PolygonAnnotation.from_mask(
            best_mask,
            class_id=self.canvas.current_class_id,
            score=best_score,
            is_suggestion=True,
            image_rgb=src_image,
            edge_snap=do_snap,
            snap_radius=snap_r,
            canny_low=canny_lo,
            canny_high=canny_hi)
        if ann is None:
            self.status_bar.showMessage("SAM returned an empty mask.", 3000)
            return

        # Replace ALL current suggestions with just this one SAM result.
        self.canvas.set_suggestions([ann])
        self.canvas.selected_suggestion_index = 0

        if auto_accept:
            # Smart Select: show the suggestion highlighted, wait for ONE key/click
            # to confirm (Enter / double-click / Space) or Esc to reject.
            # This prevents accidental saves while still being fast.
            self.sam_points.clear()
            self.sam_labels.clear()
            self.status_bar.showMessage(
                f"Smart Select — score={best_score:.3f}  |  "
                "✔ Enter/Space/double-click to accept  |  "
                "✗ Esc/right-click to reject  |  "
                "Click elsewhere for next object", 0)
        else:
            self.status_bar.showMessage(
                f"SAM score={best_score:.3f} — "
                "double-click or 'Confirm SAM mask' to accept, "
                "right-click to reject.", 5000)

    def _confirm_sam_mask(self):
        """Accept the latest SAM suggestion and clear prompts."""
        idx = self.canvas.selected_suggestion_index
        if idx < 0 and self.canvas.suggestions:
            idx = len(self.canvas.suggestions) - 1
        self.canvas.selected_suggestion_index = idx
        self.canvas.accept_selected_suggestion()
        self.sam_points.clear()
        self.sam_labels.clear()

    # ════════════════════════════════════════════════════════════════════════
    # Supervised model
    # ════════════════════════════════════════════════════════════════════════
    def _load_seg_model(self):
        if not TORCH_AVAILABLE:
            QMessageBox.warning(self, "PyTorch missing",
                                "Install PyTorch: pip install torch torchvision")
            return
        path, _ = QFileDialog.getOpenFileName(self, "Load segmentation checkpoint",
                                              self.folder or os.getcwd(),
                                              "PyTorch checkpoint (*.pt *.pth)")
        if not path:
            return
        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            dev   = "cuda" if torch.cuda.is_available() else "cpu"
            state = torch.load(path, map_location=dev)
            nc    = state.get("num_classes", len(self.canvas.classes) or 2)
            model = TinySegNet(nc)
            model.load_state_dict(state["model"])
            model.eval().to(dev)
            self.seg_model        = model
            self.seg_model_path   = path
            self.seg_num_classes  = nc
            self.model_info_lbl.setText(
                f"Model: {os.path.basename(path)}\n"
                f"Classes: {nc}  Epoch: {state.get('epoch','?')}  "
                f"mIoU: {state.get('miou', 0):.4f}")
            self.settings.setValue("seg_model_path", path)
            QApplication.restoreOverrideCursor()
            self.status_bar.showMessage("Segmentation model loaded.", 4000)
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "Load failed", str(exc))

    def _model_suggest_current(self):
        if self.seg_model is None:
            QMessageBox.information(self, "No model", "Load a segmentation model first.")
            return
        if not self.canvas.has_image():
            return
        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            dev = next(self.seg_model.parameters()).device.type
            pred_map = self.seg_model.predict_mask(self.canvas.image, device=dev)
            QApplication.restoreOverrideCursor()
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "Prediction failed", str(exc))
            return

        suggestions = []
        nc = self.seg_num_classes
        for cls_id in range(1, nc):   # skip background (0)
            mask = (pred_map == cls_id).astype(np.uint8)
            if mask.sum() == 0:
                continue
            # find connected components
            n_labels, label_map = cv2.connectedComponents(mask)
            for comp_id in range(1, n_labels):
                comp_mask = (label_map == comp_id).astype(np.uint8)
                ann = PolygonAnnotation.from_mask(comp_mask,
                                                   class_id=cls_id,
                                                   score=0.5,
                                                   is_suggestion=True)
                if ann and ann.area() > 100:
                    suggestions.append(ann)
        self.canvas.set_suggestions(suggestions)
        self.status_bar.showMessage(
            f"Generated {len(suggestions)} suggestions from model.", 4000)

    # ════════════════════════════════════════════════════════════════════════
    # Training
    # ════════════════════════════════════════════════════════════════════════
    def _choose_save_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Choose save directory",
                                             self.folder or os.getcwd())
        if d:
            self._train_save_dir_path = d
            self.train_savedir.setText(os.path.basename(d))

    def _start_training(self):
        if not TORCH_AVAILABLE:
            QMessageBox.warning(self, "PyTorch missing",
                                "Install PyTorch: pip install torch torchvision")
            return
        if not self.folder or not self.image_files:
            QMessageBox.information(self, "No data", "Open a project folder first.")
            return
        if not self.canvas.classes:
            QMessageBox.information(self, "No classes", "Define at least one class.")
            return
        save_dir = self._train_save_dir_path or self.folder
        cfg = {
            "folder":      self.folder,
            "image_files": self.image_files,
            "classes":     list(self.canvas.classes),
            "num_classes": len(self.canvas.classes) + 1,  # +1 for background
            "epochs":      self.train_epochs.value(),
            "batch":       self.train_batch.value(),
            "lr":          self.train_lr.value(),
            "img_size":    self.train_imgsize.value(),
            "save_dir":    save_dir,
            "checkpoint":  self.seg_model_path,
        }
        dialog = TrainingDialog(cfg, self)
        dialog.exec_()
        # auto-load best model after training
        last_ckpt = cfg.get("last_checkpoint", "")
        if last_ckpt and os.path.exists(last_ckpt):
            if QMessageBox.question(self, "Load trained model",
                                    f"Load trained model from {last_ckpt}?") == QMessageBox.Yes:
                self.seg_model_path = last_ckpt
                self._load_seg_model()

    # ════════════════════════════════════════════════════════════════════════
    # Settings persistence
    # ════════════════════════════════════════════════════════════════════════
    def _restore_settings(self):
        geo = self.settings.value("geometry")
        if geo:
            self.restoreGeometry(geo)
        self.folder         = self.settings.value("last_folder",      "", type=str)
        self.sam_path       = self.settings.value("sam_path",         "", type=str)
        self.seg_model_path = self.settings.value("seg_model_path",   "", type=str)
        saved_lib  = self.settings.value("sam_lib",        "", type=str)
        saved_type = self.settings.value("sam_model_type", "", type=str)
        # restore SAM combo selection
        if saved_lib and saved_type:
            for i in range(self.sam_type_combo.count()):
                d = self.sam_type_combo.itemData(i)
                if d and d == (saved_lib, saved_type):
                    self.sam_type_combo.setCurrentIndex(i)
                    break

    def _save_settings(self):
        self.settings.setValue("geometry",     self.saveGeometry())
        self.settings.setValue("last_folder",  self.folder)
        self.settings.setValue("sam_path",     self.sam_path)
        self.settings.setValue("seg_model_path", self.seg_model_path)
        self.settings.sync()

    # ════════════════════════════════════════════════════════════════════════
    # Help / About
    # ════════════════════════════════════════════════════════════════════════
    def _show_about(self):
        QMessageBox.about(self, f"About {APP_NAME}",
            f"{APP_NAME}  v{APP_VERSION}\n"
            f"Developed by {APP_DEVELOPER}\n\n"
            "Semantic segmentation annotation tool for road scenes\n\n"
            "• SAM / Smart Select click-to-segment\n"
            "• Edge Fill annotator\n"
            "• Manual polygon drawing\n"
            "• Pretrained SegFormer / DeepLab suggestions\n"
            "• YOLO-seg label format\n"
            "• Supervised model training")

    def _show_help(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Help")
        dialog.resize(600, 500)
        v = QVBoxLayout(dialog)
        te = QTextEdit(); te.setReadOnly(True)
        te.setPlainText(
            "WORKFLOW\n"
            "────────\n"
            "1. File → Open Folder containing images.\n"
            "2. Add classes in the Project tab.\n"
            "3. Load a SAM checkpoint in the Annotate tab.\n"
            "4. Press S to enter SAM mode.\n"
            "   • Left-click = foreground prompt\n"
            "   • Right-click = background prompt (exclusion)\n"
            "   • Click 'Confirm SAM mask' or double-click the suggestion to accept.\n"
            "5. Press P for manual polygon mode.\n"
            "   • Click vertices, double-click / Enter to close.\n"
            "   • Right-click / Esc to cancel.\n"
            "6. Press E (or Escape) for Edit mode.\n"
            "   • Drag polygon body to move.\n"
            "   • Drag vertex handle to reshape.\n"
            "7. Ctrl+S to save labels (YOLO-seg format).\n"
            "\n"
            "TRAINING\n"
            "─────────\n"
            "1. After annotating ≥ 20–50 images open the Train tab.\n"
            "2. Set epochs, batch size, learning rate.\n"
            "3. Click 'Start training'.\n"
            "4. When done, load the best checkpoint and click\n"
            "   'Generate model suggestions' on new images.\n"
            "5. Review suggestions (accept / reject), then\n"
            "   re-train with the growing dataset.\n"
            "\n"
            "SHORTCUTS\n"
            "──────────\n"
            "S – SAM mode        P – Polygon mode    E – Edit mode\n"
            "A / D – Prev / Next image               W – Next unlabeled\n"
            "F – Fit image       R – Toggle reviewed\n"
            "Ctrl+Z / Y – Undo / Redo\n"
            "Delete – Delete selected annotation\n"
            "Return – Accept selected suggestion\n"
            "Ctrl+Return – Accept all suggestions\n"
            "Backspace – Reject selected suggestion\n"
            "Space+drag – Pan     Scroll – Zoom\n"
        )
        v.addWidget(te)
        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(dialog.reject)
        v.addWidget(bb)
        dialog.exec_()

    # ════════════════════════════════════════════════════════════════════════
    # Qt events
    # ════════════════════════════════════════════════════════════════════════
    def closeEvent(self, event):
        if self.labels_dirty:
            ans = QMessageBox.question(self, "Unsaved labels",
                                       "You have unsaved labels. Save before closing?",
                                       QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
            if ans == QMessageBox.Cancel:
                event.ignore(); return
            if ans == QMessageBox.Yes:
                self._save_labels()
        self._save_settings()
        super().closeEvent(event)


# ════════════════════════════════════════════════════════════════════════════
# Edge-detection annotator (no SAM required)
# ════════════════════════════════════════════════════════════════════════════
def edge_annotate_at_point(image_rgb: np.ndarray, ix: float, iy: float,
                           canny_lo: int = 20, canny_hi: int = 80,
                           dilate_k: int = 3, flood_tol: int = 15,
                           min_area: int = 500) -> Optional[np.ndarray]:
    """
    Edge + color flood-fill segmentation at a clicked point. No SAM needed.

    Pipeline:
      1. Bilateral filter — smooth while preserving edges
      2. Canny on the smoothed grayscale → barrier edges
      3. Dilate edges slightly to close small gaps
      4. Build a reachability map: pixels reachable if NOT a barrier AND
         color is within flood_tol of the seed pixel (LAB color space)
      5. Flood-fill from click point → binary region mask
      6. Morphological close+open to clean the result

    Returns a uint8 binary mask (H×W, values 0/255) or None.
    """
    H, W = image_rgb.shape[:2]
    cx, cy = int(round(ix)), int(round(iy))
    if not (0 <= cx < W and 0 <= cy < H):
        return None

    # 1. Smooth
    gray   = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    smooth = cv2.bilateralFilter(gray, 9, 75, 75)

    # 2. Multi-scale Canny — combine fine and coarse edges for better coverage
    edges_fine   = cv2.Canny(smooth, canny_lo,     canny_hi)
    edges_coarse = cv2.Canny(smooth, canny_lo//2,  canny_hi//2)
    edges = cv2.bitwise_or(edges_fine, edges_coarse)

    # 3. Dilate edges → closed barrier walls (plug gaps that flood fill leaks through)
    dk      = dilate_k * 2 + 1
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dk, dk))
    barriers = cv2.dilate(edges, kernel, iterations=2)  # 2 iterations = thicker walls

    # 4. Color similarity map in LAB space
    lab       = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    seed_lab  = lab[cy, cx]
    color_diff = np.linalg.norm(lab - seed_lab, axis=2)
    color_ok   = (color_diff < flood_tol).astype(np.uint8) * 255

    # Reachable = not a barrier AND color is similar
    reachable = cv2.bitwise_and(255 - barriers, color_ok)

    # 5. Flood-fill from seed
    canvas    = np.zeros((H + 2, W + 2), dtype=np.uint8)
    canvas[1:H+1, 1:W+1] = reachable
    fill_mask = np.zeros((H + 4, W + 4), dtype=np.uint8)
    cv2.floodFill(canvas, fill_mask, (cx + 1, cy + 1), 128,
                  loDiff=0, upDiff=0,
                  flags=cv2.FLOODFILL_MASK_ONLY | (128 << 8))
    region = (fill_mask[1:H+1, 1:W+1] == 128).astype(np.uint8) * 255

    if region.sum() < min_area * 255:
        return None

    # 6. Morphological clean-up
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    region  = cv2.morphologyEx(region, cv2.MORPH_CLOSE, k_close)
    region  = cv2.morphologyEx(region, cv2.MORPH_OPEN,  k_open)
    return region if region.sum() >= min_area * 255 else None


# ════════════════════════════════════════════════════════════════════════════
# Crash handler
# ════════════════════════════════════════════════════════════════════════════
def _install_crash_handler():
    def hook(et, ev, tb):
        log = os.path.join(os.getcwd(), CRASH_LOG)
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(f"\n=== {datetime.now().isoformat()} ===\n")
            traceback.print_exception(et, ev, tb, file=fh)
        try:
            QMessageBox.critical(None, "Crash", f"Unexpected error. Log: {log}")
        except Exception:
            pass
        sys.__excepthook__(et, ev, tb)
    sys.excepthook = hook


# ════════════════════════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    _install_crash_handler()
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setApplicationName(APP_NAME)
    app.setApplicationDisplayName(APP_NAME)
    win = SegAnnotator()
    win.show()
    sys.exit(app.exec_())
