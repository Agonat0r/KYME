"""Foundation-model adapters for validated biosignal checkpoints."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

from classifiers import _import_torch


def _candidate_pydeps_dirs() -> List[Path]:
    base_dir = Path(__file__).resolve().parents[1] / "sessions"
    return [base_dir / "pydeps_rt", base_dir / "pydeps"]


def _ensure_local_pydeps_on_path() -> None:
    for local_pydeps in _candidate_pydeps_dirs():
        if not local_pydeps.exists():
            continue
        local_path = str(local_pydeps)
        if local_path not in sys.path:
            sys.path.insert(0, local_path)


def _import_safetensors():
    _ensure_local_pydeps_on_path()
    from safetensors import safe_open

    return safe_open


class _STPatchEmbed:
    def __init__(self, nn, patch_dim: int, embed_dim: int) -> None:
        self.proj = nn.Linear(patch_dim, embed_dim)

    def __call__(self, x):
        return self.proj(x)


class _STMlp:
    def __init__(self, nn, embed_dim: int, hidden_dim: int) -> None:
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)

    def __call__(self, x):
        return self.fc2(self.act(self.fc1(x)))


class _STAttention:
    def __init__(self, nn, embed_dim: int, num_heads: int) -> None:
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def __call__(self, x):
        torch, _, _ = _import_torch()
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj(out)


class _STBlock:
    def __init__(self, nn, embed_dim: int, num_heads: int, mlp_hidden_dim: int) -> None:
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = _STAttention(nn, embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = _STMlp(nn, embed_dim, mlp_hidden_dim)

    def __call__(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class STEEGFormerSmallEncoder:
    """Minimal forward wrapper reconstructed from the ST-EEGFormer small checkpoint."""

    def __init__(self, *, expected_channels: int = 8, patch_size: int = 2, embed_dim: int = 512, depth: int = 8, num_heads: int = 8) -> None:
        torch, nn, _ = _import_torch()
        self.torch = torch
        self.nn = nn
        self.expected_channels = int(expected_channels)
        self.patch_size = int(patch_size)
        self.embed_dim = int(embed_dim)
        self.depth = int(depth)
        self.num_heads = int(num_heads)
        self.patch_embed = _STPatchEmbed(nn, self.expected_channels * self.patch_size, self.embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.blocks = [_STBlock(nn, self.embed_dim, self.num_heads, self.embed_dim * 4) for _ in range(self.depth)]
        self.norm = nn.LayerNorm(self.embed_dim)

    def load_checkpoint(self, weights_path: Path) -> None:
        safe_open = _import_safetensors()
        with safe_open(str(weights_path), framework="pt", device="cpu") as handle:
            self.patch_embed.proj.weight.data.copy_(handle.get_tensor("patch_embed.proj.weight"))
            self.patch_embed.proj.bias.data.copy_(handle.get_tensor("patch_embed.proj.bias"))
            self.cls_token.data.copy_(handle.get_tensor("cls_token"))
            self.norm.weight.data.copy_(handle.get_tensor("norm.weight"))
            self.norm.bias.data.copy_(handle.get_tensor("norm.bias"))
            for idx, block in enumerate(self.blocks):
                prefix = f"blocks.{idx}"
                block.attn.qkv.weight.data.copy_(handle.get_tensor(f"{prefix}.attn.qkv.weight"))
                block.attn.qkv.bias.data.copy_(handle.get_tensor(f"{prefix}.attn.qkv.bias"))
                block.attn.proj.weight.data.copy_(handle.get_tensor(f"{prefix}.attn.proj.weight"))
                block.attn.proj.bias.data.copy_(handle.get_tensor(f"{prefix}.attn.proj.bias"))
                block.norm1.weight.data.copy_(handle.get_tensor(f"{prefix}.norm1.weight"))
                block.norm1.bias.data.copy_(handle.get_tensor(f"{prefix}.norm1.bias"))
                block.norm2.weight.data.copy_(handle.get_tensor(f"{prefix}.norm2.weight"))
                block.norm2.bias.data.copy_(handle.get_tensor(f"{prefix}.norm2.bias"))
                block.mlp.fc1.weight.data.copy_(handle.get_tensor(f"{prefix}.mlp.fc1.weight"))
                block.mlp.fc1.bias.data.copy_(handle.get_tensor(f"{prefix}.mlp.fc1.bias"))
                block.mlp.fc2.weight.data.copy_(handle.get_tensor(f"{prefix}.mlp.fc2.weight"))
                block.mlp.fc2.bias.data.copy_(handle.get_tensor(f"{prefix}.mlp.fc2.bias"))

    def eval(self) -> "STEEGFormerSmallEncoder":
        return self

    def __call__(self, x):
        torch = self.torch
        if x.ndim != 3:
            raise ValueError("Expected input shaped [batch, channels, samples]")
        batch, channels, samples = x.shape
        if channels < self.expected_channels:
            pad = torch.zeros((batch, self.expected_channels - channels, samples), dtype=x.dtype, device=x.device)
            x = torch.cat([x, pad], dim=1)
        elif channels > self.expected_channels:
            x = x[:, : self.expected_channels, :]

        remainder = samples % self.patch_size
        if remainder:
            pad = self.patch_size - remainder
            tail = torch.zeros((batch, self.expected_channels, pad), dtype=x.dtype, device=x.device)
            x = torch.cat([x, tail], dim=2)

        x = x.transpose(1, 2).reshape(batch, -1, self.expected_channels * self.patch_size)
        x = self.patch_embed(x)
        cls = self.cls_token.expand(batch, -1, -1)
        x = torch.cat([cls, x], dim=1)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x[:, 0]


class _NeuroRVQMlp:
    def __init__(self, nn, embed_dim: int, hidden_dim: int) -> None:
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)

    def __call__(self, x):
        return self.fc2(self.act(self.fc1(x)))


class _NeuroRVQAttention:
    def __init__(self, nn, embed_dim: int, num_heads: int, head_dim: int) -> None:
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.q_norm = nn.LayerNorm(head_dim)
        self.k_norm = nn.LayerNorm(head_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        torch, _, _ = _import_torch()
        self.q_bias = nn.Parameter(torch.zeros(embed_dim))
        self.v_bias = nn.Parameter(torch.zeros(embed_dim))

    def __call__(self, x):
        torch, _, F = _import_torch()
        batch, tokens, channels = x.shape
        zeros = torch.zeros_like(self.q_bias)
        qkv = F.linear(x, self.qkv.weight, torch.cat([self.q_bias, zeros, self.v_bias], dim=0))
        qkv = qkv.reshape(batch, tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q = self.q_norm(qkv[0])
        k = self.k_norm(qkv[1])
        v = qkv[2]
        attn = torch.matmul(q * self.scale, k.transpose(-2, -1))
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj(out)


class _NeuroRVQBlock:
    def __init__(self, nn, embed_dim: int, num_heads: int, head_dim: int, mlp_hidden_dim: int) -> None:
        torch, _, _ = _import_torch()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = _NeuroRVQAttention(nn, embed_dim, num_heads, head_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = _NeuroRVQMlp(nn, embed_dim, mlp_hidden_dim)
        self.gamma_1 = nn.Parameter(torch.ones(embed_dim))
        self.gamma_2 = nn.Parameter(torch.ones(embed_dim))

    def __call__(self, x):
        x = x + self.gamma_1.view(1, 1, -1) * self.attn(self.norm1(x))
        x = x + self.gamma_2.view(1, 1, -1) * self.mlp(self.norm2(x))
        return x


class _NeuroRVQPatchStem:
    """Best-effort patch stem for the released NeuroRVQ EMG encoder weights."""

    BRANCH_TOKEN_WIDTHS = (7, 6, 6, 6)

    def __init__(self, nn, token_count: int) -> None:
        self.token_count = int(token_count)
        branch_kernel_pairs = ((51, 25), (17, 9), (8, 4), (5, 3))
        self.conv1 = []
        self.conv2 = []
        self.norm1 = []
        self.norm2 = []
        for kernel_1, kernel_2 in branch_kernel_pairs:
            self.conv1.append(nn.Conv2d(1, 8, kernel_size=(1, kernel_1), bias=True))
            self.norm1.append(nn.BatchNorm2d(8, affine=True, track_running_stats=False))
            self.conv2.append(nn.Conv2d(8, 8, kernel_size=(1, kernel_2), bias=True))
            self.norm2.append(nn.BatchNorm2d(8, affine=True, track_running_stats=False))

    def load_checkpoint(self, handle) -> None:
        for index in range(4):
            branch = index + 1
            self.conv1[index].weight.data.copy_(handle.get_tensor(f"patch_embed.conv1_{branch}.weight"))
            self.conv1[index].bias.data.copy_(handle.get_tensor(f"patch_embed.conv1_{branch}.bias"))
            self.conv2[index].weight.data.copy_(handle.get_tensor(f"patch_embed.conv2_{branch}.weight"))
            self.conv2[index].bias.data.copy_(handle.get_tensor(f"patch_embed.conv2_{branch}.bias"))
            self.norm1[index].weight.data.copy_(handle.get_tensor(f"patch_embed.norm1_{branch}.weight"))
            self.norm1[index].bias.data.copy_(handle.get_tensor(f"patch_embed.norm1_{branch}.bias"))
            self.norm2[index].weight.data.copy_(handle.get_tensor(f"patch_embed.norm2_{branch}.weight"))
            self.norm2[index].bias.data.copy_(handle.get_tensor(f"patch_embed.norm2_{branch}.bias"))

    def __call__(self, x):
        torch, _, F = _import_torch()
        tokens = []
        for conv1, norm1, conv2, norm2, branch_width in zip(
            self.conv1,
            self.norm1,
            self.conv2,
            self.norm2,
            self.BRANCH_TOKEN_WIDTHS,
        ):
            out = F.gelu(norm1(conv1(x)))
            out = F.gelu(norm2(conv2(out)))
            # Collapse the sensor height and distribute each branch across the 16 output tokens.
            out = F.adaptive_avg_pool2d(out, (1, branch_width * self.token_count))
            out = out.squeeze(2).reshape(out.shape[0], 8, branch_width, self.token_count)
            out = out.permute(0, 3, 1, 2)
            tokens.append(out)
        merged = torch.cat(tokens, dim=-1)
        return merged.reshape(merged.shape[0], self.token_count, -1)


class NeuroRVQEMGFoundationEncoder:
    """Approximate forward wrapper for the released NeuroRVQ EMG foundation weights."""

    def __init__(
        self,
        *,
        expected_channels: int = 8,
        token_count: int = 16,
        embed_dim: int = 200,
        depth: int = 12,
        num_heads: int = 10,
        head_dim: int = 20,
    ) -> None:
        torch, nn, _ = _import_torch()
        self.torch = torch
        self.nn = nn
        self.expected_channels = int(expected_channels)
        self.token_count = int(token_count)
        self.embed_dim = int(embed_dim)
        self.depth = int(depth)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.patch_embed = _NeuroRVQPatchStem(nn, token_count=self.token_count)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.token_count + 1, self.embed_dim))
        self.time_embed = nn.Parameter(torch.zeros(256, self.embed_dim))
        self.blocks = [
            _NeuroRVQBlock(
                nn,
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                mlp_hidden_dim=self.embed_dim * 4,
            )
            for _ in range(self.depth)
        ]
        self.norm_pre = nn.LayerNorm(self.embed_dim)

    def load_checkpoint(self, weights_path: Path) -> None:
        safe_open = _import_safetensors()
        with safe_open(str(weights_path), framework="pt", device="cpu") as handle:
            self.patch_embed.load_checkpoint(handle)
            self.cls_token.data.copy_(handle.get_tensor("cls_token"))
            self.pos_embed.data.copy_(handle.get_tensor("pos_embed").unsqueeze(0))
            self.time_embed.data.copy_(handle.get_tensor("time_embed"))
            self.norm_pre.weight.data.copy_(handle.get_tensor("norm_pre.weight"))
            self.norm_pre.bias.data.copy_(handle.get_tensor("norm_pre.bias"))
            for idx, block in enumerate(self.blocks):
                prefix = f"blocks.{idx}"
                block.norm1.weight.data.copy_(handle.get_tensor(f"{prefix}.norm1.weight"))
                block.norm1.bias.data.copy_(handle.get_tensor(f"{prefix}.norm1.bias"))
                block.norm2.weight.data.copy_(handle.get_tensor(f"{prefix}.norm2.weight"))
                block.norm2.bias.data.copy_(handle.get_tensor(f"{prefix}.norm2.bias"))
                block.gamma_1.data.copy_(handle.get_tensor(f"{prefix}.gamma_1"))
                block.gamma_2.data.copy_(handle.get_tensor(f"{prefix}.gamma_2"))
                block.attn.qkv.weight.data.copy_(handle.get_tensor(f"{prefix}.attn.qkv.weight"))
                block.attn.q_bias.data.copy_(handle.get_tensor(f"{prefix}.attn.q_bias"))
                block.attn.v_bias.data.copy_(handle.get_tensor(f"{prefix}.attn.v_bias"))
                block.attn.q_norm.weight.data.copy_(handle.get_tensor(f"{prefix}.attn.q_norm.weight"))
                block.attn.q_norm.bias.data.copy_(handle.get_tensor(f"{prefix}.attn.q_norm.bias"))
                block.attn.k_norm.weight.data.copy_(handle.get_tensor(f"{prefix}.attn.k_norm.weight"))
                block.attn.k_norm.bias.data.copy_(handle.get_tensor(f"{prefix}.attn.k_norm.bias"))
                block.attn.proj.weight.data.copy_(handle.get_tensor(f"{prefix}.attn.proj.weight"))
                block.attn.proj.bias.data.copy_(handle.get_tensor(f"{prefix}.attn.proj.bias"))
                block.mlp.fc1.weight.data.copy_(handle.get_tensor(f"{prefix}.mlp.fc1.weight"))
                block.mlp.fc1.bias.data.copy_(handle.get_tensor(f"{prefix}.mlp.fc1.bias"))
                block.mlp.fc2.weight.data.copy_(handle.get_tensor(f"{prefix}.mlp.fc2.weight"))
                block.mlp.fc2.bias.data.copy_(handle.get_tensor(f"{prefix}.mlp.fc2.bias"))

    def eval(self) -> "NeuroRVQEMGFoundationEncoder":
        return self

    def __call__(self, x):
        torch = self.torch
        if x.ndim != 3:
            raise ValueError("Expected input shaped [batch, channels, samples]")
        batch, channels, samples = x.shape
        if channels < self.expected_channels:
            pad = torch.zeros((batch, self.expected_channels - channels, samples), dtype=x.dtype, device=x.device)
            x = torch.cat([x, pad], dim=1)
        elif channels > self.expected_channels:
            x = x[:, : self.expected_channels, :]

        stem_in = x.unsqueeze(1)
        tokens = self.patch_embed(stem_in)
        time_idx = torch.linspace(0, self.time_embed.shape[0] - 1, steps=tokens.shape[1], device=x.device).round().long()
        tokens = tokens + self.time_embed[time_idx].unsqueeze(0)
        tokens = tokens + self.pos_embed[:, 1 : 1 + tokens.shape[1]]
        cls = (self.cls_token + self.pos_embed[:, :1]).expand(batch, -1, -1)
        encoded = torch.cat([cls, tokens], dim=1)
        for block in self.blocks:
            encoded = block(encoded)
        encoded = self.norm_pre(encoded)
        return encoded[:, 0]


def load_foundation_adapter(record: Dict[str, object]):
    adapter = str(record.get("adapter") or "").strip().lower()
    weights_path = Path(record["weights_path"])
    if adapter == "st_eegformer_small_encoder":
        model = STEEGFormerSmallEncoder()
        model.load_checkpoint(weights_path)
        model.eval()
        return model
    if adapter == "neurorvq_emg_foundation_encoder":
        model = NeuroRVQEMGFoundationEncoder()
        model.load_checkpoint(weights_path)
        model.eval()
        return model
    raise RuntimeError(f"No runtime adapter is implemented for {adapter or 'unknown adapter'}")
