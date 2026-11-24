"""Slot Attention (Locatello et al., "Object-Centric Learning with Slot Attention", NeurIPS 2020), in PyTorch.

The architecture of the paper's object-discovery model (google-research/slot_attention), on CLEVR's full frame:
  encoder   4 convolutions 5x5, 64 channels, no stride, ReLU; a learned position embedding; LayerNorm + MLP
  slots     K slots of 64 numbers, drawn from one Gaussian with a learned mean and sigma; 3 iterations of
            attention with the softmax over the slots (the slots compete for the pixels), a weighted mean of
            the pixels' values, a GRU and a residual MLP
  decoder   each slot alone: broadcast to an 8 x 12 grid, a position embedding, 4 transposed convolutions of
            stride 2 (-> 128 x 192), 2 more; 4 channels out: RGB and an alpha mask
  image     the alpha masks are normalised by a softmax over the slots and mix the K RGB images

In clevr_qa the model is trained by the knowledge base (train_slots.py): the question rules and the
reconstruction axiom `forall (img, rec): forall p: Eq(rec[p], img[p])`, together, from scratch. Each slot
gives scene[o]: its vector (the appearance) and, from its alpha mask, where it is (soft centroid) and how much
of the image it covers. A copy of ../slots.py (e2_slots_pretrain.py's model), so that clevr_qa stays
self-contained.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

RES = (128, 192)            # CLEVR's 320 x 480 frame, resized keeping the aspect ratio
DECODER_INIT = (8, 12)      # x 2^4 = RES
K = 11                      # up to 10 objects, plus the background
D = 64                      # slot and feature size


def build_grid(resolution):
    """[1, H, W, 4]: (y, x, 1 - y, 1 - x), each in [0, 1]."""
    ys, xs = (torch.linspace(0.0, 1.0, r) for r in resolution)
    grid = torch.stack(torch.meshgrid(ys, xs, indexing="ij"), dim=-1)
    return torch.cat([grid, 1.0 - grid], dim=-1)[None]


class SoftPositionEmbed(nn.Module):
    """Adds a learned projection of the coordinate grid to a [B, H, W, C] map."""

    def __init__(self, size, resolution):
        super().__init__()
        self.dense = nn.Linear(4, size)
        self.register_buffer("grid", build_grid(resolution), persistent=False)

    def forward(self, x):
        return x + self.dense(self.grid)


class SlotAttention(nn.Module):
    def __init__(self, num_slots=K, size=D, iters=3, mlp_hidden=128, eps=1e-8):
        super().__init__()
        self.num_slots, self.size, self.iters, self.eps = num_slots, size, iters, eps
        self.norm_inputs, self.norm_slots, self.norm_mlp = nn.LayerNorm(size), nn.LayerNorm(size), nn.LayerNorm(size)
        self.mu = nn.Parameter(torch.empty(1, 1, size))
        self.log_sigma = nn.Parameter(torch.empty(1, 1, size))
        nn.init.xavier_uniform_(self.mu)
        nn.init.xavier_uniform_(self.log_sigma)
        self.q = nn.Linear(size, size, bias=False)
        self.k = nn.Linear(size, size, bias=False)
        self.v = nn.Linear(size, size, bias=False)
        self.gru = nn.GRUCell(size, size)
        self.mlp = nn.Sequential(nn.Linear(size, mlp_hidden), nn.ReLU(), nn.Linear(mlp_hidden, size))

    def forward(self, inputs, num_slots=None):
        """inputs [B, N, D] -> slots [B, K, D], attention [B, N, K] (each pixel's split over the slots)."""
        B, N, _ = inputs.shape
        n = num_slots or self.num_slots
        inputs = self.norm_inputs(inputs)
        k, v = self.k(inputs), self.v(inputs)
        slots = self.mu + self.log_sigma.exp() * torch.randn(B, n, self.size, device=inputs.device)
        for _ in range(self.iters):
            prev = slots
            q = self.q(self.norm_slots(slots)) * self.size ** -0.5
            attn = torch.softmax(torch.einsum("bnd,bkd->bnk", k, q), dim=-1) + self.eps   # softmax over the slots
            weights = attn / attn.sum(dim=1, keepdim=True)                                # mean over the pixels
            updates = torch.einsum("bnk,bnd->bkd", weights, v)
            slots = self.gru(updates.reshape(-1, self.size), prev.reshape(-1, self.size)).reshape(B, n, self.size)
            slots = slots + self.mlp(self.norm_mlp(slots))
        return slots, attn


def _tconv(cin, cout, k, stride):
    # "same" padding: stride 2 doubles the size, stride 1 keeps it
    return nn.ConvTranspose2d(cin, cout, k, stride=stride, padding=k // 2, output_padding=stride - 1)


class SlotAutoEncoder(nn.Module):
    def __init__(self, num_slots=K, size=D, iters=3):
        super().__init__()
        conv = lambda cin: nn.Conv2d(cin, size, 5, padding=2)
        self.encoder = nn.Sequential(conv(3), nn.ReLU(), conv(size), nn.ReLU(), conv(size), nn.ReLU(), conv(size), nn.ReLU())
        self.encoder_pos = SoftPositionEmbed(size, RES)
        self.encoder_norm = nn.LayerNorm(size)
        self.encoder_mlp = nn.Sequential(nn.Linear(size, size), nn.ReLU(), nn.Linear(size, size))
        self.slot_attention = SlotAttention(num_slots, size, iters)
        self.decoder_pos = SoftPositionEmbed(size, DECODER_INIT)
        self.decoder = nn.Sequential(
            _tconv(size, size, 5, 2), nn.ReLU(), _tconv(size, size, 5, 2), nn.ReLU(),
            _tconv(size, size, 5, 2), nn.ReLU(), _tconv(size, size, 5, 2), nn.ReLU(),
            _tconv(size, size, 5, 1), nn.ReLU(), _tconv(size, 4, 3, 1))

    @staticmethod
    def to_input(images):
        """uint8 [B, H, W, 3] -> float [B, 3, H, W] in [-1, 1], as the paper."""
        return images.permute(0, 3, 1, 2).float().div(127.5).sub(1.0)

    def encode(self, x):
        """x [B, 3, H, W] in [-1, 1] -> slots [B, K, D], attention [B, H*W, K]."""
        f = self.encoder(x).permute(0, 2, 3, 1)                         # [B, H, W, D]
        f = self.encoder_pos(f).flatten(1, 2)                           # [B, N, D]
        f = self.encoder_mlp(self.encoder_norm(f))
        return self.slot_attention(f)

    def decode(self, slots):
        """slots [B, K, D] -> per-slot RGB [B, K, 3, H, W], alpha masks [B, K, 1, H, W] (softmax over K), image."""
        B, n, d = slots.shape
        x = slots.reshape(B * n, 1, 1, d).expand(-1, *DECODER_INIT, -1)
        x = self.decoder_pos(x).permute(0, 3, 1, 2)
        x = self.decoder(x).reshape(B, n, 4, *RES)
        rgb, masks = x[:, :, :3], torch.softmax(x[:, :, 3:], dim=1)
        return rgb, masks, (rgb * masks).sum(dim=1)

    def forward(self, x):
        slots, attn = self.encode(x)
        rgb, masks, recon = self.decode(slots)
        return dict(slots=slots, attn=attn, rgb=rgb, masks=masks, recon=recon)


def mask_centroids(masks):
    """masks [B, K, 1, H, W] -> [B, K, 2]: the soft centroid (x, y) of each slot's mask, in [0, 1]; and its mass."""
    m = masks[:, :, 0]
    mass = m.sum(dim=(-2, -1)).clamp_min(1e-8)
    grid = build_grid(m.shape[-2:]).to(m)[0]                            # [H, W, 4]: (y, x, ...)
    y = (m * grid[..., 0]).sum(dim=(-2, -1)) / mass
    x = (m * grid[..., 1]).sum(dim=(-2, -1)) / mass
    return torch.stack([x, y], dim=-1), mass / (m.shape[-1] * m.shape[-2])
