"""Track J: JEPA over the observation log.

Context encoder sees the causal log up to the query time. The EMA target
encoder additionally sees the object's next sighting -- on real data that
delayed re-observation is the only supervision available, so the teacher is
given exactly that and nothing more (no ground truth anywhere in stage 1).

Anti-collapse follows the recipe that worked in the stock Graph-JEPA:
EMA target + stop-grad + asymmetric predictor, VICReg variance/covariance on
the predicted latent, and an imputation anchor that keeps the context latent
decodable. See docs/SURVEY_20260807.md section 4.
"""
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import MAX_LOC, EventTransformer, TwoHeadEventTransformer


def latent_regularizers(z, variance_target=1.0):
    """VICReg terms + collapse diagnostics. Lifted from stock_v2/graph_jepa.py
    `_latent_regularizers` (variance hinge keeps every dim expressive, which is
    exactly what a cosine objective destroys; covariance decorrelates dims)."""
    if z.dim() != 2 or z.shape[0] < 2:
        zero = z.new_tensor(0.0)
        return zero, zero, zero, zero
    zc = z - z.mean(dim=0, keepdim=True)
    n, d = zc.shape
    var = zc.var(dim=0, unbiased=False)
    std = torch.sqrt(var + 1e-6)
    var_loss = F.relu(variance_target - std).mean()
    cov = (zc.T @ zc) / max(n - 1, 1)
    off = cov - torch.diag_embed(torch.diagonal(cov))
    cov_loss = off.pow(2).sum() / d
    share = var / var.sum().clamp_min(1e-12)
    part_ratio = 1.0 / share.pow(2).sum().clamp_min(1e-12)   # 1 = collapsed, D = isotropic
    return var_loss, cov_loss, std.mean().detach(), part_ratio.detach()


class HomeJepa(nn.Module):
    def __init__(self, d=128, layers=4, heads=4, max_pos=258, ema=0.996, room_feats=False):
        super().__init__()
        self.context = TwoHeadEventTransformer(d, layers, heads, max_pos, room_feats)
        self.target = copy.deepcopy(self.context)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.target.eval()
        self.ema = ema
        self.predictor = nn.Sequential(
            nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        # event-indexed horizon conditioning: predict the latent at the k-th
        # next sighting (k in {1,2,4} -> slot 0,1,2)
        self.e_horizon = nn.Embedding(3, d)
        self.anchor_head = nn.Linear(d, MAX_LOC)     # imputation anchor

    def encode_ctx(self, b):
        return self.context.encode(b) + self.context.qdt_proj(b["qdt"])

    @torch.no_grad()
    def encode_tgt(self, b):
        self.target.eval()
        return self.target.encode(b) + self.target.qdt_proj(b["qdt"])

    @torch.no_grad()
    def update_target(self):
        for tp, cp in zip(self.target.parameters(), self.context.parameters()):
            tp.mul_(self.ema).add_(cp.detach(), alpha=1.0 - self.ema)
        for tb, cb in zip(self.target.buffers(), self.context.buffers()):
            tb.copy_(cb)

    def stage1_loss(self, b_ctx, b_tgt, w_var=1.0, w_cov=0.01, w_anchor=1.0,
                    variance_target=1.0, horizon=0):
        z_ctx = self.encode_ctx(b_ctx)
        h = self.e_horizon(torch.tensor(horizon, device=z_ctx.device))
        z_pred = self.predictor(z_ctx + h)
        z_tgt = self.encode_tgt(b_tgt)
        latent = (1.0 - F.cosine_similarity(z_pred, z_tgt, dim=-1)).mean()
        v, c, std, pr = latent_regularizers(z_pred, variance_target)
        # imputation anchor: reconstruct the observed room-dwell histogram from
        # the context latent. Computable from the context input alone (no leak),
        # and it survives stage 1 because it does not depend on the predictor.
        logits = self.anchor_head(z_ctx).masked_fill(~b_ctx["room_mask"], -1e9)
        tgt_h = b_ctx["hist"] * b_ctx["room_mask"].float()
        tgt_h = tgt_h / tgt_h.sum(-1, keepdim=True).clamp_min(1e-6)
        anchor = -(tgt_h * F.log_softmax(logits, -1)).sum(-1).mean()
        loss = latent + w_var * v + w_cov * c + w_anchor * anchor
        return loss, dict(latent=float(latent), var=float(v), cov=float(c),
                          anchor=float(anchor), std=float(std), part_ratio=float(pr))


class JepaProbe(nn.Module):
    """Stage 2: frozen context encoder + the same two-head readout Track S uses,
    so the A/B differs only in how the trunk was trained."""

    def __init__(self, jepa, freeze=True):
        super().__init__()
        self.trunk = jepa.context
        if freeze:
            for p in self.trunk.parameters():
                p.requires_grad_(False)
            self.trunk.eval()
        self.frozen = freeze
        d = self.trunk.q_head.out_features
        self.q_head = nn.Linear(d, d)
        self.gate = nn.Sequential(nn.Linear(2 * d + 5, d), nn.GELU(), nn.Linear(d, 1))
        # parity with Track S: auxiliary flat-softmax head over all slots
        self.aux_head = nn.Linear(d, d)

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.trunk.eval()
        return self

    def _feats(self, b):
        if self.frozen:
            with torch.no_grad():
                h = self.trunk.encode(b) + self.trunk.qdt_proj(b["qdt"])
                re = self.trunk.rooms(b)     # v1 rooms() already adds the anchor emb
            return h.detach(), re.detach()
        h = self.trunk.encode(b) + self.trunk.qdt_proj(b["qdt"])
        return h, self.trunk.rooms(b)

    def parts(self, b):
        h, re = self._feats(b)
        anchor_emb = re.gather(
            1, b["anchor"].view(-1, 1, 1).expand(-1, 1, re.size(-1))).squeeze(1)
        gate_logit = self.gate(torch.cat([h, anchor_emb, b["qdt"][:, 2:7]], -1)).squeeze(-1)
        hq = self.q_head(h)
        logits = torch.einsum("bd,brd->br", hq, re) / math.sqrt(hq.size(-1))
        anchor_1h = torch.zeros_like(logits, dtype=torch.bool).scatter_(
            1, b["anchor"].view(-1, 1), True)
        logits = logits.masked_fill(~b["room_mask"] | anchor_1h, -1e9)
        return gate_logit, torch.log_softmax(logits, -1)

    def log_prob(self, b):
        h, re = self._feats(b)
        anchor_emb = re.gather(
            1, b["anchor"].view(-1, 1, 1).expand(-1, 1, re.size(-1))).squeeze(1)
        gate_logit = self.gate(torch.cat([h, anchor_emb, b["qdt"][:, 2:7]], -1)).squeeze(-1)
        hq = self.q_head(h)
        logits = torch.einsum("bd,brd->br", hq, re) / math.sqrt(hq.size(-1))
        anchor_1h = torch.zeros_like(logits, dtype=torch.bool).scatter_(
            1, b["anchor"].view(-1, 1), True)
        logits = logits.masked_fill(~b["room_mask"] | anchor_1h, -1e9)
        log_g = F.logsigmoid(gate_logit).unsqueeze(-1)
        log_1mg = F.logsigmoid(-gate_logit).unsqueeze(-1)
        lp_other = log_g + torch.log_softmax(logits, -1)
        return torch.where(anchor_1h, log_1mg.expand_as(lp_other), lp_other) \
                    .masked_fill(~b["room_mask"], -1e9)

    def aux_log_prob(self, b):
        return self._aux_from(b, *self._feats(b))

    def _aux_from(self, b, h, re):
        hq = self.aux_head(h)
        logits = torch.einsum("bd,brd->br", hq, re) / math.sqrt(hq.size(-1))
        return torch.log_softmax(logits.masked_fill(~b["room_mask"], -1e9), -1)

    def _parts_from(self, b, h, re):
        anchor_emb = re.gather(
            1, b["anchor"].view(-1, 1, 1).expand(-1, 1, re.size(-1))).squeeze(1)
        gate_logit = self.gate(torch.cat([h, anchor_emb, b["qdt"][:, 2:7]], -1)).squeeze(-1)
        hq = self.q_head(h)
        logits = torch.einsum("bd,brd->br", hq, re) / math.sqrt(hq.size(-1))
        anchor_1h = torch.zeros_like(logits, dtype=torch.bool).scatter_(
            1, b["anchor"].view(-1, 1), True)
        logits = logits.masked_fill(~b["room_mask"] | anchor_1h, -1e9)
        return gate_logit, torch.log_softmax(logits, -1), anchor_1h

    def _mix_from(self, b, gate_logit, cond_lp, anchor_1h):
        log_g = F.logsigmoid(gate_logit).unsqueeze(-1)
        log_1mg = F.logsigmoid(-gate_logit).unsqueeze(-1)
        lp_other = log_g + cond_lp
        return torch.where(anchor_1h, log_1mg.expand_as(lp_other), lp_other) \
                    .masked_fill(~b["room_mask"], -1e9)

    def all_heads(self, b):
        """main log-prob, aux log-prob, gate logit -- ONE trunk pass."""
        h, re = self._feats(b)
        g, cond, a1 = self._parts_from(b, h, re)
        return self._mix_from(b, g, cond, a1), self._aux_from(b, h, re), g

    def forward(self, b):
        return self.log_prob(b)
