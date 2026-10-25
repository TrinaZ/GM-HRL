"""GM-HRL networks.

Shared encoder f_phi (edge-conditioned message passing, L layers, width d):
    h_v^0     = ReLU(W_in x_v)
    m_uv^l    = MLP_dir([h_u^l ; h_v^l ; e_uv])       (dir = forward / backward)
    a_v^l     = mean_{u in pred(v)} m_uv  ||  mean_{w in succ(v)} m_wv
    h_v^{l+1} = LayerNorm(h_v^l + MLP_upd([h_v^l ; a_v^l]))
    z_t       = mean_v h_v^L                          (global graph embedding)

Meta-Controller pi^M(g | s_t, z_t): diagonal Gaussian over g in R^{d_g};
value V^M(s_t, z_t).

Goal writing (graph as interface): the Meta-Controller's goal is written into
node attribute 17 of G_t as p_v = sigmoid(<W_g g, W_x x_v> / sqrt(d_g)); the
Sub-Controller sees g only through these node attributes, via the shared
encoder (no direct channel).

Sub-Controller pi^S: logits over dispatchable tasks MLP([h_v ; c(s_t)]) and a
WAIT logit MLP([z_t ; s_t]); masked softmax; value V^S(s_t, z_t).

Ablation switches: use_gnn=False replaces message passing by a per-node MLP
(L layers, no neighbour aggregation); hierarchical=False removes the
Meta-Controller (goal attribute fixed to 0.5).
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import NODE_DIM, EDGE_DIM, GLOBAL_DIM, GOAL_IDX


def mlp(i, h, o, layers=2):
    mods, d = [], i
    for _ in range(layers - 1):
        mods += [nn.Linear(d, h), nn.ReLU()]
        d = h
    mods.append(nn.Linear(d, o))
    return nn.Sequential(*mods)


class MPLayer(nn.Module):
    def __init__(self, d: int, use_gnn: bool = True):
        super().__init__()
        self.use_gnn = use_gnn
        if use_gnn:
            self.msg_f = mlp(2 * d + EDGE_DIM, d, d)
            self.msg_b = mlp(2 * d + EDGE_DIM, d, d)
            self.upd = mlp(3 * d, d, d)
        else:
            self.upd = mlp(d, d, d)
        self.ln = nn.LayerNorm(d)

    def forward(self, h, ei, ea):
        if not self.use_gnn:
            return self.ln(h + self.upd(h))
        n = h.size(0)
        if ei.size(1) == 0:
            agg = torch.zeros(n, 2 * h.size(1), device=h.device)
        else:
            u, v = ei[0], ei[1]
            mf = self.msg_f(torch.cat([h[u], h[v], ea], -1))     # u -> v
            mb = self.msg_b(torch.cat([h[v], h[u], ea], -1))     # v -> u
            af = torch.zeros_like(h).index_add_(0, v, mf)
            ab = torch.zeros_like(h).index_add_(0, u, mb)
            cf = torch.zeros(n, 1, device=h.device).index_add_(0, v, torch.ones(len(v), 1, device=h.device))
            cb = torch.zeros(n, 1, device=h.device).index_add_(0, u, torch.ones(len(u), 1, device=h.device))
            agg = torch.cat([af / cf.clamp(min=1), ab / cb.clamp(min=1)], -1)
        return self.ln(h + self.upd(torch.cat([h, agg], -1)))


class Encoder(nn.Module):
    def __init__(self, d: int = 64, layers: int = 2, use_gnn: bool = True):
        super().__init__()
        self.inp = nn.Linear(NODE_DIM, d)
        self.layers = nn.ModuleList([MPLayer(d, use_gnn) for _ in range(layers)])

    def forward(self, x, ei, ea):
        h = F.relu(self.inp(x))
        for l in self.layers:
            h = l(h, ei, ea)
        z = h.mean(0) if h.size(0) > 0 else torch.zeros(h.size(1), device=h.device)
        return h, z


class GMHRLNet(nn.Module):
    def __init__(self, d: int = 64, layers: int = 2, d_g: int = 8, hidden: int = 128,
                 use_gnn: bool = True, hierarchical: bool = True):
        super().__init__()
        self.d, self.d_g, self.hierarchical = d, d_g, hierarchical
        self.enc = Encoder(d, layers, use_gnn)
        # goal writer
        self.Wg = nn.Linear(d_g, d_g, bias=False)
        self.Wx = nn.Linear(NODE_DIM, d_g)
        # meta
        self.meta_mu = mlp(GLOBAL_DIM + d, hidden, d_g)
        self.meta_logstd = nn.Parameter(torch.full((d_g,), -1.5))
        self.meta_v = mlp(GLOBAL_DIM + d, hidden, 1)
        # sub
        self.ctx = mlp(GLOBAL_DIM, hidden, d)
        self.score = mlp(2 * d, hidden, 1)
        self.wait = mlp(GLOBAL_DIM + d, hidden, 1)
        self.sub_v = mlp(GLOBAL_DIM + d, hidden, 1)
        with torch.no_grad():                 # start from a (mostly) non-delay policy
            self.wait[-1].bias.fill_(-3.0)

    # --- goal written into the graph
    def write_goal(self, x: torch.Tensor, g: Optional[torch.Tensor]) -> torch.Tensor:
        x = x.clone()
        if not self.hierarchical or g is None:
            x[:, GOAL_IDX] = 0.5
        else:
            xg = x.clone()
            xg[:, GOAL_IDX] = 0.0
            p = torch.sigmoid((self.Wx(xg) * self.Wg(g)).sum(-1) / math.sqrt(self.d_g))
            x[:, GOAL_IDX] = p
        return x

    def encode(self, x, ei, ea, g):
        return self.enc(self.write_goal(x, g), ei, ea)

    # --- meta
    def meta_dist(self, s, z):
        inp = torch.cat([s, z], -1)
        mu = self.meta_mu(inp)
        std = self.meta_logstd.clamp(-3, 1).exp()
        return torch.distributions.Normal(mu, std), self.meta_v(inp).squeeze(-1)

    # --- sub
    def sub_logits(self, h, z, s, cand, can_wait: bool):
        c = self.ctx(s).expand(len(cand), -1)
        lt = self.score(torch.cat([h[cand], c], -1)).squeeze(-1)
        lw = self.wait(torch.cat([s, z], -1))
        if not can_wait:
            lw = torch.full_like(lw, -1e9)
        logits = torch.cat([lt, lw], 0)
        v = self.sub_v(torch.cat([s, z], -1)).squeeze(-1)
        return logits, v

    def goal_annotation(self, x, g):
        """Node goal attributes p_v (for Sim)."""
        return self.write_goal(x, g)[:, GOAL_IDX]
