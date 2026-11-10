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

Reservation (GM-HRL+R, n_res = 1): the Meta-Controller's Gaussian has one
extra dimension; g_res = sigmoid(g[d_g]) is the reservation level used by
gmhrl/reserve.py. The goal writer uses only g[:d_g].

GM-HRL v2 (goal_mode="workflow", MD2POS-MW, HIERARCHY_SPEC.md section 3): the
goal is written per workflow, p_w = sigmoid(<W_g g, W_x x_w> / sqrt(d_g)), where
x_w are the three workflow columns of the node features; all tasks of a
workflow share these columns and therefore receive the same p_w.

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

from .features import NODE_DIM, EDGE_DIM, GLOBAL_DIM, GOAL_IDX, WF_IDX, WF_DIM


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
    def __init__(self, d: int = 64, layers: int = 2, use_gnn: bool = True, node_dim: int = NODE_DIM):
        super().__init__()
        self.inp = nn.Linear(node_dim, d)
        self.layers = nn.ModuleList([MPLayer(d, use_gnn) for _ in range(layers)])

    def forward(self, x, ei, ea):
        h = F.relu(self.inp(x))
        for l in self.layers:
            h = l(h, ei, ea)
        z = h.mean(0) if h.size(0) > 0 else torch.zeros(h.size(1), device=h.device)
        return h, z


class GMHRLNet(nn.Module):
    def __init__(self, d: int = 64, layers: int = 2, d_g: int = 8, hidden: int = 128,
                 use_gnn: bool = True, hierarchical: bool = True, n_res: int = 0,
                 res_bias_init: float = -1.0, node_dim: int = NODE_DIM, goal_mode: str = "task"):
        super().__init__()
        assert goal_mode in ("task", "workflow")
        self.d, self.d_g, self.hierarchical, self.n_res = d, d_g, hierarchical, n_res
        self.goal_mode = goal_mode
        self.enc = Encoder(d, layers, use_gnn, node_dim)
        # goal writer
        self.Wg = nn.Linear(d_g, d_g, bias=False)
        self.Wx = nn.Linear(WF_DIM if goal_mode == "workflow" else node_dim, d_g)
        # meta
        self.meta_mu = mlp(GLOBAL_DIM + d, hidden, d_g + n_res)
        self.meta_logstd = nn.Parameter(torch.full((d_g + n_res,), -1.5))
        if n_res:
            with torch.no_grad():             # conservative start: sigmoid(-1) = 0.27
                self.meta_mu[-1].bias[d_g:] = res_bias_init
        self.meta_v = mlp(GLOBAL_DIM + d, hidden, 1)
        # sub
        self.ctx = mlp(GLOBAL_DIM, hidden, d)
        self.score = mlp(2 * d, hidden, 1)
        self.wait = mlp(GLOBAL_DIM + d, hidden, 1)
        self.sub_v = mlp(GLOBAL_DIM + d, hidden, 1)
        with torch.no_grad():                 # start from a (mostly) non-delay policy
            self.wait[-1].bias.fill_(-3.0)

    # --- goal written into the graph
    def write_goal(self, x: torch.Tensor, g: Optional[torch.Tensor],
                   p_override: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Write the goal attribute. ``p_override`` (evaluation diagnostics only)
        replaces the written attributes, e.g. by 0.5 or by shuffled goals."""
        x = x.clone()
        if p_override is not None:
            x[:, GOAL_IDX] = p_override
        elif not self.hierarchical or g is None:
            x[:, GOAL_IDX] = 0.5
        else:
            if self.goal_mode == "workflow":
                xg = x[:, WF_IDX:WF_IDX + WF_DIM].clone()
            else:
                xg = x.clone()
                xg[:, GOAL_IDX] = 0.0
            p = torch.sigmoid((self.Wx(xg) * self.Wg(g[..., :self.d_g])).sum(-1) / math.sqrt(self.d_g))
            x[:, GOAL_IDX] = p
        return x

    def encode(self, x, ei, ea, g, p_override=None):
        return self.enc(self.write_goal(x, g, p_override), ei, ea)

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

    def g_res(self, g) -> float:
        """Reservation level in [0, 1] (0 without a reservation head or goal)."""
        if not self.n_res or g is None:
            return 0.0
        return float(torch.sigmoid(g[..., self.d_g]))

    def goal_annotation(self, x, g):
        """Node goal attributes p_v (for Sim)."""
        return self.write_goal(x, g)[:, GOAL_IDX]
