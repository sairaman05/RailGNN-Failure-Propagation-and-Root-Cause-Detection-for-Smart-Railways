"""
Phase 4 — Temporal Graph Neural Network (TGNN)

Architecture:
  Input  : (T, N, F)  — T time steps, N nodes, F features
  Spatial: GCN layers aggregate neighbour information at each time step
  Temporal: GRU processes the T-step sequence per node
  Output heads:
    - risk_cls : (N, 4)   softmax — normal / low / medium / high
    - risk_reg : (N, 1)   sigmoid — continuous risk score 0-1
    - root_cause: (N,)    attention weight — propagation source probability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ════════════════════════════════════════════════════════════════════
# Graph Convolution Layer (manual, no PyG required)
# ════════════════════════════════════════════════════════════════════

class GraphConvLayer(nn.Module):
    """
    Simple GCN: h_i = ReLU( W * mean(h_j for j in N(i) ∪ {i}) )
    Works on dense (N, F) node features with edge_index in COO format.
    Does not require torch_geometric — runs on any machine with PyTorch.
    """

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.lin  = nn.Linear(in_dim, out_dim, bias=bias)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        x          : (N, F)
        edge_index : (2, E)
        returns    : (N, out_dim)
        """
        N = x.size(0)
        src, dst = edge_index[0], edge_index[1]

        # Aggregate: for each node, sum neighbour features
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, x[src])

        # Degree for normalisation (add 1 for self-loop)
        deg = torch.zeros(N, device=x.device)
        deg.index_add_(0, dst, torch.ones(len(dst), device=x.device))
        deg = (deg + 1.0).clamp(min=1.0).unsqueeze(-1)

        # Self-loop + normalise
        out = (x + agg) / deg
        out = self.lin(out)
        out = self.norm(out)
        return F.relu(out)


# ════════════════════════════════════════════════════════════════════
# RootCauseAttention
# ════════════════════════════════════════════════════════════════════

class RootCauseAttention(nn.Module):
    """
    Given the final hidden state h of each node (N, H),
    compute an attention score over edges to identify
    which node is most likely the propagation source.

    Returns per-node root-cause probability (N,).
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.edge_scorer = nn.Linear(hidden_dim * 2, 1)
        self.node_scorer = nn.Linear(hidden_dim, 1)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        h          : (N, H)
        edge_index : (2, E)
        returns    : (N,) softmax attention — root cause probability
        """
        N = h.size(0)
        src, dst = edge_index[0], edge_index[1]

        # Edge-level scores: concat src+dst features → scalar
        edge_feats  = torch.cat([h[src], h[dst]], dim=-1)   # (E, 2H)
        edge_scores = self.edge_scorer(edge_feats).squeeze(-1)  # (E,)

        # Accumulate edge scores to destination nodes
        node_edge_score = torch.zeros(N, device=h.device)
        node_edge_score.index_add_(0, dst, edge_scores)

        # Node-level self score
        node_self_score = self.node_scorer(h).squeeze(-1)   # (N,)

        combined = node_edge_score + node_self_score
        return torch.softmax(combined, dim=0)               # (N,)


# ════════════════════════════════════════════════════════════════════
# TGNN — main model
# ════════════════════════════════════════════════════════════════════

class TGNN(nn.Module):
    """
    Temporal Graph Neural Network for railway failure prediction.

    Forward input:
        x          : (B, T, N, F)   B=batch, T=seq_len, N=nodes, F=features
        edge_index : (2, E)         static graph topology

    Forward output (dict):
        risk_cls   : (B, N, 4)      classification logits
        risk_reg   : (B, N)         regression scores 0-1
        root_cause : (B, N)         root cause probability per node
        hidden     : (B, N, H)      final GRU hidden state (for inspection)
    """

    def __init__(self,
                 in_features:  int = 16,
                 gcn_hidden:   int = 64,
                 gcn_layers:   int = 2,
                 gru_hidden:   int = 128,
                 gru_layers:   int = 2,
                 n_classes:    int = 4,
                 dropout:      float = 0.3):
        super().__init__()

        self.in_features = in_features
        self.gcn_hidden  = gcn_hidden
        self.gru_hidden  = gru_hidden
        self.n_classes   = n_classes

        # ── Spatial: stacked GCN ────────────────────────────────────
        gcn_dims = [in_features] + [gcn_hidden] * gcn_layers
        self.gcn_layers = nn.ModuleList([
            GraphConvLayer(gcn_dims[i], gcn_dims[i+1])
            for i in range(gcn_layers)
        ])
        self.gcn_dropout = nn.Dropout(dropout)

        # ── Temporal: GRU per node ───────────────────────────────────
        # Input to GRU is the GCN output per time step: gcn_hidden
        self.gru = nn.GRU(
            input_size=gcn_hidden,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        self.gru_dropout = nn.Dropout(dropout)

        # ── Output heads ────────────────────────────────────────────
        self.cls_head = nn.Sequential(
            nn.Linear(gru_hidden, gru_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gru_hidden // 2, n_classes),
        )
        self.reg_head = nn.Sequential(
            nn.Linear(gru_hidden, gru_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gru_hidden // 2, 1),
            nn.Sigmoid(),
        )

        # ── Root cause attention ────────────────────────────────────
        self.root_cause_attn = RootCauseAttention(gru_hidden)

        self._init_weights()

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if "weight" in name:
                        nn.init.orthogonal_(param)
                    elif "bias" in name:
                        nn.init.zeros_(param)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor) -> dict:
        """
        x          : (B, T, N, F)
        edge_index : (2, E)
        """
        B, T, N, F = x.shape

        # ── 1. Spatial encoding: apply GCN at each time step ─────────
        # Reshape to (B*T, N, F), process, reshape back
        x_flat = x.view(B * T, N, F)           # (B*T, N, F)

        gcn_out = x_flat
        for gcn in self.gcn_layers:
            # Process each (B*T) graph independently
            # edge_index is the same for all — broadcast manually
            out_list = []
            for g in range(B * T):
                out_list.append(gcn(gcn_out[g], edge_index))
            gcn_out = torch.stack(out_list, dim=0)  # (B*T, N, gcn_hidden)
        gcn_out = self.gcn_dropout(gcn_out)

        gcn_out = gcn_out.view(B, T, N, self.gcn_hidden)  # (B, T, N, H_gcn)

        # ── 2. Temporal encoding: GRU over T steps per node ──────────
        # Reshape to (B*N, T, gcn_hidden) so GRU sees one sequence per node
        gru_in = gcn_out.permute(0, 2, 1, 3)             # (B, N, T, H_gcn)
        gru_in = gru_in.reshape(B * N, T, self.gcn_hidden)

        gru_out, _ = self.gru(gru_in)    # (B*N, T, gru_hidden)
        h = gru_out[:, -1, :]            # last step: (B*N, gru_hidden)
        h = self.gru_dropout(h)
        h = h.view(B, N, self.gru_hidden)               # (B, N, gru_hidden)

        # ── 3. Output heads ──────────────────────────────────────────
        risk_cls   = self.cls_head(h)              # (B, N, 4)
        risk_reg   = self.reg_head(h).squeeze(-1)  # (B, N)

        # ── 4. Root cause attention ───────────────────────────────────
        # Run per sample in the batch
        rc_list = []
        for b in range(B):
            rc_list.append(self.root_cause_attn(h[b], edge_index))
        root_cause = torch.stack(rc_list, dim=0)   # (B, N)

        return {
            "risk_cls":   risk_cls,
            "risk_reg":   risk_reg,
            "root_cause": root_cause,
            "hidden":     h,
        }

    # ------------------------------------------------------------------
    # Convenience: count parameters
    # ------------------------------------------------------------------

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ════════════════════════════════════════════════════════════════════
# Model factory
# ════════════════════════════════════════════════════════════════════

def build_model(cfg: dict = None) -> TGNN:
    """Build TGNN from a config dict (or defaults)."""
    defaults = dict(
        in_features=16,
        gcn_hidden=64,
        gcn_layers=2,
        gru_hidden=128,
        gru_layers=2,
        n_classes=4,
        dropout=0.3,
    )
    if cfg:
        defaults.update(cfg)
    return TGNN(**defaults)