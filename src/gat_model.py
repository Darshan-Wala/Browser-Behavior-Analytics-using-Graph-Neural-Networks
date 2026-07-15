"""
gat_model.py
============
Graph Attention Network (GAT) for browser-session anomaly classification.

Architecture
------------
    Node features  [N × in_channels]
         ↓
    GATConv(in_channels → hidden)  +  ELU  +  Dropout
         ↓
    GATConv(hidden → hidden)       +  ELU  +  Dropout
         ↓
    Global Mean + Max Pooling      →  [batch × 2*hidden]
         ↓
    MLP Classifier (2 hidden layers)
         ↓
    Logits [batch × 2]   (normal=0, malicious=1)

Attention weights from the final GATConv are exposed for explainability.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool


class BrowserGAT(nn.Module):
    """
    GAT classifier for browser-session graphs.

    Parameters
    ----------
    in_channels : int
        Node feature dimensionality (must match ``FEATURE_DIM`` in
        ``graph_builder.py``).
    hidden_channels : int
        Width of each GAT layer's output (per attention head).
    num_heads : int
        Number of attention heads in both GAT layers.
    num_classes : int
        Output classes — 2 for binary normal/malicious.
    dropout : float
        Dropout probability applied after each activation and in GAT layers.
    """

    def __init__(
        self,
        in_channels: int = 32,
        hidden_channels: int = 64,
        num_heads: int = 4,
        num_classes: int = 2,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        self.dropout_p = dropout

        # ---- Layer 1: input → hidden*num_heads ---------------------------- #
        self.conv1 = GATConv(
            in_channels,
            hidden_channels,
            heads=num_heads,
            dropout=dropout,
            concat=True,        # output: hidden * num_heads
        )

        # ---- Layer 2: hidden*num_heads → hidden (single head, return attn) #
        self.conv2 = GATConv(
            hidden_channels * num_heads,
            hidden_channels,
            heads=1,
            dropout=dropout,
            concat=False,       # output: hidden
        )

        # ---- Batch normalisation ----------------------------------------- #
        self.bn1 = nn.BatchNorm1d(hidden_channels * num_heads)
        self.bn2 = nn.BatchNorm1d(hidden_channels)

        # ---- Graph-level pooled representation: mean + max → 2*hidden ----- #
        pool_dim = hidden_channels * 2

        # ---- MLP classifier ------------------------------------------------ #
        self.classifier = nn.Sequential(
            nn.Linear(pool_dim, pool_dim // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(pool_dim // 2, num_classes),
        )

        self._last_attn_weights: torch.Tensor | None = None
        self._last_attn_edge_index: torch.Tensor | None = None

    # ---------------------------------------------------------------------- #
    # Forward pass                                                             #
    # ---------------------------------------------------------------------- #

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : [N, in_channels]
        edge_index : [2, E]
        batch : [N]  — PyG batch vector
        return_attention : bool
            If True, also return ``(logits, attention_weights, attn_edge_index)``.

        Returns
        -------
        logits : [B, num_classes]
        (optional) attn_weights : [E, 1]
        (optional) attn_edge_index : [2, E]
        """
        # ---- Conv 1 ------------------------------------------------------- #
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)

        # ---- Conv 2 (with attention weights) ------------------------------ #
        x, (attn_edge_index, attn_weights) = self.conv2(
            x, edge_index, return_attention_weights=True
        )
        x = self.bn2(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)

        # Cache for explainability
        self._last_attn_weights    = attn_weights.detach()
        self._last_attn_edge_index = attn_edge_index.detach()

        # ---- Global pooling ----------------------------------------------- #
        x_mean = global_mean_pool(x, batch)   # [B, hidden]
        x_max  = global_max_pool(x, batch)    # [B, hidden]
        x_pool = torch.cat([x_mean, x_max], dim=-1)  # [B, 2*hidden]

        # ---- Classifier --------------------------------------------------- #
        logits = self.classifier(x_pool)      # [B, num_classes]

        if return_attention:
            return logits, attn_weights, attn_edge_index
        return logits

    # ---------------------------------------------------------------------- #
    # Inference helpers                                                        #
    # ---------------------------------------------------------------------- #

    @torch.no_grad()
    def predict_proba(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Return softmax probabilities [B, 2] without gradient tracking."""
        self.eval()
        logits = self(x, edge_index, batch)
        return F.softmax(logits, dim=-1)

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Return predicted class indices [B]."""
        return self.predict_proba(x, edge_index, batch).argmax(dim=-1)

    # ---------------------------------------------------------------------- #
    # Serialisation                                                            #
    # ---------------------------------------------------------------------- #

    def save(self, path: str) -> None:
        """Save model state dict."""
        torch.save(self.state_dict(), path)

    @classmethod
    def load(
        cls,
        path: str,
        device: torch.device | str = "cpu",
        **init_kwargs,
    ) -> "BrowserGAT":
        """Load a previously saved model."""
        model = cls(**init_kwargs)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        model.to(device)
        model.eval()
        return model

    # ---------------------------------------------------------------------- #
    # Explainability                                                           #
    # ---------------------------------------------------------------------- #

    def get_attention_weights(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        """
        Return cached attention weights from the last forward pass.

        Returns
        -------
        (edge_index [2, E], weights [E, 1])  or  None if not yet computed.
        """
        if self._last_attn_weights is None:
            return None
        return self._last_attn_edge_index, self._last_attn_weights

    def get_suspicious_nodes(
        self,
        node_ids: list[str],
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """
        Return the ``top_k`` nodes with the highest summed incoming attention.

        Parameters
        ----------
        node_ids : list[str]
            Canonical node names (``Data.node_ids``).
        top_k : int

        Returns
        -------
        list of ``(node_name, attention_score)`` sorted descending.
        """
        if self._last_attn_weights is None:
            return []
        attn = self._last_attn_weights.squeeze(-1)          # [E]
        dst  = self._last_attn_edge_index[1]                # [E]
        scores = torch.zeros(len(node_ids))
        scores.scatter_add_(0, dst.cpu(), attn.cpu())
        top_indices = scores.argsort(descending=True)[:top_k]
        return [(node_ids[i], float(scores[i])) for i in top_indices]
