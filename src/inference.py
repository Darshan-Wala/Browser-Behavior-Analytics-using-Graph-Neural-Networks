"""
inference.py
============
Real-time session anomaly detection.

Provides:

* :func:`detect_session`  – high-level one-call API
* :class:`SessionDetector` – stateful object for repeated inference

Usage
-----
    from inference import SessionDetector

    detector = SessionDetector("models/browser_gat.pt")
    result = detector.detect(session_logs)
    print(result)
    # {
    #   "session_id": "1J1QD8NY44",
    #   "label": "malicious",
    #   "malicious_probability": 0.91,
    #   "confidence": "high",
    #   "suspicious_nodes": [("eval", 0.72), ("atob", 0.68), ...],
    # }
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from gat_model import BrowserGAT
from graph_builder import FEATURE_DIM, build_graph_from_session


# ---------------------------------------------------------------------------
# Default model hyper-parameters (must match training)
# ---------------------------------------------------------------------------

_DEFAULT_MODEL_KWARGS = dict(
    in_channels=FEATURE_DIM,
    hidden_channels=64,
    num_heads=4,
    num_classes=2,
    dropout=0.0,          # disable dropout at inference time
)


# ---------------------------------------------------------------------------
# Stateful detector class
# ---------------------------------------------------------------------------

class SessionDetector:
    """
    Wraps a trained :class:`BrowserGAT` for real-time session inference.

    Parameters
    ----------
    model_path : str
        Path to the saved ``browser_gat.pt`` state dict.
    device : str | torch.device, optional
        ``"cuda"`` or ``"cpu"``.  Defaults to CUDA if available.
    model_kwargs : dict, optional
        Hyper-parameters forwarded to :class:`BrowserGAT.__init__`.
        Defaults to ``_DEFAULT_MODEL_KWARGS``.
    """

    def __init__(
        self,
        model_path: str = "models/browser_gat.pt",
        device: str | torch.device | None = None,
        model_kwargs: dict | None = None,
    ) -> None:
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        kwargs = model_kwargs or _DEFAULT_MODEL_KWARGS
        self.model = BrowserGAT.load(model_path, device=self.device, **kwargs)

    # ---------------------------------------------------------------------- #
    # Core detection                                                           #
    # ---------------------------------------------------------------------- #

    def detect(
        self,
        session_logs: dict[str, list[dict]],
        top_k_nodes: int = 5,
    ) -> dict:
        """
        Build a graph from ``session_logs`` and run inference.

        Parameters
        ----------
        session_logs : dict[str, list[dict]]
            Same format as :func:`graph_builder.build_graph_from_session`.
        top_k_nodes : int
            How many suspicious nodes to highlight.

        Returns
        -------
        dict with keys:
            ``label``, ``malicious_probability``, ``normal_probability``,
            ``confidence``, ``suspicious_nodes``, ``num_nodes``, ``num_edges``
        """
        graph: Data = build_graph_from_session(session_logs, label=0)
        graph = graph.to(self.device)

        # Single-graph batch
        batch_vec = torch.zeros(graph.num_nodes, dtype=torch.long, device=self.device)

        self.model.eval()
        with torch.no_grad():
            logits, attn_weights, attn_edge_index = self.model(
                graph.x, graph.edge_index, batch_vec, return_attention=True
            )
        probs = F.softmax(logits, dim=-1)[0]  # [2]

        mal_prob = float(probs[1].item())
        nor_prob = float(probs[0].item())
        label    = "malicious" if mal_prob >= 0.5 else "normal"

        if mal_prob >= 0.8 or nor_prob >= 0.8:
            confidence = "high"
        elif mal_prob >= 0.65 or nor_prob >= 0.65:
            confidence = "medium"
        else:
            confidence = "low"

        # Suspicious nodes via attention
        node_ids = getattr(graph, "node_ids", [])
        suspicious = self.model.get_suspicious_nodes(node_ids, top_k=top_k_nodes)

        return {
            "label":                label,
            "malicious_probability": mal_prob,
            "normal_probability":    nor_prob,
            "confidence":            confidence,
            "suspicious_nodes":      suspicious,
            "num_nodes":             graph.num_nodes,
            "num_edges":             graph.num_edges,
        }


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def detect_session(
    session_logs: dict[str, list[dict]],
    model_path: str = "models/browser_gat.pt",
    device: str | None = None,
) -> dict:
    """
    One-shot session detection.  Loads the model fresh each call — use
    :class:`SessionDetector` for repeated inference to avoid reload overhead.

    Parameters
    ----------
    session_logs : dict
        Grouped log records for a single session.
    model_path : str
        Path to trained model state dict.
    device : str | None
        ``"cuda"`` / ``"cpu"`` / ``None`` (auto).

    Returns
    -------
    dict — same structure as :meth:`SessionDetector.detect`.
    """
    detector = SessionDetector(model_path=model_path, device=device)
    return detector.detect(session_logs)


# ---------------------------------------------------------------------------
# Batch inference
# ---------------------------------------------------------------------------

def detect_batch(
    sessions: list[dict[str, list[dict]]],
    model_path: str = "models/browser_gat.pt",
    device: str | None = None,
    batch_size: int = 32,
) -> list[dict]:
    """
    Run inference over multiple sessions efficiently using a DataLoader.

    Parameters
    ----------
    sessions : list of session_logs dicts
    model_path : str
    device : str | None
    batch_size : int

    Returns
    -------
    list[dict] — one result dict per session.
    """
    detector = SessionDetector(model_path=model_path, device=device)

    graphs = [build_graph_from_session(s, label=0) for s in sessions]
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)

    all_probs: list[torch.Tensor] = []
    detector.model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(detector.device)
            logits = detector.model(batch.x, batch.edge_index, batch.batch)
            probs  = F.softmax(logits, dim=-1)
            all_probs.append(probs.cpu())

    all_probs_tensor = torch.cat(all_probs, dim=0)  # [N, 2]

    results = []
    for i, (s, g) in enumerate(zip(sessions, graphs)):
        mal_prob = float(all_probs_tensor[i, 1])
        nor_prob = float(all_probs_tensor[i, 0])
        results.append({
            "label":                 "malicious" if mal_prob >= 0.5 else "normal",
            "malicious_probability":  mal_prob,
            "normal_probability":     nor_prob,
            "confidence": (
                "high"   if max(mal_prob, nor_prob) >= 0.8 else
                "medium" if max(mal_prob, nor_prob) >= 0.65 else
                "low"
            ),
            "num_nodes": g.num_nodes,
            "num_edges": g.num_edges,
        })
    return results


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json, argparse

    parser = argparse.ArgumentParser(description="Detect a session from JSON logs")
    parser.add_argument("--logs",  required=True, help="Path to session_logs JSON dict")
    parser.add_argument("--model", default="models/browser_gat.pt")
    args = parser.parse_args()

    with open(args.logs) as f:
        session_logs = json.load(f)

    result = detect_session(session_logs, model_path=args.model)
    print(json.dumps(result, indent=2))
