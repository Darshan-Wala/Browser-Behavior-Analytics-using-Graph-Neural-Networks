"""
visualize_graph.py
==================
Visualisation utilities for browser-session behavioral graphs.

Features
--------
* Draw a session graph with nodes coloured by type (JS API, DOM, network…)
* Overlay GAT attention weights on edges (thicker = higher attention)
* Highlight the exploit-chain sub-graph
* Save or display interactively

Usage
-----
    python visualize_graph.py --session 1J1QD8NY44 --root data
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from graph_builder import EXPLOIT_CHAINS, build_graph_from_session
from gat_model import BrowserGAT

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

matplotlib.rcParams.update({"font.family": "monospace"})

# Colour per node-type prefix
NODE_COLOURS: dict[str, str] = {
    "eval":           "#e74c3c",
    "atob":           "#c0392b",
    "document.write": "#e67e22",
    "innerHTML":      "#f39c12",
    "iframe":         "#8e44ad",
    "fetch":          "#2980b9",
    "dynamic_script": "#16a085",
    "setTimeout":     "#d35400",
    "dom:":           "#27ae60",
    "net:":           "#2471a3",
    "script:":        "#117a65",
    "nav:":           "#7d6608",
    "__default__":    "#7f8c8d",
}

EXPLOIT_EDGE_COLOR  = "#e74c3c"
TEMPORAL_EDGE_COLOR = "#aab7b8"
CHAIN_HIGHLIGHT     = "#e74c3c"


def _node_color(node_id: str) -> str:
    for prefix, color in NODE_COLOURS.items():
        if node_id.startswith(prefix):
            return color
    return NODE_COLOURS["__default__"]


def _short_label(node_id: str, max_len: int = 18) -> str:
    label = node_id.split(":")[-1] if ":" in node_id else node_id
    return label[:max_len] + "…" if len(label) > max_len else label


# ---------------------------------------------------------------------------
# Core drawing helpers
# ---------------------------------------------------------------------------

def build_nx_graph(
    node_ids: list[str],
    edge_index: np.ndarray,          # shape [2, E]
    attn_weights: Optional[np.ndarray] = None,  # shape [E]
) -> nx.DiGraph:
    """
    Convert PyG graph tensors to a NetworkX DiGraph.

    Parameters
    ----------
    node_ids : list[str]
    edge_index : [2, E] numpy array
    attn_weights : [E] numpy array or None

    Returns
    -------
    nx.DiGraph with node attribute ``type`` and edge attributes
    ``weight`` (attention or 1.0), ``is_chain``.
    """
    G = nx.DiGraph()
    G.add_nodes_from(range(len(node_ids)))
    nx.set_node_attributes(G, {i: nid for i, nid in enumerate(node_ids)}, "label")

    n_edges = edge_index.shape[1]
    chain_pairs = {(src, dst) for src, dst in EXPLOIT_CHAINS}

    for e in range(n_edges):
        src, dst = int(edge_index[0, e]), int(edge_index[1, e])
        weight   = float(attn_weights[e]) if attn_weights is not None else 1.0
        src_name = node_ids[src] if src < len(node_ids) else str(src)
        dst_name = node_ids[dst] if dst < len(node_ids) else str(dst)
        is_chain = any(
            (src_name.endswith(s) or s == src_name) and
            (dst_name.endswith(d) or d == dst_name)
            for s, d in chain_pairs
        )
        G.add_edge(src, dst, weight=weight, is_chain=is_chain)

    return G


def draw_session_graph(
    node_ids: list[str],
    edge_index: np.ndarray,
    attn_weights: Optional[np.ndarray] = None,
    label: int = 0,
    session_id: str = "",
    save_path: Optional[str] = None,
    figsize: tuple[int, int] = (14, 9),
    layout: str = "spring",
) -> None:
    """
    Draw a full session graph.

    Parameters
    ----------
    node_ids : list[str]
    edge_index : [2, E] numpy array
    attn_weights : [E] numpy array (from GAT), or None for uniform width
    label : int  0=normal, 1=malicious
    session_id : str
    save_path : str or None  – PNG path to save; None → plt.show()
    figsize : tuple
    layout : str  – "spring" | "kamada_kawai" | "circular"
    """
    G = build_nx_graph(node_ids, edge_index, attn_weights)

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    # Layout
    pos_fn = {
        "spring":       nx.spring_layout,
        "kamada_kawai": nx.kamada_kawai_layout,
        "circular":     nx.circular_layout,
    }.get(layout, nx.spring_layout)
    pos = pos_fn(G, seed=42)

    # Node styling
    node_colors = [_node_color(node_ids[n]) for n in G.nodes()]
    node_sizes  = [
        400 + 300 * float(G.in_degree(n))
        for n in G.nodes()
    ]

    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.9,
    )

    # Labels
    labels = {n: _short_label(node_ids[n]) for n in G.nodes()}
    nx.draw_networkx_labels(
        G, pos, labels=labels, ax=ax,
        font_size=7, font_color="white",
    )

    # Edges — exploit-chain edges in red, temporal in grey
    chain_edges    = [(u, v) for u, v, d in G.edges(data=True) if d.get("is_chain")]
    temporal_edges = [(u, v) for u, v, d in G.edges(data=True) if not d.get("is_chain")]

    # Edge widths from attention
    def _widths(edges: list) -> list[float]:
        if attn_weights is None:
            return [1.5] * len(edges)
        return [
            1.0 + 4.0 * float(G[u][v].get("weight", 0.0))
            for u, v in edges
        ]

    if temporal_edges:
        nx.draw_networkx_edges(
            G, pos, edgelist=temporal_edges, ax=ax,
            edge_color=TEMPORAL_EDGE_COLOR,
            width=_widths(temporal_edges),
            arrows=True, arrowsize=12,
            connectionstyle="arc3,rad=0.07",
            alpha=0.6,
        )
    if chain_edges:
        nx.draw_networkx_edges(
            G, pos, edgelist=chain_edges, ax=ax,
            edge_color=EXPLOIT_EDGE_COLOR,
            width=_widths(chain_edges),
            arrows=True, arrowsize=16,
            connectionstyle="arc3,rad=0.15",
            alpha=0.9,
        )

    # Title
    status_color = "#e74c3c" if label == 1 else "#2ecc71"
    status_text  = "⚠ MALICIOUS" if label == 1 else "✓ NORMAL"
    title = (
        f"Browser Session Graph — {session_id}\n"
        f"[{status_text}]  |  {len(node_ids)} nodes  |  {edge_index.shape[1]} edges"
    )
    ax.set_title(title, color=status_color, fontsize=12, pad=12)

    # Legend
    legend_entries = [
        mpatches.Patch(color=EXPLOIT_EDGE_COLOR, label="Exploit chain edge"),
        mpatches.Patch(color=TEMPORAL_EDGE_COLOR, label="Temporal sequence edge"),
    ]
    type_colors = {
        "JS API":   "#e74c3c",
        "DOM":      "#27ae60",
        "Network":  "#2471a3",
        "Script":   "#117a65",
        "Navigate": "#7d6608",
    }
    for typ, col in type_colors.items():
        legend_entries.append(mpatches.Patch(color=col, label=typ))

    ax.legend(
        handles=legend_entries,
        loc="lower left",
        fontsize=7,
        framealpha=0.3,
        labelcolor="white",
        facecolor="#161b22",
        edgecolor="#30363d",
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  Saved graph → {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Attention-overlay visualisation
# ---------------------------------------------------------------------------

def draw_attention_heatmap(
    node_ids: list[str],
    edge_index: np.ndarray,
    attn_weights: np.ndarray,
    top_k: int = 10,
    save_path: Optional[str] = None,
) -> None:
    """
    Draw a bar chart of the ``top_k`` edges by attention weight.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    n_edges = edge_index.shape[1]
    edge_labels = []
    for e in range(n_edges):
        src = edge_index[0, e]
        dst = edge_index[1, e]
        src_name = node_ids[src] if src < len(node_ids) else str(src)
        dst_name = node_ids[dst] if dst < len(node_ids) else str(dst)
        edge_labels.append(f"{_short_label(src_name)} → {_short_label(dst_name)}")

    sorted_idx = np.argsort(attn_weights)[::-1][:top_k]
    top_labels  = [edge_labels[i] for i in sorted_idx]
    top_weights = [attn_weights[i] for i in sorted_idx]

    colors = [EXPLOIT_EDGE_COLOR if w > np.median(attn_weights) * 2 else "#4a90d9"
              for w in top_weights]
    ax.barh(top_labels[::-1], top_weights[::-1], color=colors[::-1])
    ax.set_xlabel("Attention Weight", color="white")
    ax.set_title(f"Top-{top_k} Edge Attention Weights", color="white", fontsize=13)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  Saved attention heatmap → {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# High-level convenience: draw from session_logs + model
# ---------------------------------------------------------------------------

def visualize_session(
    session_logs: dict[str, list[dict]],
    model_path: Optional[str] = None,
    label: int = 0,
    session_id: str = "",
    save_dir: Optional[str] = None,
    layout: str = "spring",
) -> None:
    """
    End-to-end: build a graph, optionally run GAT, then draw.

    Parameters
    ----------
    session_logs : dict
    model_path : str | None – if provided, GAT attention is overlaid
    label : int
    session_id : str
    save_dir : str | None – directory for saved PNGs; None → plt.show()
    layout : str
    """
    graph = build_graph_from_session(session_logs, label=label)
    node_ids   = graph.node_ids
    edge_index = graph.edge_index.numpy()
    attn_weights = None

    if model_path and Path(model_path).exists():
        device = torch.device("cpu")
        model  = BrowserGAT.load(
            model_path, device=device,
            in_channels=graph.x.size(1),
            hidden_channels=64, num_heads=4, num_classes=2, dropout=0.0,
        )
        batch = torch.zeros(graph.num_nodes, dtype=torch.long)
        with torch.no_grad():
            _, aw, aei = model(
                graph.x, graph.edge_index, batch, return_attention=True
            )
        attn_weights = aw.squeeze(-1).numpy()
        edge_index   = aei.numpy()   # use attention edge_index (same as graph's)

    # Graph drawing
    graph_save  = str(Path(save_dir) / f"{session_id}_graph.png")  if save_dir else None
    attn_save   = str(Path(save_dir) / f"{session_id}_attn.png")   if save_dir else None

    draw_session_graph(
        node_ids=node_ids,
        edge_index=edge_index,
        attn_weights=attn_weights,
        label=label,
        session_id=session_id,
        save_path=graph_save,
        layout=layout,
    )
    if attn_weights is not None and len(attn_weights) > 0:
        draw_attention_heatmap(
            node_ids=node_ids,
            edge_index=edge_index,
            attn_weights=attn_weights,
            save_path=attn_save,
        )


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Visualise a browser session graph")
    parser.add_argument("--session", required=True, help="session_id to visualize")
    parser.add_argument("--root",    default="data", help="Dataset root directory")
    parser.add_argument("--model",   default=None,   help="Path to trained model .pt")
    parser.add_argument("--layout",  default="spring",
                        choices=["spring", "kamada_kawai", "circular"])
    parser.add_argument("--save",    default=None,   help="Directory to save PNGs")
    args = parser.parse_args()

    from pyg_dataset import BrowserSessionDataset, _load_json_log, _LOG_FILES

    raw_dir = Path(args.root) / "raw"
    all_records: dict[str, list[dict]] = {}
    for log_type, filename in _LOG_FILES.items():
        all_records[log_type] = _load_json_log(raw_dir / filename)

    session_logs: dict[str, list[dict]] = {lt: [] for lt in all_records}
    for lt, records in all_records.items():
        for r in records:
            if r.get("session_id") == args.session:
                session_logs[lt].append(r)

    if args.save:
        Path(args.save).mkdir(parents=True, exist_ok=True)

    visualize_session(
        session_logs=session_logs,
        model_path=args.model,
        label=0,
        session_id=args.session,
        save_dir=args.save,
        layout=args.layout,
    )
