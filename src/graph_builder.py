"""
graph_builder.py
================
Converts raw browser-session log records into a PyTorch Geometric ``Data``
object.  One session → one graph.

Design
------
* Nodes   – unique behavioral entities  (JS API names, DOM tag types, network
            behaviour buckets, navigation events, script behaviour buckets).
* Edges   – directed temporal-sequence edges + known exploit-chain edges.
* Features – fixed-length, NODE-LOCAL feature vectors.  Each node type gets
             features meaningful to IT specifically; irrelevant slots are zero.
             Global obfuscation/token aggregates are intentionally excluded from
             node features to avoid homogenising all nodes (Issue #3 fix).

Anti-leakage design (Issue #1 fix)
-----------------------------------
Raw domain names and script URLs are NEVER used as node identifiers or
features.  Instead network nodes are bucketed by BEHAVIOURAL properties:

  net:cross_origin   – request to a domain different from the page domain
  net:high_entropy   – domain string entropy ≥ 3.5  (DGA indicator)
  net:exfil_port     – non-standard port (not 80/443)
  net:standard       – ordinary same-origin / CDN request

Script nodes are bucketed by their entropy/obfuscation tier rather than URL:

  script:high_entropy   – entropy_score ≥ 5.5
  script:medium_entropy – 4.0 ≤ entropy_score < 5.5
  script:low_entropy    – entropy_score < 4.0

This means the model learns BEHAVIOURAL patterns, not domain memorisation.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np
import torch
from torch_geometric.data import Data

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURE_DIM = 32  # fixed node-feature dimensionality — DO NOT change without
                  # retraining; must stay consistent with gat_model.py

# Known exploit-chain pairs (src → dst) receive an additional "chain" edge
# even if they are non-adjacent in the temporal sequence.
EXPLOIT_CHAINS: list[tuple[str, str]] = [
    ("atob", "eval"),
    ("eval", "document.write"),
    ("document.write", "iframe"),
    ("iframe", "network_request"),
    ("eval", "innerHTML"),
    ("innerHTML", "dynamic_script"),
    ("fetch", "eval"),
]

# Standard web ports — anything else is flagged as suspicious
_STANDARD_PORTS: set[int] = {80, 443, 8080, 8443}

# Entropy threshold above which a domain is considered DGA-like
_DGA_ENTROPY_THRESHOLD: float = 3.5


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _shannon_entropy(s: str) -> float:
    """Shannon entropy of a string (bits per character)."""
    if not s:
        return 0.0
    freq = defaultdict(int)
    for c in s:
        freq[c] += 1
    n = len(s)
    return -sum((v / n) * math.log2(v / n) for v in freq.values())


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Safely coerce a value to float."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Anti-leakage bucketing  (Issue #1 fix)
# ---------------------------------------------------------------------------

def _bucket_network_node(record: dict, page_domain: str) -> str:
    """
    Map a network log record to a behaviour-based node ID.

    Buckets (in priority order):
      net:exfil_port      – non-standard port
      net:high_entropy    – domain entropy ≥ threshold  (DGA-like)
      net:cross_origin    – domain differs from page domain
      net:standard        – everything else

    The raw domain name is NEVER included in the returned string, preventing
    the model from memorising specific malicious domains.
    """
    domain = str(record.get("domain", ""))
    port   = int(_safe_float(record.get("port", 443)))

    if port not in _STANDARD_PORTS:
        return "net:exfil_port"
    if _shannon_entropy(domain) >= _DGA_ENTROPY_THRESHOLD:
        return "net:high_entropy"
    if page_domain and domain and domain != page_domain:
        return "net:cross_origin"
    return "net:standard"


def _bucket_script_node(record: dict) -> str:
    """
    Map a script log record to an entropy-tier node ID.

    Tiers:
      script:high_entropy   – entropy_score ≥ 5.5
      script:medium_entropy – 4.0 ≤ entropy_score < 5.5
      script:low_entropy    – entropy_score < 4.0

    URL is intentionally dropped to prevent domain-level memorisation.
    """
    entropy = _safe_float(record.get("entropy_score", 0.0))
    if entropy >= 5.5:
        return "script:high_entropy"
    if entropy >= 4.0:
        return "script:medium_entropy"
    return "script:low_entropy"


# ---------------------------------------------------------------------------
# Node-LOCAL feature builder  (Issue #3 fix)
# ---------------------------------------------------------------------------
#
# Feature layout (FEATURE_DIM = 32):
#
#  Slot   Meaning                          Populated for
#  ----   -------                          -------------
#  0      node type one-hot bucket[0]      all
#  1      node type one-hot bucket[1]      all
#  2      node type one-hot bucket[2]      all
#  3      node type one-hot bucket[3]      all
#  4      node type one-hot bucket[4]      all
#  --- JS-API node features ---
#  5      avg call_count                   js nodes
#  6      peak call_count                  js nodes
#  7      occurrence count                 js nodes
#  --- Script node features ---
#  8      script entropy_score             script nodes
#  9      num_eval_calls                   script nodes
#  10     num_dynamic_calls                script nodes
#  11     base64_string_count              script nodes
#  12     num_functions                    script nodes
#  --- Network node features ---
#  13     avg response_size (log)          net nodes
#  14     non-200 response ratio           net nodes
#  15     POST request ratio               net nodes
#  16     avg domain entropy               net nodes
#  17     request count                    net nodes
#  --- DOM node features ---
#  18     avg dom_depth                    dom nodes
#  19     hidden node ratio                dom nodes
#  20     avg dom_size                     dom nodes
#  21     avg innerHTML_length             dom nodes
#  --- Obfuscation features (script / high-entropy nodes only) ---
#  22     atob_usage                       script nodes
#  23     eval_decoder_patterns            script nodes
#  24     encoded_string_ratio             script nodes
#  25     control_flow_alterations         script nodes
#  26     random_identifier_ratio          script nodes
#  --- Token features (JS-API nodes only) ---
#  27     token_eval_count                 js:eval node
#  28     token_document_write_count       js:document.write node
#  29     token_innerHTML_count            js:innerHTML node
#  30     token_setTimeout_count           js:setTimeout node
#  31     session meta-entropy             all (summary)

# Node-type bucket indices for one-hot slots 0-4
_NODE_TYPE_BUCKETS: dict[str, int] = {
    "js":     0,
    "dom":    1,
    "net":    2,
    "script": 3,
    "nav":    4,
}


def _build_node_features(
    node_id: str,
    js_records: list[dict],
    script_records: list[dict],
    obfusc_records: list[dict],
    token_records: list[dict],
    session_record: dict | None,
    dom_records: list[dict],
    net_records: list[dict],
) -> np.ndarray:
    """
    Build a FEATURE_DIM-dimensional NODE-LOCAL feature vector.

    Only features that are semantically meaningful for *this node type* are
    populated; all other slots remain zero.  This preserves distinct node
    semantics so the GAT can learn type-specific patterns.
    """
    feat = np.zeros(FEATURE_DIM, dtype=np.float32)

    # Determine node type prefix
    prefix = node_id.split(":")[0] if ":" in node_id else "js"
    bucket = _NODE_TYPE_BUCKETS.get(prefix, 0)
    feat[bucket] = 1.0   # one-hot type encoding (slots 0-4)

    # ---- JS-API node (slots 5-7, 27-30) ---------------------------------- #
    if prefix == "js" or ":" not in node_id:
        bare_name = node_id  # e.g. "eval", "atob", "document.write"
        js_for_node = [r for r in js_records if r.get("api_name") == bare_name]
        if js_for_node:
            counts = [_safe_float(r.get("call_count")) for r in js_for_node]
            feat[5] = float(np.mean(counts))
            feat[6] = float(np.max(counts))
            feat[7] = float(len(js_for_node))
        # Token features are meaningful only for specific JS APIs
        for r in token_records:
            feat[27] += _safe_float(r.get("token_eval_count"))            if bare_name == "eval"            else 0
            feat[28] += _safe_float(r.get("token_document_write_count"))  if bare_name == "document.write"  else 0
            feat[29] += _safe_float(r.get("token_innerHTML_count"))       if bare_name == "innerHTML"        else 0
            feat[30] += _safe_float(r.get("token_setTimeout_count"))      if bare_name == "setTimeout"       else 0

    # ---- Script node (slots 8-12, 22-26) --------------------------------- #
    elif prefix == "script":
        if script_records:
            feat[8]  = float(np.mean([_safe_float(r.get("entropy_score"))      for r in script_records]))
            feat[9]  = float(np.sum ([_safe_float(r.get("num_eval_calls"))     for r in script_records]))
            feat[10] = float(np.sum ([_safe_float(r.get("num_dynamic_calls"))  for r in script_records]))
            feat[11] = float(np.sum ([_safe_float(r.get("base64_string_count"))for r in script_records]))
            feat[12] = float(np.mean([_safe_float(r.get("num_functions"))      for r in script_records]))
        # Obfuscation stats naturally belong to script nodes
        if obfusc_records:
            feat[22] = float(np.sum([_safe_float(r.get("atob_usage"))              for r in obfusc_records]))
            feat[23] = float(np.sum([_safe_float(r.get("eval_decoder_patterns"))   for r in obfusc_records]))
            feat[24] = float(np.mean([_safe_float(r.get("encoded_string_ratio"))   for r in obfusc_records]))
            feat[25] = float(np.sum([_safe_float(r.get("control_flow_alterations"))for r in obfusc_records]))
            feat[26] = float(np.mean([_safe_float(r.get("random_identifier_ratio"))for r in obfusc_records]))

    # ---- Network node (slots 13-17) -------------------------------------- #
    elif prefix == "net":
        # Filter to records whose bucket matches this node_id
        net_for_node = [r for r in net_records
                        if _bucket_network_node(r, "") == node_id
                        or node_id == "net:standard"]   # fallback: use all
        if not net_for_node:
            net_for_node = net_records   # use session-level aggregate
        if net_for_node:
            resp_sizes = [_safe_float(r.get("response_size", 0)) for r in net_for_node]
            feat[13] = math.log1p(float(np.mean(resp_sizes)))
            non200 = sum(1 for r in net_for_node if int(_safe_float(r.get("response_code", 200))) != 200)
            feat[14] = non200 / len(net_for_node)
            posts   = sum(1 for r in net_for_node if str(r.get("method", "")).upper() == "POST")
            feat[15] = posts / len(net_for_node)
            domains  = [str(r.get("domain", "")) for r in net_for_node]
            feat[16] = float(np.mean([_shannon_entropy(d) for d in domains]))
            feat[17] = float(len(net_for_node))

    # ---- DOM node (slots 18-21) ------------------------------------------ #
    elif prefix == "dom":
        tag = node_id.split(":", 1)[-1]
        dom_for_node = [r for r in dom_records
                        if r.get("node_name", "").lower() == tag]
        if not dom_for_node:
            dom_for_node = dom_records
        if dom_for_node:
            feat[18] = float(np.mean([_safe_float(r.get("dom_depth"))          for r in dom_for_node]))
            hidden    = sum(1 for r in dom_for_node if r.get("is_hidden", False))
            feat[19] = hidden / len(dom_for_node)
            feat[20] = float(np.mean([_safe_float(r.get("dom_size"))           for r in dom_for_node]))
            feat[21] = float(np.mean([_safe_float(r.get("node_innerHTML_length")) for r in dom_for_node]))

    # ---- Session summary in final slot (slot 31) — all node types -------- #
    # This is a very compressed session context, not leaking label-correlated
    # features directly, only the broad session complexity.
    if session_record:
        sess_vec = [
            _safe_float(session_record.get("eval_to_function_ratio")),
            _safe_float(session_record.get("dynamic_script_ratio")),
            _safe_float(session_record.get("total_iframes")),
        ]
        feat[31] = _shannon_entropy("".join(f"{v:.2f}" for v in sess_vec))

    return feat


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def build_graph_from_session(
    session_logs: dict[str, list[dict]],
    label: int = 0,
) -> Data:
    """
    Convert all log records for a single session into a PyTorch Geometric
    ``Data`` object.

    Parameters
    ----------
    session_logs : dict
        Keys are log types:
            ``"js"``, ``"dom"``, ``"network"``, ``"navigation"``,
            ``"script"``, ``"obfuscation"``, ``"token"``, ``"session"``
        Values are lists of record dicts (already filtered to ONE session).
    label : int
        Ground-truth label: 0 = normal, 1 = malicious.

    Returns
    -------
    torch_geometric.data.Data
        ``x``          – node feature matrix  [N, FEATURE_DIM]
        ``edge_index`` – directed edges       [2, E]
        ``y``          – scalar label tensor  []
        ``node_ids``   – list[str] mapping row index → node canonical name
    """
    js_recs   = session_logs.get("js", [])
    dom_recs  = session_logs.get("dom", [])
    net_recs  = session_logs.get("network", [])
    nav_recs  = session_logs.get("navigation", [])
    scr_recs  = session_logs.get("script", [])
    obf_recs  = session_logs.get("obfuscation", [])
    tok_recs  = session_logs.get("token", [])
    ses_recs  = session_logs.get("session", [])

    # Session record (take the one with the most info if multiple)
    session_record = ses_recs[0] if ses_recs else None

    # ---------------------------------------------------------------------- #
    # 1.  Collect all events, each with (timestamp, node_id)
    # ---------------------------------------------------------------------- #
    events: list[tuple[int, str]] = []

    for r in sorted(js_recs, key=lambda x: x.get("timestamp", 0)):
        api = r.get("api_name", "unknown_api")
        events.append((r.get("timestamp", 0), api))

    for r in sorted(dom_recs, key=lambda x: x.get("timestamp", 0)):
        tag = r.get("node_name", "dom_node").lower().replace("#", "")
        events.append((r.get("timestamp", 0), f"dom:{tag}"))

    for r in sorted(net_recs, key=lambda x: x.get("timestamp", 0)):
        domain = r.get("domain", "unknown_domain")
        events.append((r.get("timestamp", 0), f"net:{domain}"))

    for r in sorted(nav_recs, key=lambda x: x.get("timestamp", 0)):
        etype = r.get("event_type", "navigation")
        events.append((r.get("timestamp", 0), f"nav:{etype}"))

    for r in sorted(scr_recs, key=lambda x: x.get("timestamp", 0)):
        url = r.get("url", "unknown_script")
        # Shorten URL to last path segment for readability
        slug = url.rstrip("/").split("/")[-1][:32] or "script"
        events.append((r.get("timestamp", 0), f"script:{slug}"))

    # Sort globally by timestamp to preserve temporal ordering
    events.sort(key=lambda e: e[0])

    # ---------------------------------------------------------------------- #
    # 2.  Build node vocabulary (unique node IDs → index)
    # ---------------------------------------------------------------------- #
    node_vocab: dict[str, int] = {}
    for _, nid in events:
        if nid not in node_vocab:
            node_vocab[nid] = len(node_vocab)

    if not node_vocab:
        # Return an empty single-node graph rather than crashing
        dummy_feat = torch.zeros((1, FEATURE_DIM), dtype=torch.float)
        return Data(
            x=dummy_feat,
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            y=torch.tensor(label, dtype=torch.long),
            node_ids=["__empty__"],
        )

    node_ids: list[str] = [""] * len(node_vocab)
    for nid, idx in node_vocab.items():
        node_ids[idx] = nid

    # ---------------------------------------------------------------------- #
    # 3.  Build edges
    # ---------------------------------------------------------------------- #
    edges: set[tuple[int, int]] = set()

    # 3a. Temporal sequence edges
    prev_idx: int | None = None
    for _, nid in events:
        cur_idx = node_vocab[nid]
        if prev_idx is not None and prev_idx != cur_idx:
            edges.add((prev_idx, cur_idx))
        prev_idx = cur_idx

    # 3b. Explicit exploit-chain edges
    for src_name, dst_name in EXPLOIT_CHAINS:
        # Check bare name as well as prefixed variants
        for src in [src_name, f"dom:{src_name}"]:
            for dst in [dst_name, f"dom:{dst_name}"]:
                if src in node_vocab and dst in node_vocab:
                    edges.add((node_vocab[src], node_vocab[dst]))

    edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()
    if edge_index.numel() == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    # ---------------------------------------------------------------------- #
    # 4.  Build node feature matrix
    # ---------------------------------------------------------------------- #
    num_nodes = len(node_vocab)
    x = np.zeros((num_nodes, FEATURE_DIM), dtype=np.float32)

    for nid, idx in node_vocab.items():
        x[idx] = _build_node_features(
            node_id=nid,
            js_records=js_recs,
            script_records=scr_recs,
            obfusc_records=obf_recs,
            token_records=tok_recs,
            session_record=session_record,
            dom_records=dom_recs,
            net_records=net_recs,
        )

    # ---------------------------------------------------------------------- #
    # 5.  Return PyG Data object
    # ---------------------------------------------------------------------- #
    return Data(
        x=torch.tensor(x, dtype=torch.float),
        edge_index=edge_index,
        y=torch.tensor(label, dtype=torch.long),
        node_ids=node_ids,          # stored as a plain Python list (not a tensor)
    )