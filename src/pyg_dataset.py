"""
pyg_dataset.py
==============
PyTorch Geometric ``InMemoryDataset`` that:

1. Loads all raw JSON log files from ``data/raw/``.
2. Groups records by ``session_id``.
3. Derives a binary label per session from heuristics (or from a
   pre-existing label column if present).
4. Calls :func:`graph_builder.build_graph_from_session` for each session.
5. Caches the processed graphs to ``data/processed/``.

Usage
-----
>>> from pyg_dataset import BrowserSessionDataset
>>> ds = BrowserSessionDataset(root="data")
>>> print(ds[0])
Data(x=[42, 32], edge_index=[2, 67], y=0)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable

import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset

from graph_builder import build_graph_from_session

# ---------------------------------------------------------------------------
# Heuristic labelling
# ---------------------------------------------------------------------------

def _heuristic_label(session_logs: dict[str, list[dict]]) -> int:
    """
    Assign a binary label (0=normal, 1=malicious) when no ground-truth is
    available.  Uses rule-based thresholds derived from the log features.

    This is a *research heuristic* — replace with real labels when available.
    """
    ses = session_logs.get("session", [{}])[0]
    scr = session_logs.get("script", [])
    obf = session_logs.get("obfuscation", [])
    js  = session_logs.get("js", [])

    score = 0

    # High eval-to-function ratio suggests heavy dynamic code execution
    if float(ses.get("eval_to_function_ratio", 0)) > 0.5:
        score += 2

    # High dynamic script ratio
    if float(ses.get("dynamic_script_ratio", 0)) > 0.6:
        score += 2

    # Many iframes — common in drive-by downloads
    if float(ses.get("total_iframes", 0)) >= 3:
        score += 1

    # High entropy script → possible obfuscation
    for r in scr:
        if float(r.get("entropy_score", 0)) > 5.5:
            score += 1
            break

    # atob usage → base64 decode → common in exploits
    for r in obf:
        if float(r.get("atob_usage", 0)) > 0:
            score += 2
        if float(r.get("eval_decoder_patterns", 0)) > 0:
            score += 2

    # Excessive eval call count
    for r in js:
        if float(r.get("call_count", 0)) > 100 and r.get("api_name") == "eval":
            score += 2

    return 1 if score >= 4 else 0


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

_LOG_FILES = {
    "js":           "js_behavior_logs_output.json",
    "dom":          "dom_logs_output.json",
    "network":      "network_logs_output.json",
    "navigation":   "navigation_logs_output.json",
    "script":       "script_logs_output.json",
    "obfuscation":  "obfuscation_logs_output.json",
    "token":        "token_logs_output.json",
    "session":      "session_logs_output.json",
}

# The log sample provided uses these actual filenames:
_LOG_FILES_ALT = {
    "js":           "js_behaviour_logs_output.json",   # British spelling variant
}


def _load_json_log(path: Path) -> list[dict]:
    """Load a newline-delimited JSON or JSON-array file."""
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    # Try JSON array first
    if text.startswith("["):
        return json.loads(text)
    # Otherwise newline-delimited JSON
    records = []
    for line in text.splitlines():
        line = line.strip().rstrip(",")
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


class BrowserSessionDataset(InMemoryDataset):
    """
    One graph per browser session derived from the 8 raw log files.

    Parameters
    ----------
    root : str
        Root data directory.  Should contain ``raw/`` with the JSON files.
    transform : Callable, optional
        Standard PyG transform applied at access time.
    pre_transform : Callable, optional
        Standard PyG transform applied before caching.
    label_map : dict[str, int] | None
        Optional override mapping ``session_id → label`` for supervised
        training.  If ``None``, heuristic labelling is used.
    """

    def __init__(
        self,
        root: str = "data",
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        label_map: dict[str, int] | None = None,
    ) -> None:
        self._label_map = label_map or {}
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(
            self.processed_paths[0], weights_only=False
        )

    @property
    def raw_file_names(self) -> list[str]:
        return list(_LOG_FILES.values())

    @property
    def processed_file_names(self) -> list[str]:
        return ["browser_session_graphs.pt"]

    def download(self) -> None:
        # Raw files are expected to be placed manually in data/raw/
        pass

    # ---------------------------------------------------------------------- #
    # Core processing                                                          #
    # ---------------------------------------------------------------------- #

    def process(self) -> None:
        raw_dir = Path(self.raw_dir)

        # ---- Load each log type ------------------------------------------ #
        all_records: dict[str, list[dict]] = {}
        for log_type, filename in _LOG_FILES.items():
            path = raw_dir / filename
            # Try the alt spelling as a fallback
            if not path.exists() and log_type in _LOG_FILES_ALT:
                path = raw_dir / _LOG_FILES_ALT[log_type]
            all_records[log_type] = _load_json_log(path)
            print(f"  Loaded {len(all_records[log_type]):>6} records  [{log_type}]")

        # ---- Gather all session IDs -------------------------------------- #
        session_ids: set[str] = set()
        for records in all_records.values():
            for r in records:
                sid = r.get("session_id")
                if sid:
                    session_ids.add(sid)

        print(f"  Total unique sessions: {len(session_ids)}")

        # ---- Group by session_id ----------------------------------------- #
        grouped: dict[str, dict[str, list[dict]]] = {
            sid: {lt: [] for lt in all_records} for sid in session_ids
        }
        for log_type, records in all_records.items():
            for r in records:
                sid = r.get("session_id")
                if sid in grouped:
                    grouped[sid][log_type].append(r)

        # ---- Build graphs ------------------------------------------------ #
        data_list: list[Data] = []
        for sid in sorted(session_ids):
            session_logs = grouped[sid]
            label = (
                self._label_map[sid]
                if sid in self._label_map
                else _heuristic_label(session_logs)
            )
            graph = build_graph_from_session(session_logs, label=label)
            graph.session_id = sid          # store for debugging
            if self.pre_transform is not None:
                graph = self.pre_transform(graph)
            data_list.append(graph)

        malicious = sum(1 for g in data_list if g.y.item() == 1)
        print(
            f"  Graph stats: {len(data_list)} graphs | "
            f"{malicious} malicious | {len(data_list) - malicious} normal"
        )

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print(f"  Saved processed dataset → {self.processed_paths[0]}")
