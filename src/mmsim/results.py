"""Reading and writing the JSON results the README and website quote."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"


def _jsonable(obj):
    """NumPy scalars/arrays -> Python; non-finite floats -> null (valid JSON for JavaScript)."""
    if isinstance(obj, dict):
        return {str(key): _jsonable(val) for key, val in obj.items()}
    if isinstance(obj, (list, tuple, np.ndarray)):
        return [_jsonable(val) for val in obj]
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def write_json(relpath: str, obj) -> Path:
    path = RESULTS / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(obj), indent=2, allow_nan=False), encoding="utf-8")
    return path


def read_json(relpath: str):
    return json.loads((RESULTS / relpath).read_text(encoding="utf-8"))


def update_summary(section: str, values: dict) -> None:
    """Replace one section of results/summary.json, leaving the others intact."""
    path = RESULTS / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    summary[section] = values
    write_json("summary.json", summary)
