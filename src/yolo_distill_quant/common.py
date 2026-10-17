"""Helpers shared by the yolo_distill_quant command modules."""
import argparse
import copy
import csv
import json
import os
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path

import yaml


class InputError(Exception):
    """Bad user input (missing file, malformed candidate, ...)."""


def _load_mapping(path):
    path = Path(path)
    if not path.is_file():
        raise InputError(f"file not found: {path}")
    with open(path, "r") as f:
        if path.suffix.lower() == ".json":
            return json.load(f)
        return yaml.safe_load(f)


def load_base_config(path):
    cfg = _load_mapping(path)
    if not isinstance(cfg, dict):
        raise InputError(f"{path}: expected a YAML mapping at the top level")
    if "scales" not in cfg:
        raise InputError(f"{path}: no 'scales' block; this tool needs an Ultralytics model YAML with one")
    return cfg


def load_candidates(path):
    """Load {name: [depth, width, max_channels]} from a YAML or JSON file."""
    data = _load_mapping(path)
    if not isinstance(data, dict) or not data:
        raise InputError(f"{path}: expected a non-empty mapping of name -> [depth, width, max_channels]")
    candidates = {}
    for name, scales in data.items():
        if (
            not isinstance(scales, (list, tuple))
            or len(scales) != 3
            or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in scales)
        ):
            raise InputError(f"candidate '{name}': expected 3 numbers [depth, width, max_channels], got {scales!r}")
        if any(v <= 0 for v in scales):
            raise InputError(f"candidate '{name}': values must be > 0, got {scales!r}")
        candidates[str(name)] = list(scales)
    return candidates


def str2bool(value):
    """argparse type for true/false flags: accepts true/false, yes/no, 1/0 (any case)."""
    v = str(value).strip().lower()
    if v in {"true", "yes", "1", "y", "t"}:
        return True
    if v in {"false", "no", "0", "n", "f"}:
        return False
    raise argparse.ArgumentTypeError(f"expected true or false, got {value!r}")


def slugify(name):
    """
    turns any arbitrary candidate name into a filesystem-safe, clean string (a"slug").
    """
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


@contextmanager
def temp_model_yaml(base_cfg, scales, name):
    """Write a model YAML forcing a single 'custom' scale; delete it on exit."""
    cfg = copy.deepcopy(base_cfg)
    cfg["scales"] = {"custom": list(scales)}
    fd, path = tempfile.mkstemp(prefix=f"scale_{slugify(name)}_", suffix=".yaml")
    try:
        with os.fdopen(fd, "w") as f:
            yaml.dump(cfg, f)
        yield path
    finally:
        if os.path.exists(path):
            os.remove(path)


def print_table(rows, columns):
    """columns: list of (header, key, width, formatter-or-None). Failed rows print as-is."""
    divider = "-" * (sum(w for _, _, w, _ in columns) + 3 * (len(columns) - 1))
    print(divider)
    print(" | ".join(f"{h:<{w}}" for h, _, w, _ in columns))
    print(divider)
    for r in rows:
        cells = []
        for _, key, w, fmt in columns:
            v = r.get(key, "")
            if fmt and not isinstance(v, str):
                try:
                    v = fmt(v)
                except (TypeError, ValueError):
                    pass
            cells.append(f"{str(v):<{w}}")
        print(" | ".join(cells))
    print(divider)


def write_results(rows, json_out=None, csv_out=None):
    if json_out:
        with open(json_out, "w") as f:
            json.dump(rows, f, indent=2)
    if csv_out and rows:
        with open(csv_out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
