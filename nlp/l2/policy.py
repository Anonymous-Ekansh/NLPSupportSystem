"""Loader / writer for the single Level-2 policy file.

One file, `config/l2_policy.json`, holds every tunable knob used by A3 (and
later A4/A5): confidence bands, top-2 margin, OOD similarity limit, retrieval
similarity and neighbour agreement minima, sensitive-term lists and the
intent -> required-slot map.

Rules
-----
* Read at call time (never cached) so a tuning script can overwrite the file
  and the next request picks it up without a restart.
* Missing keys fall back to `DEFAULT_POLICY`; the file on disk is the source of
  truth, so a partial file is a valid override.
* `L2_POLICY_PATH` env var overrides the path (useful for ablations).
* Writes are atomic (tmp file + rename) so a crash mid-write cannot leave an
  unreadable policy behind.
"""

import copy
import json
import os
import tempfile
from typing import Any, Optional

POLICY_PATH = "config/l2_policy.json"
POLICY_ENV = "L2_POLICY_PATH"

# Minimal fallback: intentionally small. config/l2_policy.json carries the
# comments, provenance and tuning log.
DEFAULT_POLICY: dict[str, Any] = {
    "version": 1,
    "source": "defaults",
    "thresholds": {
        "confidence_high": 0.75,
        "confidence_low": 0.40,
        "top2_margin_min": 0.20,
        "ood_similarity_min": 0.55,
        "similarity_min": 0.70,
        "neighbor_agreement_min": 0.60,
    },
    "retrieval": {"mode": "hybrid", "k": 5, "k_widened": 10},
    "routing": {
        "clarify_on_missing_slot": True,
        "answer_when_top2_same_category": True,
        "gibberish_min_chars": 3,
        "gibberish_max_token_chars": 25,
    },
    "escalation": {"sensitive_terms": {}},
    "slots": {
        "required": {},
        "slot_labels": {},
        "questions": {},
        "not_have_phrases": [],
        "derived_from_train": {},
    },
    "tuning": {},
}


def policy_path(path: Optional[str] = None) -> str:
    """Resolve the policy path: explicit arg > env var > default."""
    return path or os.environ.get(POLICY_ENV) or POLICY_PATH


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into a copy of *base* (dicts only)."""
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_policy(path: Optional[str] = None) -> dict:
    """Return the merged policy. Missing file → DEFAULT_POLICY copy."""
    p = policy_path(path)
    if not os.path.exists(p):
        return copy.deepcopy(DEFAULT_POLICY)
    with open(p) as f:
        loaded = json.load(f)
    return _deep_merge(DEFAULT_POLICY, loaded)


def save_policy(policy: dict, path: Optional[str] = None) -> str:
    """Atomically write *policy* to disk and return the path used."""
    p = policy_path(path)
    parent = os.path.dirname(p)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=parent or ".")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(policy, f, indent=2)
            f.write("\n")
        os.replace(tmp, p)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return p


def get(policy: dict, dotted: str, default: Any = None) -> Any:
    """Read a nested key: `get(policy, "thresholds.confidence_low")`."""
    node: Any = policy
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


__all__ = ["POLICY_PATH", "POLICY_ENV", "DEFAULT_POLICY",
           "policy_path", "load_policy", "save_policy", "get"]
