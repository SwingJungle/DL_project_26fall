"""Project configuration and credential loading.

Credential resolution order (first hit wins, never logged):
1. environment variables ``DSL_BASE_URL`` / ``DSL_API_KEY``
2. standard gateway names ``URL_OPENAI`` / ``API`` in the environment
3. ``Qwen.txt`` in the repo root (current convention)
4. ``.env`` in the repo root

The repository is not a git repo today, but ``Qwen.txt`` and ``.env`` are
listed in ``.gitignore`` so the plaintext key cannot be committed by accident.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"

_REQUIRED_MODEL_FIELDS = ("provider", "family", "tier", "pair_code", "api_model")


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


@lru_cache(maxsize=1)
def models_config() -> dict:
    return _read_json(CONFIG_DIR / "models.json")


@lru_cache(maxsize=1)
def budget_config() -> dict:
    return _read_json(CONFIG_DIR / "budget.json")


@lru_cache(maxsize=1)
def schema_config() -> dict:
    return _read_json(CONFIG_DIR / "schema.json")


def model_registry() -> dict[str, dict]:
    reg = models_config()["models"]
    for key, spec in reg.items():
        missing = [f for f in _REQUIRED_MODEL_FIELDS if f not in spec]
        if missing:
            raise ValueError(f"model {key!r} missing fields: {missing}")
    return reg


def model_spec(model_key: str) -> dict:
    reg = model_registry()
    if model_key not in reg:
        raise KeyError(f"unknown model_key {model_key!r}; known: {sorted(reg)}")
    return reg[model_key]


def pair_mapping() -> dict[str, str]:
    """pair_code (As/Al/Bs/Bl) -> model_key. Frozen by D-1."""
    return dict(models_config()["pairs"])


def all_model_keys() -> list[str]:
    return list(model_registry())


def _parse_env_file(path: Path) -> dict[str, str]:
    vals: dict[str, str] = {}
    if not path.exists():
        return vals
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        vals[key.strip()] = value.strip().strip('"').strip("'")
    return vals


@lru_cache(maxsize=1)
def credentials() -> tuple[str, str]:
    """Return (base_url, api_key). Raises with a redacted message if missing."""
    env = dict(os.environ)
    base = env.get("DSL_BASE_URL") or env.get("URL_OPENAI")
    key = env.get("DSL_API_KEY") or env.get("API")

    if not (base and key):
        cfg = models_config()["gateway"]
        for name in cfg.get("credential_files", []):
            vals = _parse_env_file(ROOT / name)
            base = base or vals.get("URL_OPENAI") or vals.get("DSL_BASE_URL")
            key = key or vals.get("API") or vals.get("DSL_API_KEY")
            if base and key:
                break

    if not base or not key:
        raise RuntimeError(
            "API credentials not found. Set DSL_BASE_URL/DSL_API_KEY or add Qwen.txt "
            "with URL_OPENAI=... and API=... (see .env.example)."
        )
    return base.rstrip("/"), key


def run_dir(run_id: str) -> Path:
    path = ROOT / "runs" / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def cache_path() -> Path:
    path = ROOT / "runs" / "_cache.sqlite3"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def results_path(name: str) -> Path:
    path = ROOT / "results" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
