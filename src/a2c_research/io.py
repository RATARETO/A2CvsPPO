"""Small, atomic result files. No secrets or machine environment variables are recorded."""

from __future__ import annotations

import ast
import csv
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
                         encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def source_hash() -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def training_hash() -> str:
    """Hash training/config/evaluation ASTs; retain full source provenance separately."""
    digest = hashlib.sha256()
    for name in ("config.py", "envs.py", "evaluation.py", "train.py"):
        path = Path(__file__).with_name(name)
        digest.update(name.encode())
        digest.update(ast.dump(ast.parse(path.read_text(encoding="utf-8"))).encode())
    return digest.hexdigest()


def training_identity(metadata: dict) -> str:
    runtime = metadata.get("runtime", {})
    old_hash = runtime.get("source_sha256")
    manifest = Path(__file__).with_name("compatibility.json")
    mapping = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {}
    identity = (runtime.get("training_sha256") or mapping.get(
        "legacy_source_to_training", {}).get(old_hash, f"unverified:{old_hash}"))
    protocol = metadata.get("config", {}).get("protocol", "cartpole-stability-v1")
    return mapping.get("training_aliases", {}).get(protocol, {}).get(identity, identity)


def runtime_metadata() -> dict:
    return {
        "versions": {"python": platform.python_version(), **{
            name: importlib.metadata.version(name)
            for name in ("stable-baselines3", "gymnasium", "torch", "numpy", "pandas", "matplotlib")
        }},
        "platform": {"system": platform.system(), "machine": platform.machine(),
                     "processor": platform.processor()},
        "source_sha256": source_hash(),
        "training_sha256": training_hash(),
        "device": "cpu", "torch_threads": 1,
    }


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
