"""Small, atomic result files. No secrets or machine environment variables are recorded."""

from __future__ import annotations

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


def runtime_metadata() -> dict:
    return {
        "versions": {"python": platform.python_version(), **{
            name: importlib.metadata.version(name)
            for name in ("stable-baselines3", "gymnasium", "torch", "numpy", "pandas", "matplotlib")
        }},
        "platform": {"system": platform.system(), "machine": platform.machine(),
                     "processor": platform.processor()},
        "source_sha256": source_hash(),
        "device": "cpu", "torch_threads": 1,
    }


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
