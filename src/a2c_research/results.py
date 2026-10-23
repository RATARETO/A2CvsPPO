"""Read new and colleague-format results without altering either source directory."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .config import PROTOCOL, ExperimentConfig, canonical_json
from .io import file_hash
from .metrics import curve_metrics


@dataclass
class Run:
    path: Path
    config: dict
    metadata: dict
    summary: dict
    curve: list[dict]
    metrics: dict
    protocol: str
    stratum: str


def read_curve(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return [{"timesteps": int(row["timesteps"]),
             "eval_sampled_mean": float(row["eval_sampled_mean"])} for row in rows]


def condition_spec(config: dict, metadata: dict, curve: list[dict], protocol: str) -> dict:
    versions = dict(metadata.get("versions", {}))
    if "stable_baselines3" in versions:
        versions["stable-baselines3"] = versions.pop("stable_baselines3")
    # Algorithm-specific optimizers/rollouts are deliberately not matched: they are
    # part of the declared implementations being compared, not controlled away.
    return {"protocol": protocol,
            "conditions": {key: config.get(key) for key in (
                "phase", "env_id", "total_timesteps", "n_envs", "net_arch", "activation",
                "eval_freq", "n_eval_episodes", "final_eval_episodes", "eval_seed",
                "final_eval_seed", "threshold", "sustain_evals")},
            "versions": versions,
            "source_sha256": metadata.get("runtime", {}).get("source_sha256"),
            "grid": [point["timesteps"] for point in curve]}


def load_runs(roots: list[Path]) -> tuple[list[Run], list[dict]]:
    if not roots:
        raise ValueError("Supply at least one results directory")
    for root in roots:
        if not root.is_dir():
            raise ValueError(f"Results directory does not exist: {root}")
    runs, audit, seen = [], [], {}
    paths = sorted({path.resolve() for root in roots
                    for path in root.glob("*/*/*/seed_*/config.json")})
    for config_path in paths:
        path = config_path.parent
        try:
            if (path / ".running").exists() or not (path / "summary.json").exists():
                audit.append({"path": str(path), "status": "incomplete", "reason": "No completed run"})
                continue
            metadata = json.loads(config_path.read_text(encoding="utf-8"))
            cfg = metadata["config"]
            summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
            protocol = cfg.get("protocol", "reference-unversioned")
            if cfg["env_id"] != "CartPole-v1":
                raise ValueError("Only CartPole-v1 is covered by this analysis protocol")
            if cfg["algo"] not in {"a2c", "ppo"}:
                raise ValueError("Unknown algorithm")
            if protocol == PROTOCOL:
                validated = ExperimentConfig(**cfg)
                if metadata.get("config_sha256") != validated.fingerprint():
                    raise ValueError("Configuration checksum mismatch")
                if summary.get("timesteps_done") != cfg["total_timesteps"]:
                    raise ValueError("Actual budget mismatch")
                for name in ("eval.csv", "evaluation_episodes.csv", "final_episodes.csv", "model.zip"):
                    artifact = path / name
                    if not artifact.exists() or file_hash(artifact) != summary.get("artifact_sha256", {}).get(name):
                        raise ValueError(f"Missing or modified artifact: {name}")
            curve = read_curve(path / "eval.csv")
            budget = cfg["total_timesteps"]
            # The reference's extra final rollout evaluation must not extend its stated horizon.
            curve = [row for row in curve if row["timesteps"] <= budget]
            metrics = curve_metrics([r["timesteps"] for r in curve],
                                    [r["eval_sampled_mean"] for r in curve], budget,
                                    cfg.get("threshold", 475.0), cfg.get("sustain_evals", 2))
            value = summary.get("final_eval_sampled_mean")
            if value is not None:
                value = float(value)
                if not 0 <= value <= 500:
                    raise ValueError("Invalid independent final evaluation")
            metrics["final_eval_sampled_mean"] = value
            spec = condition_spec(cfg, metadata, curve, protocol)
            stratum = hashlib.sha256(canonical_json(spec).encode()).hexdigest()[:12]
            identity = (stratum, cfg["algo"], cfg["group"], cfg["name"], cfg["seed"])
            digest = (canonical_json(metadata), file_hash(path / "eval.csv"),
                      canonical_json(summary))
            if identity in seen:
                if seen[identity] != digest:
                    conflicting = [r for r in runs if (r.stratum, r.config["algo"],
                                   r.config["group"], r.config["name"], r.config["seed"]) == identity]
                    for previous in conflicting:
                        runs.remove(previous)
                        for item in audit:
                            if item["path"] == str(previous.path):
                                item.update(status="excluded", reason="Conflicting duplicate run identity")
                    raise ValueError(f"Conflicting duplicate run identity: {identity}")
                audit.append({"path": str(path), "status": "duplicate", "reason": "Identical copy"})
                continue
            seen[identity] = digest
            runs.append(Run(path, cfg, metadata, summary, curve, metrics, protocol, stratum))
            audit.append({"path": str(path), "status": "loaded", "stratum": stratum,
                          "reason": ("Versioned protocol" if protocol == PROTOCOL
                                     else "Legacy protocol: descriptive analysis only")})
        except (OSError, ValueError, KeyError, TypeError) as error:
            audit.append({"path": str(path), "status": "excluded", "reason": str(error)})
    # Include experiments that failed before producing config.json, and runs never started.
    audited = {Path(row["path"]).resolve() for row in audit}
    for root in roots:
        for plan_path in sorted(root.glob("plan_*.json")):
            try:
                plan = json.loads(plan_path.read_text(encoding="utf-8"))
                for config in plan["configs"]:
                    cfg = ExperimentConfig(**config)
                    planned = cfg.run_dir(root).resolve()
                    if planned not in audited:
                        audit.append({"path": str(planned), "status": "missing",
                                      "reason": f"Planned run missing: {plan_path.name}"})
                        audited.add(planned)
            except (OSError, ValueError, KeyError, TypeError) as error:
                audit.append({"path": str(plan_path), "status": "invalid_plan", "reason": str(error)})
    return runs, audit
