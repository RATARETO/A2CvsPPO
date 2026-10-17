"""Описание одного запуска (эксперимента).

Главная идея: ВСЁ, что влияет на результат (алгоритм, среда, сид, гиперпараметры),
лежит в одном объекте ExperimentConfig. Его же мы сохраняем в config.json рядом
с результатами, поэтому по любой папке результатов можно восстановить,
с какими параметрами она получена.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"


@dataclass
class ExperimentConfig:
    # --- идентификация запуска ---
    algo: str = "ppo"          # "ppo" или "a2c"
    group: str = "baseline"    # к какой серии относится запуск (например "learning_rate")
    name: str = "default"      # имя конфигурации внутри серии (например "learning_rate=0.001")
    sweep_index: int = 0       # порядковый номер конфигурации в серии (нужен для порядка на графиках)
    seed: int = 0

    # --- среда и бюджет ---
    env_id: str = "CartPole-v1"
    total_timesteps: int = 100_000
    n_envs: int = 1            # сколько копий среды работает параллельно

    # --- архитектура сети (общая для PPO и A2C, чтобы сравнение было честным) ---
    net_arch: list[int] = field(default_factory=lambda: [64, 64])
    activation: str = "tanh"   # "tanh" или "relu"

    # --- протокол оценки ---
    eval_freq: int = 5_000         # как часто (в шагах среды) оцениваем политику во время обучения
    n_eval_episodes: int = 10      # сколько эпизодов в каждой промежуточной оценке
    final_eval_episodes: int = 30  # сколько эпизодов в итоговой оценке

    # --- гиперпараметры алгоритма: то, что передаётся прямо в PPO(...) / A2C(...) ---
    # Хранятся только ПЕРЕОПРЕДЕЛЕНИЯ дефолтов. Пустой словарь = дефолты Stable-Baselines3.
    algo_params: dict[str, Any] = field(default_factory=dict)

    def run_dir(self, results_root: Path | str = RESULTS_DIR) -> Path:
        """results/<algo>/<group>/<name>/seed_<seed>/"""
        return Path(results_root) / self.algo / self.group / self.name / f"seed_{self.seed}"

    def policy_kwargs_spec(self) -> dict[str, Any]:
        return {"net_arch": list(self.net_arch), "activation": self.activation}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


CONFIG_FIELDS = {f.name for f in fields(ExperimentConfig)}


def format_value(value: Any) -> str:
    """Короткая запись значения для имени папки: [64, 64] -> '64x64', 0.001 -> '0.001'."""
    if isinstance(value, (list, tuple)):
        return "x".join(format_value(v) for v in value)
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def name_from_overrides(overrides: dict[str, Any]) -> str:
    if not overrides:
        return "default"
    return ",".join(f"{k}={format_value(v)}" for k, v in overrides.items())


def make_config(
    algo: str,
    group: str,
    overrides: dict[str, Any],
    seed: int,
    sweep_index: int = 0,
    **common: Any,
) -> ExperimentConfig:
    """Собирает ExperimentConfig из словаря «что поменять относительно дефолтов».

    Ключи, которые являются полями ExperimentConfig (n_envs, activation, net_arch ...),
    попадают в поля; все остальные (learning_rate, clip_range ...) уходят в algo_params.
    """
    top_level = {k: v for k, v in overrides.items() if k in CONFIG_FIELDS}
    algo_params = {k: v for k, v in overrides.items() if k not in CONFIG_FIELDS}
    return ExperimentConfig(
        algo=algo,
        group=group,
        name=name_from_overrides(overrides),
        sweep_index=sweep_index,
        seed=seed,
        algo_params=algo_params,
        **{**common, **top_level},
    )
