"""Массовый запуск серий экспериментов, параллельно на процессорных ядрах.

Примеры (из корня проекта):
    python -m src.sweep --list                         # показать серии и число запусков
    python -m src.sweep --quick                        # быстрая проверка (20k шагов, 2 сида, 2 серии) -> results_quick/
    python -m src.sweep --groups baseline learning_rate --seeds 5
    python -m src.sweep                                # всё для PPO, 5 сидов

Уже посчитанные запуски пропускаются, поэтому прерванный свип можно просто запустить заново.
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from .config import RESULTS_DIR, ExperimentConfig, make_config
from .experiments import COMMON, SWEEPS


def build_configs(algo: str, groups: list[str] | None, n_seeds: int, **common) -> list[ExperimentConfig]:
    sweeps = SWEEPS[algo]
    selected = groups or list(sweeps)
    unknown = [g for g in selected if g not in sweeps]
    if unknown:
        raise SystemExit(f"Нет таких серий для {algo}: {unknown}. Доступны: {list(sweeps)}")
    configs = []
    for group in selected:
        for index, overrides in enumerate(sweeps[group]):
            for seed in range(n_seeds):
                configs.append(make_config(algo, group, overrides, seed, sweep_index=index, **common))
    return configs


def _run_one(task: tuple[dict, str, bool]) -> str:
    """Выполняется в отдельном процессе. Импорт внутри — чтобы каждый процесс грузил torch у себя."""
    cfg_dict, results_root, force = task
    from .train import run_experiment

    cfg = ExperimentConfig(**cfg_dict)
    summary = run_experiment(cfg, results_root=results_root, force=force)
    return f"{cfg.algo}/{cfg.group}/{cfg.name}/seed_{cfg.seed}  final={summary['final_eval_mean']:.0f}  ({summary['train_time_sec']}s)"


def main() -> None:
    parser = argparse.ArgumentParser(description="Запуск серий экспериментов")
    parser.add_argument("--algo", default="ppo", choices=sorted(SWEEPS))
    parser.add_argument("--groups", nargs="*", default=None, help="какие серии запускать (по умолчанию все)")
    parser.add_argument("--seeds", type=int, default=5, help="сколько сидов на конфигурацию")
    parser.add_argument("--steps", type=int, default=None, help="переопределить бюджет шагов")
    parser.add_argument("--workers", type=int, default=5, help="число параллельных процессов (ядер у i5-9600K — 6)")
    parser.add_argument("--out", default=None, help="папка результатов (по умолчанию results/)")
    parser.add_argument("--quick", action="store_true", help="быстрая проверка: 20k шагов, 2 сида, серии baseline+learning_rate")
    parser.add_argument("--force", action="store_true", help="пересчитать даже готовые запуски")
    parser.add_argument("--list", action="store_true", help="только показать план и выйти")
    args = parser.parse_args()

    common = dict(COMMON)
    groups, n_seeds, out = args.groups, args.seeds, args.out
    if args.quick:
        common.update(total_timesteps=20_000, eval_freq=2_000, n_eval_episodes=5)
        groups = groups or ["baseline", "learning_rate"]
        n_seeds = min(n_seeds, 2)
        out = out or str(RESULTS_DIR.parent / "results_quick")
    if args.steps:
        common["total_timesteps"] = args.steps
    out = out or str(RESULTS_DIR)

    configs = build_configs(args.algo, groups, n_seeds, **common)

    if args.list:
        per_group: dict[str, int] = {}
        for c in configs:
            per_group[c.group] = per_group.get(c.group, 0) + 1
        for group, count in per_group.items():
            print(f"{group:15s} {count} запусков")
        print(f"ИТОГО: {len(configs)} запусков по {common['total_timesteps']} шагов")
        return

    print(f"Запусков: {len(configs)}, процессов: {args.workers}, результаты: {out}")
    tasks = [(c.to_dict(), out, args.force) for c in configs]
    started = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run_one, t): t for t in tasks}
        for future in as_completed(futures):
            done += 1
            try:
                message = future.result()
            except Exception as error:  # один упавший запуск не должен останавливать весь свип
                cfg = futures[future][0]
                message = f"ОШИБКА {cfg['group']}/{cfg['name']}/seed_{cfg['seed']}: {error!r}"
            elapsed = (time.time() - started) / 60
            print(f"[{done}/{len(tasks)}] {elapsed:5.1f} мин  {message}", flush=True)


if __name__ == "__main__":  # обязательно на Windows: иначе дочерние процессы запустят свип рекурсивно
    main()
