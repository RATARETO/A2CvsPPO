"""Тест на прочность: уже обученные модели играют эпизоды ДЛИННЕЕ обычного лимита.

В CartPole-v1 эпизод обрывается на 500 шагах, поэтому политика, которая держит шест 500 шагов, и политика,
которая держала бы его 5000, выглядят одинаково. Здесь мы снимаем лимит (по умолчанию до 2000 шагов) и смотрим,
кто реально продолжает держать шест. Переобучать ничего не нужно: используются сохранённые model.zip.

    python -m src.stress_test                                   # все модели из results/, 5 процессов
    python -m src.stress_test --groups learning_rate n_epochs   # только выбранные серии
    python -m src.stress_test --max-steps 5000 --episodes 5 --greedy

Результат (папка analysis/ по умолчанию):
    stress_test.csv                  по одной строке на запуск
    stress_summary.csv               среднее по сидам для каждой конфигурации
    stress_<algo>_<серия>.png        столбики по значениям параметра (точки = сиды)
Только для CartPole-v1 (у других сред нет смысла «держать дольше»).
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analyze import _bar_dots, _colors, _label, _ordered_names, _save, _set_value_ticks


def _evaluate_run(task: tuple[str, int, int, bool]) -> dict | None:
    """Выполняется в отдельном процессе: загружает модель и прогоняет длинные эпизоды."""
    run_dir, max_steps, episodes, greedy = task
    import torch
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.evaluation import evaluate_policy

    from .algos import ALGOS

    torch.set_num_threads(1)
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))["config"]
    if meta["env_id"] != "CartPole-v1":
        return None
    model = ALGOS[meta["algo"]].load(run_dir / "model", device="cpu")
    torch.manual_seed(meta["seed"])

    row = {k: meta[k] for k in ("algo", "group", "name", "sweep_index", "seed")}
    for mode, deterministic in (("sampled", False), ("greedy", True)):
        if deterministic and not greedy:
            continue
        env = make_vec_env(meta["env_id"], n_envs=1, seed=meta["seed"] + 30_000,
                           env_kwargs={"max_episode_steps": max_steps})
        _, lengths = evaluate_policy(model, env, n_eval_episodes=episodes, deterministic=deterministic,
                                     return_episode_rewards=True)
        env.close()
        lengths = np.asarray(lengths, dtype=float)
        row[f"survival_{mode}"] = float(lengths.mean())                    # средняя длина эпизода
        row[f"full_rate_{mode}"] = float((lengths >= max_steps).mean())    # доля эпизодов, дошедших до лимита
    return row


def plot_stress(df: pd.DataFrame, out_dir: Path, max_steps: int) -> None:
    """Для каждой серии: средняя длина эпизода и доля эпизодов, дошедших до лимита."""
    modes = [m for m in ("sampled", "greedy") if f"survival_{m}" in df.columns]
    for (algo, group), sub in df.groupby(["algo", "group"]):
        names = _ordered_names(sub)
        colors = _colors(len(names))
        labels = [_label(n) for n in names]
        fig, axes = plt.subplots(1, 2 * len(modes), figsize=(7 * len(modes) + 2, 4.6), squeeze=False)
        for k, mode in enumerate(modes):
            ax_len, ax_rate = axes[0, 2 * k], axes[0, 2 * k + 1]
            for i, (name, color) in enumerate(zip(names, colors)):
                r = sub[sub["name"] == name]
                _bar_dots(ax_len, i, r[f"survival_{mode}"], color)
                _bar_dots(ax_rate, i, r[f"full_rate_{mode}"], color)
            ax_len.axhline(500, color="gray", linestyle=":")
            ax_len.text(len(names) - 0.5, 500, " обычный лимит 500", va="bottom", ha="right", fontsize=8, color="gray")
            ax_len.set_ylim(0, max_steps * 1.05)
            ax_len.set_title(f"Средняя длина эпизода, шагов ({mode}); лимит {max_steps}", fontsize=10)
            ax_rate.set_ylim(0, 1.05)
            ax_rate.set_title(f"Доля эпизодов, дошедших до {max_steps} шагов ({mode})", fontsize=10)
            for ax in (ax_len, ax_rate):
                _set_value_ticks(ax, labels)
        fig.suptitle(f"{algo.upper()} — серия «{group}»: тест на прочность (столбик = среднее по сидам, точки = сиды)",
                     fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        _save(fig, out_dir / f"stress_{algo}_{group}.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Тест на прочность: длинные эпизоды для обученных моделей")
    parser.add_argument("--results", default="results", help="папка с результатами (вход)")
    parser.add_argument("--out", default="analysis", help="папка для таблиц и графиков (выход)")
    parser.add_argument("--groups", nargs="*", default=None, help="только эти серии (по умолчанию все)")
    parser.add_argument("--algo", default=None, help="только этот алгоритм (ppo или a2c)")
    parser.add_argument("--max-steps", type=int, default=2000, help="лимит длины эпизода (по умолчанию 2000)")
    parser.add_argument("--episodes", type=int, default=5, help="эпизодов на модель (по умолчанию 5)")
    parser.add_argument("--greedy", action="store_true", help="дополнительно оценить жадной политикой (дольше)")
    parser.add_argument("--workers", type=int, default=5, help="число параллельных процессов")
    args = parser.parse_args()

    results_dir, out_dir = Path(args.results), Path(args.out)
    run_dirs = []
    for model_path in sorted(results_dir.glob("*/*/*/seed_*/model.zip")):
        run_dir = model_path.parent
        algo, group = run_dir.parents[2].name, run_dir.parents[1].name
        if args.algo and algo != args.algo:
            continue
        if args.groups and group not in args.groups:
            continue
        run_dirs.append(str(run_dir))
    if not run_dirs:
        raise SystemExit(f"В {results_dir} не найдено моделей (ищу */*/*/seed_*/model.zip) для выбранных фильтров.")

    print(f"Моделей: {len(run_dirs)}, лимит эпизода: {args.max_steps}, эпизодов на модель: {args.episodes}, "
          f"процессов: {args.workers}")
    tasks = [(d, args.max_steps, args.episodes, args.greedy) for d in run_dirs]
    rows, started = [], time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_evaluate_run, t): t for t in tasks}
        for done, future in enumerate(as_completed(futures), start=1):
            try:
                row = future.result()
            except Exception as error:  # одна упавшая модель не должна останавливать всё
                print(f"[{done}/{len(tasks)}] ОШИБКА {futures[future][0]}: {error!r}", flush=True)
                continue
            if row is not None:
                rows.append(row)
            if done % 10 == 0 or done == len(tasks):
                print(f"[{done}/{len(tasks)}] {(time.time() - started) / 60:5.1f} мин", flush=True)

    if not rows:
        raise SystemExit("Нет результатов (тест работает только для CartPole-v1).")
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).sort_values(["algo", "group", "sweep_index", "seed"]).reset_index(drop=True)
    df.to_csv(out_dir / "stress_test.csv", index=False)

    value_cols = [c for c in df.columns if c.startswith(("survival_", "full_rate_"))]
    summary = df.groupby(["algo", "group", "name", "sweep_index"])[value_cols].mean().reset_index()
    summary = summary.sort_values(["algo", "group", "sweep_index"]).reset_index(drop=True)
    summary.to_csv(out_dir / "stress_summary.csv", index=False)

    plot_stress(df, out_dir, args.max_steps)
    print(f"Готово. Таблицы и графики: {out_dir.resolve()}")
    print(summary.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
