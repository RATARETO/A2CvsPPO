"""Видео для уже обученных моделей.

    python -m src.render --run results/ppo/baseline/default/seed_0           # одна модель
    python -m src.render --random                                            # случайная политика (до обучения)
    python -m src.render --compare random results/ppo/baseline/default/seed_0
    python -m src.render --compare results/ppo/learning_rate/learning_rate=1e-05/seed_0 \
                                   results/ppo/learning_rate/learning_rate=0.01/seed_0

Флаг --sampled: действия СЭМПЛИРУЮТСЯ из политики (как в кривых обучения на графиках).
Без него используется «жадная» политика (самое вероятное действие) — она выглядит лучше, чем кривые,
и может скрывать нестабильность. Для сопоставления видео с графиками используй --sampled.

--compare принимает 2-4 пути к папкам запусков (слово random означает случайную политику)
и записывает их БОК О БОК с одинакового стартового состояния: удобно показывать «до/после» и «плохой/хороший».
Результат: папка videos/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .algos import ALGOS
from .config import ROOT
from .video import record_episodes, record_side_by_side


def _load(run: str):
    """-> (model или None, подпись, сид, env_id)"""
    if run == "random":
        return None, "RANDOM policy (no training)", 0, None
    run_dir = Path(run)
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))["config"]
    model = ALGOS[cfg["algo"]].load(run_dir / "model", device="cpu")
    return model, f"{cfg['algo'].upper()} | {cfg['name']} | seed {cfg['seed']}", cfg["seed"], cfg["env_id"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Записать видео политики")
    parser.add_argument("--run", help="папка запуска, например results/ppo/baseline/default/seed_0")
    parser.add_argument("--random", action="store_true", help="видео случайной политики (до обучения)")
    parser.add_argument("--compare", nargs="+", metavar="RUN", help="2-4 запуска бок о бок (можно слово random)")
    parser.add_argument("--env", default="CartPole-v1")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--sampled", action="store_true",
                        help="сэмплировать действия (как на графиках) вместо жадной политики")
    args = parser.parse_args()
    deterministic = not args.sampled
    suffix = "_sampled" if args.sampled else ""
    mode = "sampled" if args.sampled else "greedy"
    import torch
    torch.manual_seed(0)  # чтобы видео с --sampled воспроизводились

    if args.compare:
        if not 2 <= len(args.compare) <= 4:
            parser.error("--compare принимает от 2 до 4 запусков")
        loaded = [_load(r) for r in args.compare]
        env_id = next((e for *_, e in loaded if e), args.env)
        title = "_vs_".join(
            "random" if r == "random" else f"{Path(r).parent.name}_{Path(r).name}" for r in args.compare
        )
        out = ROOT / "videos" / f"compare_{title}{suffix}"
        paths = record_side_by_side([x[0] for x in loaded], [f"{x[1]} | {mode}" if x[0] is not None else x[1] for x in loaded],
                                    env_id, out, n_episodes=min(args.episodes, 3), seed=loaded[-1][2],
                                    deterministic=deterministic)
    elif args.random:
        paths = record_episodes(None, args.env, ROOT / "videos" / "random_policy",
                                label="RANDOM policy (no training)", n_episodes=args.episodes)
    elif args.run:
        model, label, seed, env_id = _load(args.run)
        paths = record_episodes(model, env_id, Path(args.run) / f"policy{suffix}", label=f"{label} | {mode}",
                                n_episodes=args.episodes, seed=seed, deterministic=deterministic)
    else:
        parser.error("укажи --run <папка>, --random или --compare <запуск> <запуск>")
    print("Сохранено:", *paths, sep="\n  ")


if __name__ == "__main__":
    main()
