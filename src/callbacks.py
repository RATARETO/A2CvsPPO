"""Callback: каждые eval_freq шагов оцениваем политику и записываем строку в таблицу.

Callback в Stable-Baselines3 — это объект, который библиотека вызывает на каждом шаге обучения.
Мы используем его, чтобы получить КРИВУЮ обучения (награда как функция числа шагов),
а не только итоговое число.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

# Диагностика, которую SB3 сама считает при обновлении сети.
# Для PPO: approx_kl (насколько сдвинулась политика), clip_fraction (доля обрезанных шагов).
DIAG_KEYS = (
    "train/approx_kl",
    "train/clip_fraction",
    "train/entropy_loss",
    "train/value_loss",
    "train/explained_variance",
)


class EvalLogger(BaseCallback):
    def __init__(self, eval_env, eval_freq: int, n_eval_episodes: int):
        super().__init__(verbose=0)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.rows: list[dict] = []
        self._next_eval = 0
        self._t0 = 0.0

    def _on_training_start(self) -> None:
        self._t0 = time.time()
        self._evaluate()  # точка «до обучения» (случайная политика)

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next_eval:
            self._evaluate()
        return True

    def _on_training_end(self) -> None:
        if not self.rows or self.rows[-1]["timesteps"] != self.num_timesteps:
            self._evaluate()

    def _evaluate(self) -> None:
        # Две оценки на одну контрольную точку:
        #   sampled — действия СЭМПЛИРУЮТСЯ из распределения политики. Это ровно та политика, которую оптимизирует
        #             PPO/A2C. Именно её используем для сравнения: до обучения она честно даёт ~20, а не «повезло/нет».
        #   greedy  — берём самое вероятное действие (argmax). Нужна для видео и итоговых цифр «как работает агент».
        #             Внимание: у необученной сети со случайными весами greedy-политика иногда СЛУЧАЙНО держит шест
        #             все 500 шагов (зависит от сида), поэтому для кривых обучения она не годится.
        mean_s, std_s = evaluate_policy(
            self.model, self.eval_env, n_eval_episodes=self.n_eval_episodes, deterministic=False
        )
        mean_g, std_g = evaluate_policy(
            self.model, self.eval_env, n_eval_episodes=self.n_eval_episodes, deterministic=True
        )
        buffer = self.model.ep_info_buffer  # последние эпизоды ОБУЧЕНИЯ
        train_return = float(np.mean([ep["r"] for ep in buffer])) if len(buffer) else np.nan

        row = {
            "timesteps": int(self.num_timesteps),
            "eval_sampled_mean": float(mean_s),
            "eval_sampled_std": float(std_s),
            "eval_greedy_mean": float(mean_g),
            "eval_greedy_std": float(std_g),
            "train_return_mean": train_return,
            "wall_time_sec": time.time() - self._t0,
        }
        values = self.logger.name_to_value
        for key in DIAG_KEYS:
            row[key.split("/")[1]] = float(values.get(key, np.nan))
        self.rows.append(row)
        self._next_eval = (self.num_timesteps // self.eval_freq + 1) * self.eval_freq

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)
