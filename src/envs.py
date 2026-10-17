"""Создание сред. Все среды для обучения/оценки создаются здесь, чтобы у PPO и A2C они были одинаковыми."""
from __future__ import annotations

import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env

from .config import ExperimentConfig


def make_train_env(cfg: ExperimentConfig):
    """Векторная среда для обучения (внутри уже есть Monitor, он считает награду эпизодов)."""
    return make_vec_env(cfg.env_id, n_envs=cfg.n_envs, seed=cfg.seed)


def make_eval_env(cfg: ExperimentConfig, seed_offset: int = 10_000):
    """Отдельная среда для оценки. Сид сдвинут, чтобы оценка не повторяла обучение."""
    return make_vec_env(cfg.env_id, n_envs=1, seed=cfg.seed + seed_offset)


def make_render_env(env_id: str):
    """Среда с картинкой — нужна только для видео, при обучении не используется."""
    return gym.make(env_id, render_mode="rgb_array")
