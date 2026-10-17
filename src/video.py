"""Запись видео. Кадры пишутся потоком (а не копятся в памяти), с подписями на кадре."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .envs import make_render_env

SUPPORTED_FORMATS = {".mp4", ".webm", ".gif"}


def _as_rgb_uint8(frame: np.ndarray) -> np.ndarray:
    """Проверяет, что кадр — RGB (высота, ширина, 3), и приводит к uint8."""
    array = np.asarray(frame)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"Expected an RGB frame (height, width, 3), got {array.shape}.")
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating) and array.max() <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def _writer_kwargs(suffix: str, fps: float) -> dict[str, object]:
    if suffix == ".mp4":
        return {"fps": fps, "codec": "libx264", "pixelformat": "yuv420p", "macro_block_size": None}
    if suffix == ".webm":
        return {"fps": fps, "codec": "libvpx-vp9", "macro_block_size": None}
    if suffix == ".gif":  # GIF — превью; для него длительность кадра задаётся в секундах
        return {"duration": 1.0 / fps, "loop": 0}
    raise ValueError(f"Unsupported format {suffix!r}. Choose from: {sorted(SUPPORTED_FORMATS)}")


class MultiFormatWriter:
    """Пишет один и тот же поток кадров сразу в несколько форматов.

    Использование:
        with MultiFormatWriter("videos/demo", formats=(".mp4", ".gif")) as w:
            w.append(frame)
    """

    def __init__(self, base_path: str | Path, formats: Iterable[str] = (".mp4", ".gif"),
                 fps: int = 30, gif_stride: int = 2):
        base = Path(base_path)
        base.parent.mkdir(parents=True, exist_ok=True)
        self.gif_stride = gif_stride
        self._count = 0
        self.paths: list[Path] = []
        self._writers: dict[str, object] = {}
        for suffix in formats:
            if suffix not in SUPPORTED_FORMATS:
                raise ValueError(f"Unsupported format {suffix!r}")
            path = Path(str(base) + suffix)
            # GIF получает каждый gif_stride-й кадр в половинном размере: файл получается намного легче
            writer_fps = fps / gif_stride if suffix == ".gif" else fps
            self._writers[suffix] = imageio.get_writer(path, **_writer_kwargs(suffix, writer_fps))
            self.paths.append(path)

    def append(self, frame: np.ndarray) -> None:
        frame = _as_rgb_uint8(frame)
        for suffix, writer in self._writers.items():
            if suffix == ".gif":
                if self._count % self.gif_stride == 0:
                    writer.append_data(np.ascontiguousarray(frame[::2, ::2]))
            else:
                writer.append_data(frame)
        self._count += 1

    def close(self) -> None:
        for writer in self._writers.values():
            writer.close()

    def __enter__(self) -> "MultiFormatWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def annotate(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    """Рисует строки текста в левом верхнем углу кадра (шрифт по умолчанию — только латиница)."""
    image = Image.fromarray(_as_rgb_uint8(frame))
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default(size=18)
    except TypeError:  # старый Pillow без параметра size
        font = ImageFont.load_default()
    y = 8
    for text in lines:
        draw.text((10, y), text, fill=(0, 0, 0), font=font)
        y += 24
    return np.asarray(image)


def record_episodes(
    model,
    env_id: str,
    base_path: str | Path,
    label: str,
    n_episodes: int = 3,
    seed: int = 0,
    fps: int = 30,
    formats: Iterable[str] = (".mp4", ".gif"),
    deterministic: bool = True,
) -> list[Path]:
    """Записывает n_episodes ПОЛНЫХ эпизодов. model=None -> случайные действия.

    deterministic=True  — «жадная» политика (самое вероятное действие);
    deterministic=False — действия сэмплируются, как в кривых обучения на графиках.

    На кадре: подпись (label), номер эпизода, шаг и накопленная награда.
    После конца эпизода последний кадр удерживается ~1 секунду с пометкой, чем он закончился.
    """
    env = make_render_env(env_id)
    if model is None:
        env.action_space.seed(seed)

    with MultiFormatWriter(base_path, formats=formats, fps=fps) as writer:
        for episode in range(n_episodes):
            obs, _ = env.reset(seed=seed + episode)
            total, step = 0.0, 0
            while True:
                if model is None:
                    action = env.action_space.sample()
                else:
                    action, _ = model.predict(obs, deterministic=deterministic)
                    action = int(action)
                obs, reward, terminated, truncated, _ = env.step(action)
                total += float(reward)
                step += 1
                header = [label, f"episode {episode + 1}/{n_episodes}   step {step}", f"return {total:.0f}"]
                frame = annotate(env.render(), header)
                writer.append(frame)
                if terminated or truncated:
                    break
            status = "TIME LIMIT REACHED (success)" if truncated else "POLE FELL"
            held = annotate(env.render(), header + [status])
            for _ in range(fps):
                writer.append(held)
    env.close()
    return list(writer.paths)


def record_side_by_side(
    models: list,
    labels: list[str],
    env_id: str,
    base_path: str | Path,
    n_episodes: int = 1,
    seed: int = 0,
    fps: int = 30,
    formats: Iterable[str] = (".mp4", ".gif"),
    deterministic: bool = True,
) -> list[Path]:
    """Несколько политик рядом в одном кадре (слева направо). model=None -> случайные действия.

    Все среды стартуют с ОДИНАКОВОГО состояния (один сид), поэтому различие в поведении — заслуга политики.
    Когда эпизод одной панели закончился, её кадр замирает с пометкой, остальные доигрывают.
    """
    n = len(models)
    envs = [make_render_env(env_id) for _ in range(n)]
    for env, model in zip(envs, models):
        if model is None:
            env.action_space.seed(seed)

    with MultiFormatWriter(base_path, formats=formats, fps=fps) as writer:
        for episode in range(n_episodes):
            obs = [env.reset(seed=seed + episode)[0] for env in envs]
            alive = [True] * n
            totals, steps = [0.0] * n, [0] * n
            status = [""] * n
            last = [env.render() for env in envs]
            while any(alive):
                panels = []
                for i, env in enumerate(envs):
                    if alive[i]:
                        if models[i] is None:
                            action = env.action_space.sample()
                        else:
                            action, _ = models[i].predict(obs[i], deterministic=deterministic)
                            action = int(action)
                        obs[i], reward, terminated, truncated, _ = env.step(action)
                        totals[i] += float(reward)
                        steps[i] += 1
                        last[i] = env.render()
                        if terminated or truncated:
                            alive[i] = False
                            status[i] = "TIME LIMIT REACHED (success)" if truncated else "POLE FELL"
                    lines = [labels[i], f"episode {episode + 1}/{n_episodes}   step {steps[i]}", f"return {totals[i]:.0f}"]
                    if status[i]:
                        lines.append(status[i])
                    panels.append(annotate(last[i], lines))
                frame = np.concatenate(panels, axis=1)
                writer.append(frame)
            for _ in range(fps):  # удерживаем итоговый кадр ~1 секунду
                writer.append(frame)
    for env in envs:
        env.close()
    return list(writer.paths)
