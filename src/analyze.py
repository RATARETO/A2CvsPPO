"""Анализ результатов: метрики, графики, сравнение алгоритмов.

Запуск (из корня проекта):
    python -m src.analyze                                   # читает results/, пишет в analysis/
    python -m src.analyze --results results_quick --out analysis_quick

Какие файлы получаются в папке analysis/ (на каждый алгоритм, <algo> = ppo или a2c):

  <algo>_<серия>_overview.png      ГЛАВНЫЙ график серии: кривые обучения + итог, AUC, шаги до решения, просадки
  <algo>_<серия>_seeds.png         каждый сид отдельной линией: видно разброс и срывы (это и есть «стабильность»)
  <algo>_<серия>_diagnostics.png   внутренности алгоритма (kl, clip_fraction, энтропия...): объясняют ПОЧЕМУ так получилось
  <algo>_overview_all_params.png   все параметры на одной странице: как меняется AUC при изменении каждого параметра
  <algo>_sensitivity.png           рейтинг параметров по силе влияния (с линией «шум сидов»)
  config_summary.csv, run_metrics.csv, <algo>_sensitivity.csv    таблицы с цифрами

  Если в results/ есть и PPO, и A2C, дополнительно:
  compare_<серия>.png              одна и та же серия у двух алгоритмов рядом
  compare_sensitivity.png          чувствительность параметров: PPO против A2C
  compare_best.png                 лучшая найденная конфигурация каждого алгоритма и их дефолты

Сетки сочетаний (серии с префиксом grid_, например grid_lr_clip) получают отдельные графики:
  <algo>_<серия>_heatmap.png       тепловые карты: AUC, итог, доля решивших, шаги до решения, просадка, разброс сидов
  <algo>_<серия>_interaction.png   линии AUC: непараллельные линии = параметры влияют друг на друга
Награды всех сред приводятся к шкале 0..500 (см. ENV_SPECS), поэтому графики разных сред сопоставимы.

Метрики одного запуска (по кривой eval_sampled_mean):
    final              средняя оценка за последние LAST_K точек (итоговое качество)
    auc                средняя оценка по всей кривой / 500  (0..1; скорость + качество обучения)
    solved             держался ли порог «решено» SUSTAIN оценок подряд (одиночный всплеск не считается)
    steps_to_threshold число шагов до начала первой такой серии (NaN, если не достигнут)
    min_after_solve    минимум оценки ПОСЛЕ этого момента (низкий = «срыв», нестабильность)
Чувствительность параметра = разброс средних AUC между значениями этого параметра (max - min).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # рисуем в файлы, окно не нужно
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

MAX_RETURN = 500.0  # все награды приводятся к шкале 0..500 («нормированная награда»)
# Для каждой среды: минимальная и максимальная награда за эпизод и порог «решено» (в исходных единицах среды).
# CartPole: 0..500, порог 475. Acrobot: -500..0, порог -100. Нормировка: score = 500 * (R - min) / (max - min).
# Чтобы подключить другую среду, добавь строку сюда.
ENV_SPECS = {
    "CartPole-v1": {"min": 0.0, "max": 500.0, "threshold": 475.0},
    "Acrobot-v1": {"min": -500.0, "max": 0.0, "threshold": -100.0},
}


def normalize_return(value, env_id: str):
    spec = ENV_SPECS[env_id]
    return MAX_RETURN * (value - spec["min"]) / (spec["max"] - spec["min"])
LAST_K = 5          # сколько последних оценок усредняем для «итогового» качества
SUSTAIN = 2         # сколько оценок ПОДРЯД должно быть >= порога, чтобы считать, что агент «решил» задачу
Y = "eval_sampled_mean"

KEYS = ["algo", "group", "name", "sweep_index", "seed"]
ALGO_COLORS = {"ppo": "tab:blue", "a2c": "tab:orange"}

# (колонка в eval.csv, заголовок графика, логарифмическая шкала?)
DIAGNOSTICS = [
    ("approx_kl", "approx_kl: насколько сдвинулась политика за обновление", False),
    ("clip_fraction", "clip_fraction: доля шагов, где сработал клиппинг (только PPO)", False),
    ("entropy_loss", "entropy_loss: чем ближе к 0, тем увереннее политика", False),
    ("value_loss", "value_loss: ошибка критика (лог. шкала)", True),
    ("explained_variance", "explained_variance: 1 = критик предсказывает идеально", False),
]


# ----------------------------------------------------------------------------------------
# Загрузка и метрики
# ----------------------------------------------------------------------------------------
def load_runs(results_dir: Path) -> pd.DataFrame:
    frames = []
    for eval_path in sorted(results_dir.glob("*/*/*/seed_*/eval.csv")):
        run_dir = eval_path.parent
        meta = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))["config"]
        df = pd.read_csv(eval_path)
        df = df[df["timesteps"] <= meta["total_timesteps"]]  # отбрасываем «перебег» из-за размера rollout
        if Y not in df.columns:
            raise SystemExit(f"{eval_path}: старый формат eval.csv. Удали results/ (и results_quick/) и перезапусти эксперименты.")
        env_id = meta["env_id"]
        if env_id not in ENV_SPECS:
            raise SystemExit(f"Среда {env_id} не описана в ENV_SPECS (src/analyze.py): добавь минимум, максимум и порог награды.")
        df[Y] = normalize_return(df[Y], env_id)
        df["threshold"] = float(normalize_return(ENV_SPECS[env_id]["threshold"], env_id))
        for key in KEYS:
            df[key] = meta[key]
        frames.append(df)
    if not frames:
        raise SystemExit(f"В {results_dir} нет завершённых запусков (ищу */*/*/seed_*/eval.csv).")
    return pd.concat(frames, ignore_index=True)


def run_metrics(g: pd.DataFrame) -> dict:
    g = g.sort_values("timesteps")
    y, t = g[Y].to_numpy(), g["timesteps"].to_numpy()
    ok = y >= float(g["threshold"].iloc[0])
    first = next((i for i in range(len(y) - SUSTAIN + 1) if ok[i:i + SUSTAIN].all()), None)
    return {
        "final": float(y[-LAST_K:].mean()),
        "auc": float(y.mean() / MAX_RETURN),
        "solved": first is not None,
        "steps_to_threshold": float(t[first]) if first is not None else np.nan,
        "min_after_solve": float(y[first:].min()) if first is not None else np.nan,
    }


def per_run_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(KEYS):
        rows.append({**dict(zip(KEYS, key)), **run_metrics(g)})
    return pd.DataFrame(rows)


def per_config_table(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (algo, group, name, idx), g in runs.groupby(["algo", "group", "name", "sweep_index"]):
        rows.append({
            "algo": algo, "group": group, "name": name, "sweep_index": idx, "n_seeds": len(g),
            "final_mean": g["final"].mean(), "final_std": g["final"].std(ddof=0),
            "auc_mean": g["auc"].mean(), "auc_std": g["auc"].std(ddof=0),
            "solved_rate": g["solved"].mean(),
            "median_steps_to_threshold": g["steps_to_threshold"].median(),
            "mean_min_after_solve": g["min_after_solve"].mean(),
        })
    return pd.DataFrame(rows).sort_values(["algo", "group", "sweep_index"]).reset_index(drop=True)


SENS_COLUMNS = ["group", "n_values", "auc_range", "final_range", "share_solved", "worst_value", "best_value"]


def sensitivity_table(configs: pd.DataFrame, algo: str) -> pd.DataFrame:
    rows = []
    for group, g in configs[configs["algo"] == algo].groupby("group"):
        if group == "baseline" or group.startswith("grid_") or len(g) < 2:
            continue
        rows.append({
            "group": group,
            "n_values": len(g),
            "auc_range": g["auc_mean"].max() - g["auc_mean"].min(),
            "final_range": g["final_mean"].max() - g["final_mean"].min(),
            "share_solved": float((g["solved_rate"] > 0).mean()),
            "worst_value": g.loc[g["auc_mean"].idxmin(), "name"],
            "best_value": g.loc[g["auc_mean"].idxmax(), "name"],
        })
    table = pd.DataFrame(rows, columns=SENS_COLUMNS)  # пустая таблица тоже корректна (если есть только baseline)
    return table.sort_values("auc_range", ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------------------------
# Вспомогательные функции для графиков
# ----------------------------------------------------------------------------------------
def _label(name: str) -> str:
    """'learning_rate=0.001' -> '0.001'; 'n_envs=2,n_steps=1024' -> '2 (n_steps=1024)'."""
    first, *rest = name.split(",")
    value = first.split("=", 1)[1] if "=" in first else first
    return f"{value} ({','.join(rest)})" if rest else value


def _ordered_names(sub: pd.DataFrame) -> list[str]:
    return sub[["sweep_index", "name"]].drop_duplicates().sort_values("sweep_index")["name"].tolist()


def _colors(n: int) -> list:
    cmap = plt.get_cmap("viridis")
    return [cmap(i / max(1, n - 1)) for i in range(n)]


def _curve(sub: pd.DataFrame, name: str, col: str) -> pd.DataFrame:
    curve = sub[sub["name"] == name].groupby("timesteps")[col].agg(["mean", "std"]).reset_index()
    curve["std"] = curve["std"].fillna(0.0)
    return curve


def _draw_curves(ax, sub: pd.DataFrame, names: list[str], colors: list, lw: float = 1.8) -> None:
    for name, color in zip(names, colors):
        c = _curve(sub, name, Y)
        ax.plot(c["timesteps"], c["mean"], color=color, label=_label(name), linewidth=lw)
        ax.fill_between(c["timesteps"], (c["mean"] - c["std"]).clip(lower=0),
                        (c["mean"] + c["std"]).clip(upper=MAX_RETURN), color=color, alpha=0.15)
    ax.axhline(MAX_RETURN, color="gray", linestyle=":")
    ax.set_ylim(0, MAX_RETURN * 1.05)


def _bar_dots(ax, i: int, values, color) -> None:
    """Столбик = среднее по сидам, усы = std, точки = отдельные сиды."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return
    ax.bar(i, values.mean(), color=color, alpha=0.8, width=0.7)
    ax.errorbar(i, values.mean(), yerr=values.std(), color="black", capsize=3, linewidth=1)
    jitter = np.linspace(-0.2, 0.2, len(values)) if len(values) > 1 else np.zeros(1)
    ax.scatter(i + jitter, values, color="black", s=10, zorder=3)


def _set_value_ticks(ax, labels: list[str]) -> None:
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)


def _save(fig, path: Path) -> None:
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------------------
# Графики одной серии
# ----------------------------------------------------------------------------------------
def plot_group_overview(df, runs, algo: str, group: str, out_dir: Path) -> None:
    sub = df[(df["algo"] == algo) & (df["group"] == group)]
    rsub = runs[(runs["algo"] == algo) & (runs["group"] == group)]
    names = _ordered_names(sub)
    colors = _colors(len(names))
    labels = [_label(n) for n in names]
    max_t = sub["timesteps"].max()

    fig = plt.figure(figsize=(17, 9))
    gs = GridSpec(2, 3, figure=fig)
    ax_curve = fig.add_subplot(gs[0, :2])
    ax_final = fig.add_subplot(gs[0, 2])
    ax_auc = fig.add_subplot(gs[1, 0])
    ax_steps = fig.add_subplot(gs[1, 1])
    ax_min = fig.add_subplot(gs[1, 2])

    _draw_curves(ax_curve, sub, names, colors)
    for i, (name, color) in enumerate(zip(names, colors)):
        r = rsub[rsub["name"] == name]
        solved = r[r["solved"].astype(bool)]
        _bar_dots(ax_final, i, r["final"], color)
        _bar_dots(ax_auc, i, r["auc"], color)
        _bar_dots(ax_steps, i, solved["steps_to_threshold"], color)
        ax_steps.text(i, max_t * 1.03, f"{len(solved)}/{len(r)}", ha="center", fontsize=8)
        _bar_dots(ax_min, i, solved["min_after_solve"], color)

    ax_curve.set_title("Кривые обучения: среднее ± std по сидам")
    ax_curve.set_xlabel("шаги среды")
    ax_curve.set_ylabel("нормированная награда (0..500)")
    ax_curve.legend(title=group, fontsize=8, loc="lower right")

    ax_final.set_title(f"Итоговая награда (последние {LAST_K} оценок)")
    ax_final.set_ylim(0, MAX_RETURN * 1.05)
    ax_auc.set_title("AUC: скорость и качество обучения вместе (0..1)")
    ax_auc.set_ylim(0, 1.02)
    ax_steps.set_title("Шагов до решения (подпись: сколько сидов решили)")
    ax_steps.set_ylim(0, max_t * 1.1)
    ax_min.set_title("Просадка после решения (выше = стабильнее)")
    ax_min.set_ylim(0, MAX_RETURN * 1.05)
    for ax in (ax_final, ax_auc, ax_steps, ax_min):
        _set_value_ticks(ax, labels)

    fig.suptitle(f"{algo.upper()} — серия «{group}»  (столбик = среднее, усы = std, точки = отдельные сиды)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, out_dir / f"{algo}_{group}_overview.png")


def plot_group_seeds(df, algo: str, group: str, out_dir: Path) -> None:
    sub = df[(df["algo"] == algo) & (df["group"] == group)]
    names = _ordered_names(sub)
    colors = _colors(len(names))
    n = len(names)
    ncols = min(4, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows), sharex=True, sharey=True, squeeze=False)
    for ax in axes.flat[n:]:
        ax.axis("off")
    for i, name in enumerate(names):
        ax = axes.flat[i]
        d = sub[sub["name"] == name]
        for _, g in d.groupby("seed"):
            g = g.sort_values("timesteps")
            ax.plot(g["timesteps"], g[Y], color=colors[i], alpha=0.75, linewidth=1.2)
        c = _curve(sub, name, Y)
        ax.plot(c["timesteps"], c["mean"], color="black", linewidth=2)
        ax.axhline(MAX_RETURN, color="gray", linestyle=":")
        ax.set_ylim(0, MAX_RETURN * 1.05)
        ax.set_title(name, fontsize=9)
    fig.supxlabel("шаги среды")
    fig.supylabel("награда за эпизод")
    fig.suptitle(f"{algo.upper()} — серия «{group}»: каждый сид отдельно (чёрная линия — среднее)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, out_dir / f"{algo}_{group}_seeds.png")


def plot_group_diagnostics(df, algo: str, group: str, out_dir: Path) -> None:
    sub = df[(df["algo"] == algo) & (df["group"] == group)]
    available = [(k, t, log) for k, t, log in DIAGNOSTICS if k in sub.columns and sub[k].notna().any()]
    if not available:
        return
    names = _ordered_names(sub)
    colors = _colors(len(names))
    n = len(available)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 3.9 * nrows), squeeze=False)
    for ax in axes.flat[n:]:
        ax.axis("off")
    for ax, (key, title, log) in zip(axes.flat, available):
        for name, color in zip(names, colors):
            c = _curve(sub, name, key).dropna(subset=["mean"])
            ax.plot(c["timesteps"], c["mean"], color=color, label=_label(name), linewidth=1.6)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("шаги среды")
        if log:
            ax.set_yscale("log")
    axes.flat[0].legend(title=group, fontsize=7)
    fig.suptitle(f"{algo.upper()} — серия «{group}»: диагностика обучения (среднее по сидам)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, out_dir / f"{algo}_{group}_diagnostics.png")


# ----------------------------------------------------------------------------------------
# Графики по всему алгоритму
# ----------------------------------------------------------------------------------------
def _baseline_row(configs: pd.DataFrame, algo: str):
    base = configs[(configs["algo"] == algo) & (configs["group"] == "baseline")]
    return base.iloc[0] if len(base) else None


def plot_overview_all_params(configs, runs, algo: str, out_dir: Path) -> None:
    cfg = configs[configs["algo"] == algo]
    groups = [g for g in cfg["group"].unique()
              if g != "baseline" and not g.startswith("grid_") and (cfg["group"] == g).sum() >= 2]
    if not groups:
        return
    base = _baseline_row(configs, algo)
    ncols = min(4, len(groups))
    nrows = math.ceil(len(groups) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.4 * nrows), sharey=True, squeeze=False)
    for ax in axes.flat[len(groups):]:
        ax.axis("off")
    for ax, group in zip(axes.flat, groups):
        c = cfg[cfg["group"] == group].sort_values("sweep_index")
        r = runs[(runs["algo"] == algo) & (runs["group"] == group)]
        xs = list(range(len(c)))
        for x, name in zip(xs, c["name"]):
            vals = r[r["name"] == name]["auc"].to_numpy()
            ax.scatter(np.full(len(vals), x), vals, color="gray", s=10, alpha=0.6, zorder=2)
        ax.errorbar(xs, c["auc_mean"], yerr=c["auc_std"], fmt="-o", color="tab:blue", capsize=3, zorder=3)
        if base is not None:
            ax.axhline(base["auc_mean"], linestyle="--", color="tab:red", linewidth=1)
        _set_value_ticks(ax, [_label(n) for n in c["name"]])
        ax.set_title(group, fontsize=10)
        ax.set_ylim(0, 1.02)
    fig.supylabel("AUC (0..1, выше = лучше)")
    fig.suptitle(f"{algo.upper()} — как меняется результат при изменении каждого параметра\n"
                 "(синяя линия = среднее ± std по сидам, серые точки = сиды, красный пунктир = дефолт)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, out_dir / f"{algo}_overview_all_params.png")


def plot_sensitivity(table: pd.DataFrame, configs: pd.DataFrame, algo: str, out_dir: Path) -> None:
    if table.empty:
        return
    t = table.sort_values("auc_range")
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(t) + 1.8))
    ax.barh(t["group"], t["auc_range"], color=ALGO_COLORS.get(algo, "tab:blue"))
    base = _baseline_row(configs, algo)
    if base is not None and base["n_seeds"] > 1:
        noise = 2 * base["auc_std"]
        ax.axvline(noise, linestyle="--", color="tab:red")
        ax.text(noise, 0.15, " уровень шума сидов\n (2×std дефолта)", color="tab:red", fontsize=8, va="center")
    ax.set_xlabel("чувствительность = max − min средней AUC по значениям параметра")
    ax.set_title(f"{algo.upper()}: насколько сильно параметр влияет на обучение\n(столбик короче красной линии ≈ эффект неотличим от шума)")
    _save(fig, out_dir / f"{algo}_sensitivity.png")



# ----------------------------------------------------------------------------------------
# Сетки сочетаний гиперпараметров (серии grid_*)
# ----------------------------------------------------------------------------------------
def _num(text: str):
    try:
        return float(text)
    except ValueError:
        return text


def _grid_frame(configs: pd.DataFrame, algo: str, group: str):
    """-> (cfg с колонками p1, p2; имена параметров; отсортированные значения) или None, если это не сетка 2 параметров."""
    cfg = configs[(configs["algo"] == algo) & (configs["group"] == group)].copy()
    parsed = [dict(item.split("=", 1) for item in name.split(",")) for name in cfg["name"]]
    keys = list(parsed[0].keys())
    if len(keys) != 2 or any(list(p.keys()) != keys for p in parsed):
        return None
    cfg["p1"] = [p[keys[0]] for p in parsed]
    cfg["p2"] = [p[keys[1]] for p in parsed]
    v1 = sorted(cfg["p1"].unique(), key=_num)
    v2 = sorted(cfg["p2"].unique(), key=_num)
    return cfg, keys, v1, v2


def _grid_array(cfg, col: str, v1, v2) -> np.ndarray:
    grid = np.full((len(v1), len(v2)), np.nan)
    for _, r in cfg.iterrows():
        grid[v1.index(r["p1"]), v2.index(r["p2"])] = r[col]
    return grid


def plot_grid(configs: pd.DataFrame, algo: str, group: str, out_dir: Path) -> None:
    frame = _grid_frame(configs, algo, group)
    if frame is None:
        return
    cfg, keys, v1, v2 = frame
    metrics = [
        ("auc_mean", "AUC (выше = лучше)", "viridis", 0, 1, lambda v: f"{v:.2f}"),
        ("final_mean", "Итоговая награда (0..500)", "viridis", 0, MAX_RETURN, lambda v: f"{v:.0f}"),
        ("solved_rate", "Доля сидов, решивших задачу", "viridis", 0, 1, lambda v: f"{v * 100:.0f}%"),
        ("median_steps_to_threshold", "Шагов до решения (меньше = лучше)", "viridis_r", None, None, lambda v: f"{v / 1000:.1f}k"),
        ("mean_min_after_solve", "Просадка после решения (выше = стабильнее)", "viridis", 0, MAX_RETURN, lambda v: f"{v:.0f}"),
        ("auc_std", "Разброс AUC между сидами (меньше = стабильнее)", "viridis_r", 0, None, lambda v: f"{v:.3f}"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, (col, title, cmap_name, vmin, vmax, fmt) in zip(axes.flat, metrics):
        grid = _grid_array(cfg, col, v1, v2)
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad("lightgray")
        image = ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        for i in range(len(v1)):
            for j in range(len(v2)):
                text = "—" if np.isnan(grid[i, j]) else fmt(grid[i, j])
                ax.text(j, i, text, ha="center", va="center", fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.75, lw=0))
        ax.set_xticks(range(len(v2)))
        ax.set_xticklabels(v2)
        ax.set_yticks(range(len(v1)))
        ax.set_yticklabels(v1)
        ax.set_xlabel(keys[1])
        ax.set_ylabel(keys[0])
        ax.set_title(title, fontsize=10)
        fig.colorbar(image, ax=ax, fraction=0.046)
    fig.suptitle(f"{algo.upper()} — сетка «{keys[0]} × {keys[1]}» (прочерк = ни один сид не решил)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, out_dir / f"{algo}_{group}_heatmap.png")

    auc, std = _grid_array(cfg, "auc_mean", v1, v2), _grid_array(cfg, "auc_std", v1, v2)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for j, b in enumerate(v2):
        ax1.errorbar(range(len(v1)), auc[:, j], yerr=std[:, j], marker="o", capsize=3, label=f"{keys[1]}={b}")
    for i, a in enumerate(v1):
        ax2.errorbar(range(len(v2)), auc[i, :], yerr=std[i, :], marker="o", capsize=3, label=f"{keys[0]}={a}")
    for ax, key, values, other in ((ax1, keys[0], v1, keys[1]), (ax2, keys[1], v2, keys[0])):
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(values)
        ax.set_xlabel(key)
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=8)
    ax1.set_ylabel("AUC (выше = лучше)")
    ax1.set_title(f"AUC как функция {keys[0]} (линии: значения {keys[1]})")
    ax2.set_title(f"AUC как функция {keys[1]} (линии: значения {keys[0]})")
    fig.suptitle(f"{algo.upper()} — взаимодействие параметров: непараллельные линии = параметры влияют друг на друга", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, out_dir / f"{algo}_{group}_interaction.png")

# ----------------------------------------------------------------------------------------
# Сравнение алгоритмов (если в results/ есть больше одного)
# ----------------------------------------------------------------------------------------
def plot_compare_group(df, group: str, algos: list[str], out_dir: Path) -> None:
    present = [a for a in algos if ((df["algo"] == a) & (df["group"] == group)).any()]
    if len(present) < 2:
        return
    fig, axes = plt.subplots(1, len(present), figsize=(7.5 * len(present), 4.8), sharey=True, squeeze=False)
    for ax, algo in zip(axes.flat, present):
        sub = df[(df["algo"] == algo) & (df["group"] == group)]
        names = _ordered_names(sub)
        _draw_curves(ax, sub, names, _colors(len(names)))
        ax.set_title(algo.upper())
        ax.set_xlabel("шаги среды")
        ax.legend(title=group, fontsize=8, loc="lower right")
    axes.flat[0].set_ylabel("награда за эпизод")
    fig.suptitle(f"Серия «{group}»: алгоритмы рядом (одинаковые оси)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, out_dir / f"compare_{group}.png")


def plot_compare_sensitivity(tables: dict[str, pd.DataFrame], out_dir: Path) -> None:
    series = [t.set_index("group")["auc_range"].rename(a) for a, t in tables.items() if not t.empty]
    if len(series) < 2:
        return
    p = pd.concat(series, axis=1).dropna()
    if p.empty:
        return
    p = p.loc[p.mean(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(8, 0.7 * len(p) + 1.8))
    p.plot.barh(ax=ax, color=[ALGO_COLORS.get(c, None) for c in p.columns])
    ax.set_xlabel("чувствительность (max − min средней AUC)")
    ax.set_title("Чувствительность к общим гиперпараметрам: кто капризнее")
    _save(fig, out_dir / "compare_sensitivity.png")


def plot_compare_best(df, configs, algos: list[str], out_dir: Path) -> None:
    if len(algos) < 2:
        return
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for algo in algos:
        color = ALGO_COLORS.get(algo)
        cfg = configs[configs["algo"] == algo]
        best = cfg.sort_values("auc_mean").iloc[-1]
        entries = [(best, "-", f"{algo.upper()} лучший: {best['group']} / {best['name']}")]
        base = _baseline_row(configs, algo)
        if base is not None and not (base["group"] == best["group"] and base["name"] == best["name"]):
            entries.append((base, "--", f"{algo.upper()} дефолт"))
        for row, style, label in entries:
            sub = df[(df["algo"] == algo) & (df["group"] == row["group"]) & (df["name"] == row["name"])]
            c = _curve(sub, row["name"], Y)
            ax.plot(c["timesteps"], c["mean"], style, color=color, label=label, linewidth=2)
            ax.fill_between(c["timesteps"], (c["mean"] - c["std"]).clip(lower=0),
                            (c["mean"] + c["std"]).clip(upper=MAX_RETURN), color=color, alpha=0.12)
    ax.axhline(MAX_RETURN, color="gray", linestyle=":")
    ax.set_ylim(0, MAX_RETURN * 1.05)
    ax.set_xlabel("шаги среды")
    ax.set_ylabel("награда за эпизод")
    ax.set_title("Лучшие найденные конфигурации и дефолты (среднее ± std по сидам)")
    ax.legend(fontsize=8, loc="lower right")
    _save(fig, out_dir / "compare_best.png")


# ----------------------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Анализ результатов экспериментов")
    parser.add_argument("--results", default="results", help="папка с результатами (вход)")
    parser.add_argument("--out", default="analysis", help="папка для графиков и таблиц (выход)")
    args = parser.parse_args()

    results_dir, out_dir = Path(args.results), Path(args.out)
    df = load_runs(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = per_run_table(df)
    configs = per_config_table(runs)
    runs.to_csv(out_dir / "run_metrics.csv", index=False)
    configs.to_csv(out_dir / "config_summary.csv", index=False)

    algos = sorted(df["algo"].unique())
    sens_tables: dict[str, pd.DataFrame] = {}
    for algo in algos:
        for group in df[df["algo"] == algo]["group"].unique():
            if group.startswith("grid_"):
                plot_grid(configs, algo, group, out_dir)
            else:
                plot_group_overview(df, runs, algo, group, out_dir)
            plot_group_seeds(df, algo, group, out_dir)
            plot_group_diagnostics(df, algo, group, out_dir)
        plot_overview_all_params(configs, runs, algo, out_dir)
        sens = sensitivity_table(configs, algo)
        sens.to_csv(out_dir / f"{algo}_sensitivity.csv", index=False)
        plot_sensitivity(sens, configs, algo, out_dir)
        sens_tables[algo] = sens

    if len(algos) > 1:
        for group in df["group"].unique():
            plot_compare_group(df, group, algos, out_dir)
        plot_compare_sensitivity(sens_tables, out_dir)
        plot_compare_best(df, configs, algos, out_dir)

    print(f"Готово. Таблицы и графики: {out_dir.resolve()}")
    print("Файлы:", *sorted(p.name for p in out_dir.iterdir()), sep="\n  ")
    print()
    print(configs.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
