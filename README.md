# PPO vs A2C на CartPole: стабильность обучения и чувствительность к гиперпараметрам

Студенческое исследование. Код основан на учебнике Stone Tao «RL with Lux: 1 — Intro to RL»
(часть CartPole + PPO), перенесён на современный Gymnasium / Stable-Baselines3 2.9 и расширен
до фреймворка для сравнения PPO и A2C: запуск серий экспериментов, сохранение результатов вместе
с гиперпараметрами, графики, видео.

Все команды выполняются **из корня проекта** (папка, где лежат `src/` и `requirements.txt`),
при **активированном виртуальном окружении**.

---

## 1. Установка (один раз)

Python 3.10.

**Git Bash (Windows):**

    python -m venv venv
    source venv/Scripts/activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt

**PowerShell (Windows):**

    py -3.10 -m venv venv
    .\venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    pip install -r requirements.txt

**macOS / Linux:**

    python3.10 -m venv venv
    source venv/bin/activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt

Активировать окружение нужно **в каждом новом окне терминала** (одна команда `source ...` / `.\...Activate.ps1`
из блока выше, без установки). Признак активного окружения — `(venv)` в начале строки.

---

## 2. Жизненный цикл проекта: команды по порядку

### Шаг 0. Проверка, что всё работает (5 минут)

    python -m src.main
    python -m src.sweep --quick
    python -m src.analyze --results results_quick --out analysis_quick

- `src.main` — демо: видео случайной политики, обучение PPO, видео обученной политики.
- `--quick` — мини-свип (20 000 шагов, 2 сида, 2 серии), результаты в `results_quick/` (не смешиваются с основными).

### Шаг 1. Посмотреть план: какие серии есть и сколько запусков

    python -m src.sweep --list
    python -m src.sweep --algo a2c --list

### Шаг 2. Базовый запуск (дефолтные гиперпараметры, 10 сидов)

    python -m src.sweep --groups baseline --seeds 10
    python -m src.analyze

### Шаг 3. Серии гиперпараметров PPO

Все сразу (около 2 часов при 50 000 шагов и 5 сидах; можно оставить на ночь):

    python -m src.sweep

Или по частям, самые важные серии первыми:

    python -m src.sweep --groups learning_rate clip_range n_epochs n_steps
    python -m src.sweep --groups batch_size gamma gae_lambda ent_coef n_envs
    python -m src.sweep --groups vf_coef max_grad_norm net_arch activation

Если запуск прервался (закрыла терминал, выключился компьютер) — просто запусти ту же команду ещё раз:
готовые запуски пропускаются.

### Шаг 4. Анализ: графики и таблицы

    python -m src.analyze

Результат в папке `analysis/` (описание файлов — в разделе 5).

### Шаг 5. Видео для презентации и статьи

    # одна обученная модель (3 эпизода)
    python -m src.render --run results/ppo/baseline/default/seed_0

    # «до и после» бок о бок
    python -m src.render --compare random results/ppo/baseline/default/seed_0

    # плохой и хороший гиперпараметр бок о бок
    python -m src.render --compare results/ppo/learning_rate/learning_rate=1e-05/seed_0 results/ppo/learning_rate/learning_rate=0.01/seed_0

    # то же, но с СЭМПЛИРОВАНИЕМ действий (как на графиках) — рекомендуется для сравнения с графиками
    python -m src.render --compare results/ppo/learning_rate/learning_rate=1e-05/seed_0 results/ppo/learning_rate/learning_rate=0.0003/seed_0 results/ppo/learning_rate/learning_rate=0.01/seed_0 --sampled

    # видео-сравнения по ВСЕМ гиперпараметрам (от минимума к максимуму) одной командой
    bash make_compare_videos.sh          # сид 0
    bash make_compare_videos.sh 3        # другой сид

Видео сохраняются в `videos/` (сравнения) или в папку запуска (одиночные). Без `--sampled` агент играет «жадно»
(самое вероятное действие) и выглядит лучше, чем на графиках: слабая нестабильность в таком видео не видна.

### Шаг 6. Сочетания гиперпараметров (сетки)

Серии с префиксом `grid_` меняют ДВА параметра сразу (все пары значений). В `src/experiments.py` заданы три сетки:

| Серия | Что проверяем | Значения | Запусков (5 сидов) |
|---|---|---|---|
| `grid_lr_clip` | сдерживает ли малый `clip_range` большой `learning_rate` | lr: 0.0001, 0.0003, 0.001, 0.01 × clip: 0.05, 0.1, 0.2, 0.5 | 80 |
| `grid_lr_epochs` | компенсирует ли большое число эпох малый `learning_rate` | lr: 0.0001, 0.0003, 0.001, 0.01 × n_epochs: 3, 10, 20, 40 | 80 |
| `grid_gamma_lambda` | пара «горизонт и шум оценки» (есть и у A2C) | gamma: 0.95, 0.99, 0.999 × lambda: 0.9, 0.95, 1.0 | 45 |

Всего 205 запусков. По времени ориентир около 1,5–2 часов (серия с `n_epochs=40` считается дольше): в полном свипе было
около 2,2 запуска в минуту. Сначала посмотри план, потом запускай по очереди:

    python -m src.sweep --groups grid_lr_clip grid_lr_epochs grid_gamma_lambda --list
    python -m src.sweep --groups grid_lr_clip
    python -m src.sweep --groups grid_lr_epochs
    python -m src.sweep --groups grid_gamma_lambda
    python -m src.analyze

(или все три сразу: `python -m src.sweep --groups grid_lr_clip grid_lr_epochs grid_gamma_lambda`).
Для финальных цифр статьи можно повторить на 10 сидах: добавь `--seeds 10`.

Новые графики в `analysis/` для каждой сетки:
`ppo_grid_<имя>_heatmap.png` (тепловые карты шести метрик), `ppo_grid_<имя>_interaction.png` (линии AUC: если линии
непараллельны, параметры влияют друг на друга), а также `_seeds.png` и `_diagnostics.png`.

Видео-сравнения «четырёх углов» сетки (слева направо: мин+мин, мин+макс, макс+мин, макс+макс):

    python -m src.render --compare "results/ppo/grid_lr_clip/learning_rate=0.0001,clip_range=0.05/seed_0" "results/ppo/grid_lr_clip/learning_rate=0.0001,clip_range=0.5/seed_0" "results/ppo/grid_lr_clip/learning_rate=0.01,clip_range=0.05/seed_0" "results/ppo/grid_lr_clip/learning_rate=0.01,clip_range=0.5/seed_0" --episodes 2 --sampled
    python -m src.render --compare "results/ppo/grid_lr_epochs/learning_rate=0.0001,n_epochs=3/seed_0" "results/ppo/grid_lr_epochs/learning_rate=0.0001,n_epochs=40/seed_0" "results/ppo/grid_lr_epochs/learning_rate=0.01,n_epochs=3/seed_0" "results/ppo/grid_lr_epochs/learning_rate=0.01,n_epochs=40/seed_0" --episodes 2 --sampled
    python -m src.render --compare "results/ppo/grid_gamma_lambda/gamma=0.95,gae_lambda=0.9/seed_0" "results/ppo/grid_gamma_lambda/gamma=0.95,gae_lambda=1/seed_0" "results/ppo/grid_gamma_lambda/gamma=0.999,gae_lambda=0.9/seed_0" "results/ppo/grid_gamma_lambda/gamma=0.999,gae_lambda=1/seed_0" --episodes 2 --sampled

(или `bash make_compare_videos.sh`: он запишет и видео для сеток, если они посчитаны).

Свои сетки: в `src/experiments.py` добавь строку
`"grid_моё_имя": grid("параметр1", [значения], "параметр2", [значения])` — имя серии обязательно начинается с `grid_`.

### Шаг 6б. Тест на прочность: модели на эпизодах длиннее 500 шагов

Переобучать не нужно: используются уже сохранённые `model.zip`. Модели играют эпизоды до 2000 шагов (вместо 500) и
показывают, кто продолжает держать шест, а кто просто дотянул до лимита.

    python -m src.stress_test --groups learning_rate n_epochs --workers 5
    python -m src.stress_test --workers 5

Первая команда (две главные серии) нужна для быстрой проверки; вторая прогоняет все модели из `results/`
(ориентир 10–20 минут при 5 процессах, это оценка). Результат: `analysis/stress_test.csv`,
`analysis/stress_summary.csv` и графики `analysis/stress_<algo>_<серия>.png`. Параметры: `--max-steps` (лимит, по
умолчанию 2000), `--episodes` (эпизодов на модель, по умолчанию 5), `--greedy` (дополнительно жадная политика).

### Шаг 6в. Вторая среда: Acrobot-v1 (проверка, что выводы не специфичны для CartPole)

Награда Acrobot отрицательная (−1 за шаг, от −500 до 0, «решено» около −100). Анализ сам приводит её к шкале 0..500
(см. `ENV_SPECS` в `src/analyze.py`), так что графики сравнимы с CartPole. Результаты идут в отдельную папку
`results_acrobot/`. Бюджет по умолчанию 100 000 шагов. Сначала замер на базовой серии (проверь, что среда учится и
сколько это занимает):

    python -m src.sweep --env Acrobot-v1 --groups baseline --seeds 3
    python -m src.analyze --results results_acrobot --out analysis_acrobot

Если базовая серия выходит на разумный результат, повтори главные серии (3 сида):

    python -m src.sweep --env Acrobot-v1 --groups learning_rate n_epochs batch_size clip_range gamma ent_coef --seeds 3
    python -m src.analyze --results results_acrobot --out analysis_acrobot

Это около 100 запусков по 100 000 шагов; время на запуск у Acrobot заранее неизвестно (оценить можно по baseline),
ориентир несколько часов. Тест на прочность (шаг 6б) к Acrobot не применяется.

### Шаг 7. Сравнение с A2C (когда сокомандник посчитал свои серии)

    python -m src.sweep --algo a2c
    python -m src.analyze

`analyze` сам увидит оба алгоритма и дополнительно нарисует `compare_*.png`.

### Шаг 8. Начать заново (удалить результаты)

Git Bash / macOS / Linux:

    rm -rf results results_quick analysis analysis_quick videos

PowerShell:

    Remove-Item -Recurse -Force results, results_quick, analysis, analysis_quick, videos

(Если какой-то папки нет, PowerShell покажет ошибку — это нормально.)

---

## 3. Справочник команд и флагов

### `python -m src.sweep` — запуск серий (много запусков, параллельно)

| Флаг | Что делает | По умолчанию |
|---|---|---|
| `--algo ppo\|a2c` | какой алгоритм | `ppo` |
| `--env ИМЯ` | среда: `CartPole-v1` или `Acrobot-v1` (для Acrobot результаты идут в `results_acrobot/`) | `CartPole-v1` |
| `--groups A B C` | какие серии запускать (несколько через пробел) | все серии |
| `--seeds N` | сколько сидов на конфигурацию | `5` |
| `--steps N` | переопределить бюджет шагов | из `experiments.py` |
| `--workers N` | число параллельных процессов | `5` |
| `--out ПАПКА` | куда писать результаты | `results/` |
| `--quick` | быстрая проверка (20 000 шагов, 2 сида, серии baseline + learning_rate, папка `results_quick/`) | выкл. |
| `--force` | пересчитать даже готовые запуски | выкл. |
| `--list` | только показать план и выйти | выкл. |

### `python -m src.train` — один запуск вручную

| Флаг | Что делает | По умолчанию |
|---|---|---|
| `--algo ppo\|a2c` | алгоритм | `ppo` |
| `--group ИМЯ` | в какую серию записать (папка `results/<algo>/<ИМЯ>/...`) | `manual` |
| `--steps N` | бюджет шагов | `100000` |
| `--seed K` | сид | `0` |
| `--set ключ=значение` | переопределить гиперпараметр; можно несколько раз | — |
| `--video` | записать видео после обучения | выкл. |
| `--force` | перезаписать, даже если запуск уже есть | выкл. |

Пример: `python -m src.train --set learning_rate=0.001 --set clip_range=0.1 --video`

### `python -m src.analyze` — графики и таблицы

| Флаг | Что делает | По умолчанию |
|---|---|---|
| `--results ПАПКА` | откуда читать результаты | `results` |
| `--out ПАПКА` | куда писать графики и таблицы | `analysis` |

### `python -m src.stress_test` — тест на прочность (длинные эпизоды)

| Флаг | Что делает | По умолчанию |
|---|---|---|
| `--results ПАПКА` | откуда читать модели | `results` |
| `--out ПАПКА` | куда писать таблицы и графики | `analysis` |
| `--groups A B` | только эти серии | все |
| `--algo ppo\|a2c` | только этот алгоритм | оба |
| `--max-steps N` | лимит длины эпизода | `2000` |
| `--episodes N` | эпизодов на модель | `5` |
| `--greedy` | дополнительно оценить жадной политикой | выкл. |
| `--workers N` | параллельных процессов | `5` |

### `python -m src.render` — видео готовых моделей

| Команда | Что делает |
|---|---|
| `--run ПАПКА_ЗАПУСКА` | видео одной модели |
| `--random` | видео случайной политики |
| `--compare A B [C D]` | 2–4 запуска бок о бок (`random` = случайная политика) |
| `--episodes N` | число эпизодов (по умолчанию 3, для `--compare` не больше 3) |
| `--sampled` | сэмплировать действия (как на графиках) вместо жадной политики; файлы получают суффикс `_sampled` |

### `python -m src.main` — демо (без флагов)

---

## 4. Словарь: какие слова что значат

| Слово | Значение | Пример |
|---|---|---|
| **серия** (`group`) | набор конфигураций, где меняется один параметр (или два, если это сетка `grid_*`) | `learning_rate`, `grid_lr_clip` |
| **конфигурация** (`name`) | одно значение параметра внутри серии | `learning_rate=0.001` |
| **сид** (`seed`) | число, фиксирующее случайность; разные сиды = независимые повторы | `seed_0` … `seed_4` |
| **бюджет** (`steps`, `total_timesteps`) | сколько шагов среды отводится на обучение | `50000` |
| **запуск** (run) | одна конфигурация × один сид | `results/ppo/baseline/default/seed_0` |
| **дефолт** (`baseline`) | гиперпараметры по умолчанию Stable-Baselines3 | серия `baseline` |

Не путать: у `sweep` флаг называется **`--groups`** (несколько серий), у `train` — **`--group`** (одна).
У `sweep` флаг `--out` — папка *результатов*, у `analyze` флаг `--results` — вход, а `--out` — выход.

---

## 5. Что получается после `analyze` (папка `analysis/`)

Для каждой серии (`<algo>` = `ppo` или `a2c`):

| Файл | Что показывает |
|---|---|
| `<algo>_<серия>_overview.png` | **главный график серии**: кривые обучения, итоговая награда, AUC, шагов до решения, просадка после решения |
| `<algo>_<серия>_seeds.png` | каждый сид отдельной линией: видно разброс и срывы (это и есть «стабильность») |
| `<algo>_<серия>_diagnostics.png` | `approx_kl`, `clip_fraction`, энтропия, ошибка критика: объясняют, ПОЧЕМУ кривые такие |

Для алгоритма в целом:

| Файл | Что показывает |
|---|---|
| `<algo>_overview_all_params.png` | все параметры на одной странице: как меняется AUC при изменении каждого |
| `<algo>_sensitivity.png` | рейтинг параметров по силе влияния; красная линия — уровень шума сидов |
| `config_summary.csv` | таблица метрик по конфигурациям |
| `run_metrics.csv` | таблица метрик по каждому запуску |
| `<algo>_sensitivity.csv` | цифры чувствительности |

Для сеток (серии `grid_*`) вместо `_overview.png`:

| Файл | Что показывает |
|---|---|
| `<algo>_grid_<имя>_heatmap.png` | тепловые карты: AUC, итоговая награда, доля решивших, шаги до решения, просадка, разброс сидов |
| `<algo>_grid_<имя>_interaction.png` | линии AUC; непараллельные линии = параметры влияют друг на друга |

Тест на прочность (`stress_test`): `stress_test.csv`, `stress_summary.csv`, `stress_<algo>_<серия>.png`.

Если посчитаны оба алгоритма:

| Файл | Что показывает |
|---|---|
| `compare_<серия>.png` | одна и та же серия у PPO и A2C рядом, одинаковые оси |
| `compare_sensitivity.png` | чувствительность общих параметров: кто капризнее |
| `compare_best.png` | лучшие найденные конфигурации и дефолты обоих алгоритмов |

Метрики: **final** — итоговая награда (последние 5 оценок); **AUC** — средняя награда по всей кривой / 500
(скорость и качество обучения вместе); **solved** — награда ≥ 475 держалась 2 оценки подряд;
**steps_to_threshold** — шагов до решения; **min_after_solve** — минимальная награда после решения
(низкое значение = срывы).

---

## 6. Как менять гиперпараметры

Единственный файл, который правится: `src/experiments.py`.

- **Изменить значения серии:** поменяй список, например
  `"clip_range": values("clip_range", [0.05, 0.1, 0.2, 0.3, 0.5])`.
- **Добавить серию:** добавь строку `"имя": values("имя", [...])`. Имя — параметр SB3 (`target_kl`, `n_epochs`...)
  или поле конфига (`n_envs`, `net_arch`, `activation`).
- **Комбинация нескольких параметров** (серия называется как хочешь):

      "lr_x_clip": [
          {"learning_rate": 1e-3, "clip_range": 0.1},
          {"learning_rate": 1e-3, "clip_range": 0.3},
          {"learning_rate": 3e-4, "clip_range": 0.1},
      ],

  Запуск: `python -m src.sweep --groups lr_x_clip --seeds 10`.
- **Бюджет и частота оценки:** блок `COMMON` в начале файла (должен быть одинаковым у PPO и A2C).

Правила: `batch_size` должен делить `n_steps × n_envs`; значения `net_arch` — списки, например `[64, 64]`;
если изменился СМЫСЛ значений в уже посчитанной серии — удали её папку в `results/` (или используй `--force`).
Если поменялся только конфиг (например бюджет), `sweep` сам увидит несовпадение с сохранённым `config.json`
и пересчитает запуск.

---

## 7. Структура проекта

    src/config.py       ExperimentConfig — один объект описывает запуск целиком
    src/experiments.py  ЧТО запускаем: серии гиперпараметров PPO и заготовка для A2C
    src/algos.py        реестр алгоритмов (ppo, a2c), запись фактических гиперпараметров
    src/envs.py         создание сред
    src/callbacks.py    оценка политики каждые N шагов -> кривая обучения + диагностика
    src/train.py        run_experiment(cfg): обучение и сохранение результатов
    src/sweep.py        параллельный запуск серий
    src/analyze.py      метрики, графики, сетки сочетаний, сравнение алгоритмов, нормировка наград сред
    src/stress_test.py  тест на прочность: обученные модели на эпизодах длиннее 500 шагов
    src/video.py        запись видео с подписями (mp4 / gif / webm), запись бок о бок
    src/render.py       видео для уже обученных моделей
    src/main.py         демо

Что сохраняется для каждого запуска:

    results/<algo>/<серия>/<конфигурация>/seed_<k>/
        config.json   конфиг + фактические гиперпараметры + версии библиотек
        eval.csv      кривая обучения: eval_sampled_* (основная метрика), eval_greedy_*, диагностика
        summary.json  итоговая оценка и время обучения
        model.zip     обученная модель
        policy.mp4/.gif   (если записывали видео)

---

## 8. Для сокомандника (A2C)

1. Открой `src/experiments.py`, раздел `A2C_SWEEPS`: подбери значения и дополни серии.
   Названия серий оставляй такими же, как у PPO, — тогда сравнение сработает автоматически.
2. Для серии `n_envs` держи постоянным размер rollout (`n_steps × n_envs`), как сделано у PPO.
3. Запуск: `python -m src.sweep --algo a2c --list`, затем `python -m src.sweep --algo a2c`.
4. Анализ: `python -m src.analyze`. Остальной код менять не нужно.
5. Бюджет и частота оценки (`COMMON`) должны остаться такими же, как у PPO.

---

## 9. Частые проблемы

| Сообщение | Причина и решение |
|---|---|
| `No module named src` | команда запущена не из корня проекта: перейди в папку с `src/` |
| `No module named stable_baselines3` (и т.п.) | не активировано окружение: `source venv/Scripts/activate` |
| `старый формат eval.csv` | результаты посчитаны старой версией кода: удали `results/` и перезапусти |
| `нет завершённых запусков` | в указанной папке нет результатов: проверь `--results` |
| Пустая папка `analysis_quick/` | остаток от запуска `analyze` с `--out analysis_quick`, когда результатов не было; её можно удалить |
| В `analysis/` мало графиков | посчитаны не все серии: графики серий и сводные рисунки строятся из того, что есть в `results/` |
| Запуск «завис» | параллельные запуски по 3–4 минуты это норма; прогресс печатается по мере завершения запусков |

---

## 10. Замечания об оценке

- **Какая оценка основная.** Для кривых и сравнения используется `eval_sampled_*` (действия сэмплируются из
  политики — это то, что реально оптимизируют PPO и A2C). `eval_greedy_*` (argmax) нужна для видео: у необученной
  сети greedy-политика у некоторых сидов случайно держит шест все 500 шагов, поэтому на кривых она вводит в
  заблуждение.
- **«Решил задачу»** = награда ≥ 475 держится 2 контрольные точки подряд (одиночный всплеск не считается).
- **Сид** фиксирует инициализацию сети и среду: одна и та же конфигурация с тем же сидом даёт одинаковый результат.
