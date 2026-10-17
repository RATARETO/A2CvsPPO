#!/usr/bin/env bash
# Видео-сравнения по каждому гиперпараметру: от минимального значения к максимальному (слева направо).
# Запуск из корня проекта при активированном окружении:
#     bash make_compare_videos.sh          # сид 0
#     bash make_compare_videos.sh 3        # другой сид (например 3)
# Результаты: папка videos/ (файлы compare_*_sampled.mp4 и .gif).
# Видео записываются с --sampled: действия сэмплируются, как в кривых обучения на графиках.

S="seed_${1:-0}"
R="results/ppo"
vid() { python -m src.render --compare "$@" --episodes 2 --sampled; }

echo "=== Блок A: минимум | дефолт | максимум ==="
vid $R/learning_rate/learning_rate=1e-05/$S  $R/learning_rate/learning_rate=0.0003/$S  $R/learning_rate/learning_rate=0.01/$S
vid $R/n_epochs/n_epochs=1/$S                $R/n_epochs/n_epochs=10/$S                $R/n_epochs/n_epochs=40/$S
vid $R/batch_size/batch_size=16/$S           $R/batch_size/batch_size=64/$S            $R/batch_size/batch_size=512/$S
vid $R/clip_range/clip_range=0.05/$S         $R/clip_range/clip_range=0.2/$S           $R/clip_range/clip_range=0.5/$S
vid $R/n_steps/n_steps=64/$S                 $R/n_steps/n_steps=2048/$S                $R/n_steps/n_steps=4096/$S
vid $R/gamma/gamma=0.9/$S                    $R/gamma/gamma=0.99/$S                    $R/gamma/gamma=0.999/$S
vid $R/gae_lambda/gae_lambda=0.8/$S          $R/gae_lambda/gae_lambda=0.95/$S          $R/gae_lambda/gae_lambda=1/$S
vid $R/ent_coef/ent_coef=0/$S                $R/ent_coef/ent_coef=0.01/$S              $R/ent_coef/ent_coef=0.05/$S   $R/ent_coef/ent_coef=0.1/$S
vid $R/max_grad_norm/max_grad_norm=0.1/$S    $R/max_grad_norm/max_grad_norm=0.5/$S     $R/max_grad_norm/max_grad_norm=5/$S
vid $R/vf_coef/vf_coef=0.1/$S                $R/vf_coef/vf_coef=0.5/$S                 $R/vf_coef/vf_coef=2/$S
vid $R/net_arch/net_arch=16x16/$S            $R/net_arch/net_arch=64x64/$S             $R/net_arch/net_arch=256x256/$S
vid $R/n_envs/n_envs=1,n_steps=2048/$S       $R/n_envs/n_envs=2,n_steps=1024/$S        $R/n_envs/n_envs=4,n_steps=512/$S  $R/n_envs/n_envs=8,n_steps=256/$S
vid $R/activation/activation=tanh/$S         $R/activation/activation=relu/$S

echo "=== Блок B: подробнее, 4 ступеньки от минимума к максимуму ==="
vid $R/learning_rate/learning_rate=1e-05/$S  $R/learning_rate/learning_rate=0.0001/$S  $R/learning_rate/learning_rate=0.001/$S  $R/learning_rate/learning_rate=0.01/$S
vid $R/n_epochs/n_epochs=1/$S                $R/n_epochs/n_epochs=3/$S                 $R/n_epochs/n_epochs=20/$S               $R/n_epochs/n_epochs=40/$S
vid $R/batch_size/batch_size=16/$S           $R/batch_size/batch_size=64/$S            $R/batch_size/batch_size=128/$S          $R/batch_size/batch_size=512/$S
vid $R/clip_range/clip_range=0.05/$S         $R/clip_range/clip_range=0.1/$S           $R/clip_range/clip_range=0.3/$S          $R/clip_range/clip_range=0.5/$S
vid $R/n_steps/n_steps=64/$S                 $R/n_steps/n_steps=256/$S                 $R/n_steps/n_steps=1024/$S               $R/n_steps/n_steps=4096/$S
vid $R/gamma/gamma=0.9/$S                    $R/gamma/gamma=0.95/$S                    $R/gamma/gamma=0.99/$S                   $R/gamma/gamma=0.999/$S
vid $R/gae_lambda/gae_lambda=0.8/$S          $R/gae_lambda/gae_lambda=0.9/$S           $R/gae_lambda/gae_lambda=0.99/$S         $R/gae_lambda/gae_lambda=1/$S

echo "Готово. Видео в папке videos/"
