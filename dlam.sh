#!/usr/bin/env bash
# GTR + iTransformer on AIML-TUDA/dlam-ts-project-data-2026
#
# Hyperparameters follow scripts/Ablation/GTRiTransformer.sh from macovaseas/GTR
# (the electricity block -- hourly data, many channels, RevIN on):
#
#     e_layers 3   d_model 512   d_ff 512   batch_size 16
#     learning_rate 0.0005   train_epochs 30   patience 5   itr 1
#
# Dataset-specific changes from that block, and why:
#   --enc_in    321 -> 23   22 covariates + target, the channel count of a window
#   --pred_len  swept -> 336   fixed: it is the competition horizon
#   --label_len 0           Combined_Model is encoder-only (no decoder input)
#   --features  M -> MS     the task scores the `target` column only
#   --period_len 24         matches the GTRiTransformer default (not a CLI flag upstream)
#
# GTR picks `cycle` as one day or one week in steps: 24 for ETTh1 (hourly),
# 168 for electricity/traffic (hourly), 144 for weather (10-min). This data is
# hourly and its target autocorrelation is 0.44 at lag 24 vs 0.34 at lag 168,
# so both are worth a run -- the loop below tries each.

model_name=GTRiTransformer
backbone=iTransformer
model_id_name=DLAM
data_name=DLAM
est_horizon=336
pred_len=24

# try pred_len 336

mkdir -p ./logs/GTRiTransformer


seq_len=96

for cycle in 24 168 672
do
for random_seed in 2026
do
    tag=$model_id_name'_'$seq_len'_'$pred_len'_cycle'$cycle'_seed'$random_seed
    python -u run.py \
      --do_predict \
      --is_training 1 \
      --model_id $tag \
      --model $model_name \
      --backbone $backbone \
      --data $data_name \
      --features MS \
      --seq_len $seq_len \
      --label_len 0 \
      --pred_len $pred_len \
      --est_horizon $est_horizon \
      --enc_in 23 \
      --cycle $cycle \
      --period_len 24 \
      --use_revin 1 \
      --e_layers 3 \
      --n_heads 8 \
      --d_model 512 \
      --d_ff 512 \
      --dropout 0.1 \
      --embed timeF \
      --freq h \
      --lradj type3 \
      --pct_start 0.3 \
      --train_epochs 30 \
      --patience 5 \
      --num_workers 0 \
      --itr 1 --batch_size 16 --learning_rate 0.0005 --random_seed $random_seed \
      2>&1 | tee ./logs/GTRiTransformer/$tag.log
done
done


seq_len=336

for cycle in 24 168 672
do
for random_seed in 2026
do
    tag=$model_id_name'_'$seq_len'_'$pred_len'_cycle'$cycle'_seed'$random_seed
    python -u run.py \
      --do_predict \
      --is_training 1 \
      --model_id $tag \
      --model $model_name \
      --backbone $backbone \
      --data $data_name \
      --features MS \
      --seq_len $seq_len \
      --label_len 0 \
      --pred_len $pred_len \
      --est_horizon $est_horizon \
      --enc_in 23 \
      --cycle $cycle \
      --period_len 24 \
      --use_revin 1 \
      --e_layers 3 \
      --n_heads 8 \
      --d_model 512 \
      --d_ff 512 \
      --dropout 0.1 \
      --embed timeF \
      --freq h \
      --lradj type3 \
      --pct_start 0.3 \
      --train_epochs 30 \
      --patience 5 \
      --num_workers 0 \
      --itr 1 --batch_size 16 --learning_rate 0.0005 --random_seed $random_seed \
      2>&1 | tee ./logs/GTRiTransformer/$tag.log
done
done
