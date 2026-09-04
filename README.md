# Multivariate Time Series Forecasting: GTR + iTransformer

Global Temporal Retriever (GTR) with an inverted Transformer (iTransformer) backbone for the
DLAM hourly forecasting task. The model forecasts a target operational-load index for 96 units
over a 336-hour horizon from a 96-hour look-back and a set of known-future covariates, producing
the horizon with an autoregressive 24-hour rollout.

## Repository layout

```
run.py                     training / inference entrypoint (argument parsing)
exp.py                     train / validate / test / predict loops
data_provider.py           DLAM_Dataset: data loading, scaling, windows, cycle index
preprocessing.py           autocorrelation (ACF) analysis to choose the cycle length
models/
  combined_model.py        GTR + backbone (RevIN optional)
  GTR.py                   Global Temporal Retriever (cyclic pattern bank + conv fusion)
  DLinear.py               decomposition-linear backbone (comparison)
  iTransformer_imp/        inverted Transformer backbone
metrics.py, tools.py, timefeatures.py   metrics, early stopping / LR schedule, time features
dlam.sh                    cycle-length sweep
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
python -m pip install torch pandas numpy scikit-learn statsmodels scipy matplotlib huggingface_hub
```

The dataset is read from `hf://datasets/AIML-TUDA/dlam-ts-project-data-2026` (via
`huggingface_hub`), so both training and inference need network access to fetch the data. Each
unit is hourly with 22 covariates and one target channel (23 channels total). Covariates are
standardised with statistics fit on the training split only.

## Cycle length

The cycle length is chosen from an autocorrelation analysis rather than fixed by hand:

```bash
python preprocessing.py
```

The daily (24) and weekly (168) lags dominate; 24, 168, and 672 were tested and 24 was best. The
chosen value is passed to training with `--cycle`.

## Training

Best configuration (cycle 24, learning rate 1e-4, RevIN off):

```bash
python -u run.py --is_training 1 --model_id DLAM_96_24_cycle24 \
  --model GTRiTransformer --backbone iTransformer --data DLAM --features MS \
  --seq_len 96 --label_len 0 --pred_len 24 --est_horizon 336 --enc_in 23 \
  --cycle 24 --period_len 24 --use_revin 0 \
  --dropout 0.1 --lradj type3 --learning_rate 1e-4 \
  --train_epochs 30 --patience 5 --batch_size 16 --random_seed 2024
```

Weights are written to `checkpoints/<setting>/checkpoint.pth`, where `<setting>` is built from the
arguments above. Swap the backbone with `--backbone DLinear` (use `--batch_size 32` for DLinear)
to train the comparison model. Run the full cycle sweep with `bash dlam.sh` (Git Bash or WSL on
Windows).

Key hyperparameters: look-back 96, prediction block 24, horizon 336, 23 input channels, cycle 24
(from the ACF analysis), `d_model` 512, `d_ff` 512, 3 encoder layers, 8 heads, dropout 0.1 in
both GTR and the backbone, RevIN off, Huber loss (delta 0.5), Adam, learning rate 1e-4 with the
`type3` schedule, batch size 16, gradient clipping at norm 1.0, early stopping patience 5. Seeds
for Python, NumPy, and PyTorch are fixed at 2024.

## Inference

Load a trained checkpoint and generate the 336-hour forecast with the autoregressive rollout.
Reuse the same flags as training so the `<setting>` matches the saved checkpoint, and add
`--is_training 0 --do_predict`:

```bash
python -u run.py --is_training 0 --do_predict --model_id DLAM_96_24_cycle24 \
  --model GTRiTransformer --backbone iTransformer --data DLAM --features MS \
  --seq_len 96 --label_len 0 --pred_len 24 --est_horizon 336 --enc_in 23 \
  --cycle 24 --period_len 24 --use_revin 0 \
  --batch_size 16 --random_seed 2024
```

This loads `checkpoints/<setting>/checkpoint.pth`, prints the test metrics (WAPE, MAE, MSE) on the
internal validation split, rolls the 24-hour block forward to fill the 336-hour horizon while
feeding the predicted target back and reading the known future covariates at each step, and writes
the forecasts under `results/<setting>/`. To train and forecast in one run, add `--do_predict` to
the training command above instead.

## Results

Table 2 reports validation results on the internal split (target channel, 336-hour horizon by
autoregressive rollout). The best configuration uses a daily cycle and a learning rate of 1e-4.
Lower is better for all metrics.

| Method | WAPE | MAE | MSE |
|---|---|---|---|
| Naive last value baseline | 61.35 | 5.29 | 48.61 |
| GTR + DLinear backbone | 33.67 | 3.70 | 25.70 |
| **GTR + iTransformer backbone (ours)** | **30.69** | **3.37** | **22.87** |

## Reproducibility

- Model implemented in PyTorch.
- Random seeds fixed at 2024 for Python, NumPy, and PyTorch.
- All important hyperparameters are listed above and set through `run.py` flags.
