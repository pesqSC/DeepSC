# Deep Learning Enabled Semantic Communication Systems

PyTorch implementation of **Deep Learning Enabled Semantic Communication Systems (DeepSC)** for text transmission over noisy channels, with both baseline training and receiver-side knowledge distillation workflows.

## Reference

- Huiqiang Xie, Zhijin Qin, Geoffrey Ye Li, and Biing-Hwang Juang,  
  "Deep Learning Enabled Semantic Communication Systems,"  
  *IEEE Transactions on Signal Processing*, 2021.

```bibtex
@article{xie2021deep,
  author  = {H. Xie and Z. Qin and G. Y. Li and B.-H. Juang},
  title   = {Deep Learning Enabled Semantic Communication Systems},
  journal = {IEEE Transactions on Signal Processing},
  year    = {2021}
}
```

## Environment

- Python `>= 3.12` (from `pyproject.toml`)
- PyTorch
- numpy
- nltk
- tqdm
- w3lib
- scikit-learn

Install dependencies:

```bash
pip install -e .
```

## Project Structure

- `preprocess_text.py`: builds vocabulary and train/test pickle files from raw text.
- `main.py`: baseline DeepSC teacher training.
- `performance.py`: BLEU-based evaluation across SNR points.
- `train_tr_kd.py`: receiver-only KD (teacher TX fixed, student TR trained).
- `train_student.py`: single-student TR KD experiment.
- `train_multi_students.py`: multi-student TR KD experiment.
- `teacher.py`: teacher/student model builders and KD utilities.
- `student.py`: student receiver architecture.
- `models/transceiver.py`: DeepSC backbone.
- `models/tx_model.py`: transmitter split module.
- `models/rx_model.py`: receiver split module.
- `utils.py`: masks, channels, decoding, losses, and save helpers.

## Data Preparation

1. Place Europarl text files under `data/txt/en/`.
2. Run preprocessing:

```bash
python preprocess_text.py --input-data-dir en
```

Expected outputs:

- `data/train/europarl/train_data.pkl`
- `data/train/europarl/test_data.pkl`
- `data/train/europarl/vocab.json`

## Baseline Teacher Training

Train DeepSC with:

```bash
python main.py --channel Rayleigh --checkpoint-path checkpoints/deepsc-Rayleigh
```

Supported channels:

- `AWGN`
- `Rayleigh`
- `Rician`

## Receiver KD Training

`train_tr_kd.py` trains only the receiver-side student while using a fixed teacher checkpoint.

Example:

```bash
python train_tr_kd.py \
  --teacher-checkpoint checkpoints/deepsc-Rayleigh/checkpoint_100.pth \
  --channel Rayleigh \
  --save-dir checkpoints/tr_kd
```

Main KD hyperparameters:

- `--temperature`
- `--alpha` (CE weight)
- `--beta` (KD KL weight)
- `--gamma` (feature distillation weight)
- `--snr-mode {fixed,range}`

## Multi-Student KD

For parallel student receiver training:

```bash
python train_multi_students.py \
  --teacher-checkpoint checkpoints/deepsc-Rayleigh/checkpoint_100.pth \
  --channel Rayleigh \
  --save-dir checkpoints/tr_kd_multi
```

This script saves per-student checkpoints and best models under the configured `--save-dir`.

## Evaluation

Evaluate a trained checkpoint with:

```bash
python performance.py --channel Rayleigh --checkpoint-path checkpoints/deepsc-Rayleigh
```

The script loads the newest checkpoint in the target directory and computes BLEU over SNR values such as `[0, 3, 6, 9, 12, 15, 18]`.

## Notes

- Ensure preprocessing is completed before any training or evaluation run.
- CUDA is used automatically when available; otherwise CPU is used.
- Some scripts still contain research/experimental code paths; verify defaults before long runs.
