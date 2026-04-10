# Deep Learning Enabled Semantic Communication Systems

PyTorch implementation of **Deep Learning Enabled Semantic Communication Systems** for text transmission over noisy channels.

This repository includes:
- text preprocessing for the Europarl corpus
- Transformer-based semantic transceiver model (`DeepSC`)
- channel simulation (AWGN, Rayleigh, Rician)
- training and BLEU-based evaluation scripts

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
- nltk
- w3lib
- tqdm
- scikit-learn

Install dependencies with one of the following:

```bash
pip install -r requirements.txt
```

or

```bash
pip install -e .
```

## Project Structure

- `preprocess_text.py`: build vocabulary and train/test pickle files from raw text
- `main.py`: model training
- `performance.py`: evaluation (BLEU over SNR points)
- `dataset.py`: dataset loading and batch collation
- `models/transceiver.py`: Transformer + channel encoder/decoder
- `models/mutual_info.py`: MINE network utilities
- `utils.py`: channels, masks, decoding, training/validation helpers

## Data Preparation

1. Place Europarl text files under:
   - `data/txt/en/`
2. Run preprocessing:

```bash
python preprocess_text.py --input-data-dir en
```

This creates:
- `data/train/europarl/train_data.pkl`
- `data/train/europarl/test_data.pkl`
- `data/train/europarl/vocab.json`

## Training

Run:

```bash
python main.py --channel Rayleigh
```

Common channels:
- `AWGN`
- `Rayleigh`
- `Rician`

By default, checkpoints are saved to:
- `checkpoints/deepsc-Rayleigh/`

You can change path/channel/hyperparameters through CLI flags in `main.py`.

## Evaluation

After training, run:

```bash
python performance.py --channel Rayleigh --checkpoint-path checkpoints/deepsc-Rayleigh
```

The evaluation script:
- loads the newest `.pth` checkpoint from the checkpoint directory
- computes BLEU-1 across SNR values `[0, 3, 6, 9, 12, 15, 18]`

## Notes

- `main.py` currently trains `DeepSC` without mutual-information regularization by default, although MI-related utilities are included.
- Ensure the preprocessed files exist before training or evaluation.
- If CUDA is available, scripts automatically use GPU (`cuda:0`), otherwise CPU.
