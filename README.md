# Pretrained Character-Level Masked Language Model

This repository contains the implementation and pretrained weights for a character-level masked language model designed to predict masked content within noisy or incomplete text.
The model can be fine-tuned on downstream tasks such as text reconstruction, infilling, or metadata prediction.

## Features

- Pretrained on a large corpus of raw character data.
- Implements character masking with two special tokens: `[MASK]` for masked regions and `[PAD]` for padding.
- Configurable block size and vocabulary.
- Supports fine-tuning and inference on custom datasets.
- Built with PyTorch.

## Usage
### Pretraining
Pretrain the model on a large character corpus:
```sh
python train.py --data_file path/to/pretraining_data.txt
```
### Fine-tuning
Fine-tune the pretrained model on a downstream task:
```sh
python train.py --data_file path/to/fine_tune.txt --pretrained_model vanilla.pt
```
### Testing
Run predictions on a test set:
```sh
python test.py --model vanilla.finetune.pt --test_file test_data.txt
```
Predictions will be saved to:
vanilla.finetune.test.predictions
## Model Architecture
1. Character-level tokenization (256-character vocab).
2. Transformer-based encoder.
3. Masked Language Modeling (MLM) loss objective.
4. Positional embeddings for sequence structure.

## Evaluation
The pretrained model achieves over 10% accuracy on masked prediction tasks without fine-tuning — significantly better than random (≈ 0.4%).
Fine-tuning on task-specific data further improves performance.
