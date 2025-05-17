# Continuous Sign Language Recognition (CSLR) Pipeline

This repository contains a complete end-to-end pipeline for Continuous Sign Language Recognition (CSLR) using a two-stage training approach:

1. **Stage 1**: Visual-text contrastive learning for sign language features
2. **Stage 2**: CSLR recognition with CTC loss for gloss sequence prediction

## Architecture Overview

The CSLR system consists of the following key components:

- **Visual Stream**: ResNet18 with Multi-Head Attention (MHA) and BiLSTM
- **Text Stream**: BERT-based gloss embeddings
- **Cross-Modal Attention**: Alignment between visual and text features
- **CTC Classifier**: Predicts gloss sequences from video frames

## Requirements

- Python 3.7+
- PyTorch 1.10+
- Transformers (Hugging Face)
- OpenCV
- NumPy
- Matplotlib
- Tqdm
- Levenshtein
- Seaborn
- Pandas

## File Structure

```
.
├── data_loader.py                # Data loading utilities
├── sign_language_training.py     # Stage 1 training (contrastive learning)
├── cslr_model.py                 # Stage 2 CSLR model with CTC loss
├── run_training.py               # Script to run Stage 1 training
├── evaluate_cslr.py              # Evaluation utilities
├── inference.py                  # Inference utilities
├── run_pipeline.py               # End-to-end pipeline script
└── README.md                     # Documentation
```

## Training Pipeline

### 1. Stage 1: Visual-Text Contrastive Learning

In this stage, we train a model to learn the association between sign language videos and their corresponding gloss annotations using contrastive learning.

```bash
python sign_language_training.py \
    --csv_path=/path/to/train_gloss_eng.csv \
    --frames_root_dir=/path/to/frames/ \
    --batch_size=4 \
    --epochs=100 \
    --learning_rate=1e-4 \
    --save_path=./saved_models/stage1 \
    --validate_files
```

### 2. Stage 2: CSLR Model with CTC Loss

In this stage, we build on the pretrained Stage 1 model and add CTC loss to enable sequence prediction.

```bash
python cslr_model.py \
    --pretrained_model=./saved_models/stage1/final_model.pth \
    --csv_path=/path/to/train_gloss_eng.csv \
    --frames_root_dir=/path/to/frames/ \
    --batch_size=4 \
    --epochs=50 \
    --learning_rate=1e-3 \
    --save_path=./saved_models/cslr \
    --phase=1 \
    --freeze_visual \
    --freeze_text \
    --unfreeze_attention
```

#### Phase 1 vs Phase 2

- **Phase 1**: Freeze visual and text backbones, only train cross-modal attention and CTC classifier
- **Phase 2**: Fine-tune parts of the visual backbone and optionally the text encoder

### 3. Evaluation

Evaluate the trained CSLR model on a test set:

```bash
python evaluate_cslr.py \
    --model_path=./saved_models/cslr/final_cslr_model.pth \
    --csv_path=/path/to/test_gloss_eng.csv \
    --frames_root_dir=/path/to/test_frames/ \
    --save_dir=./evaluation_results
```

### 4. Inference on Video

Run inference on a single video file:

```bash
python inference.py \
    --model_path=./saved_models/cslr/final_cslr_model.pth \
    --video_path=/path/to/video.mp4 \
    --save_dir=./inference_results
```

## Complete Pipeline

You can run the entire pipeline using the `run_pipeline.py` script:

```bash
python run_pipeline.py \
    --start_stage=1 \
    --end_stage=4 \
    --train_csv=/path/to/train_gloss_eng.csv \
    --train_frames=/path/to/frames/ \
    --test_csv=/path/to/test_gloss_eng.csv \
    --test_frames=/path/to/test_frames/ \
    --video_path=/path/to/video.mp4 \
    --stage1_epochs=100 \
    --stage2_epochs=50 \
    --batch_size=4
```

## Model Training Strategy

The two-stage training strategy is designed to:

1. **Stage 1**: Learn robust visual-text representations through contrastive learning
2. **Stage 2**: Use these representations to train a sequence model with CTC loss

### Key Architectural Features:

- **Dynamic Positional Encoding**: Enhances temporal information in frame sequences
- **Multi-Head Attention**: Captures spatial relationships within frames
- **Cross-Modal Attention**: Aligns visual features with text embeddings
- **BiLSTM**: Models temporal context across frames
- **CTC Loss**: Enables sequence prediction without frame-level annotations

## Performance Metrics

The model is evaluated using the following metrics:

- **Word Error Rate (WER)**: The primary metric for sequence prediction
- **Word Recognition Rate**: Accuracy of recognized gloss tokens
- **Substitution, Insertion, Deletion Rates**: Detailed error analysis

## Data Format

The system expects the following data format:

- **CSV file**: Contains frame folder paths and corresponding gloss annotations
- **Frames directory**: Contains subfolders with image frames for each sign video
- **Frame naming**: Numbered frame images (e.g., 1.jpg, 2.jpg, etc.)

## Visualization

The evaluation includes several visualizations:

- **Confusion Matrix**: For the most common gloss tokens
- **Error Distribution**: Breakdown of correct, substitution, insertion, and deletion errors
- **Attention Visualization**: Shows cross-modal attention between frames and gloss tokens

## Memory Requirements

- Training requires a GPU with at least 8GB of memory for default settings
- Batch size can be reduced for GPUs with less memory
- Inference can be run on CPU, but will be slower

## Troubleshooting

- If you encounter memory issues, try reducing batch size or frame resolution
- For improved performance, make sure all data files exist and are correctly referenced
- If validation fails due to missing files, use the `--validate_files` flag to enable robust error handling

## Acknowledgments

The architecture is based on state-of-the-art approaches for sign language recognition, combining visual-text contrastive learning with CTC-based sequence modeling.