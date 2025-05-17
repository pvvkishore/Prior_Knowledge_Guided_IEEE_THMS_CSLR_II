#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 14 16:40:46 2025

@author: pvvkishore
"""

# Sign Language Visual-Text Alignment Model

This repository contains the implementation of a sign language model that aligns visual features from sign language videos with text gloss embeddings using contrastive learning.

## Overview

The model consists of two main streams:

1. **Visual Stream**:
   - Takes sequence of video frames of shape (T, 3, 224, 224)
   - Resizes frames to 112x112
   - Applies dynamic positional encoding to each frame
   - Processes through a ResNet18 with Multi-Head Attention per ResBlock (heads: 2→4→4→8)
   - Extracts framewise features and processes through a Bi-LSTM
   - Applies L2 normalization to visual features

2. **Text Stream**:
   - Takes gloss sentences as input
   - Processes through BERT (frozen by default)
   - Uses [CLS] token for gloss representation
   - Projects to the same dimension as visual features (D=1024)
   - Applies L2 normalization to text features

The model is trained using contrastive learning (NTXent loss) with cosine similarity between visual and text features.

## Files Structure

- `sign_language_training.py`: Main model architecture and training code
- `run_training.py`: Script to run the training process
- `feature_extraction.py`: Script to extract features and evaluate alignment
- `run_extraction.py`: Script to run feature extraction
- `analyze_results.py`: Script to analyze alignment results
- `data_loader.py`: Data loading utilities (provided externally)

## Requirements

- Python 3.7+
- PyTorch 1.8+
- torchvision
- transformers (for BERT)
- pandas
- numpy
- tqdm
- matplotlib
- seaborn

## Installation

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd <repo-directory>
   ```

2. Install dependencies:
   ```bash
   pip install torch torchvision transformers pandas numpy tqdm matplotlib seaborn
   ```

## Usage

### Training the Model

To train the model using default paths from data_loader.py:

```bash
python run_training.py
```

With custom parameters:

```bash
python run_training.py --batch_size 8 --epochs 50 --learning_rate 2e-4
```

To use custom data paths:

```bash
python run_training.py --csv_path "/path/to/gloss_csv.csv" --frames_root_dir "/path/to/frames/"
```

#### Training Parameters

- `--csv_path`: Path to the CSV file with gloss annotations (optional, uses default from data_loader.py)
- `--frames_root_dir`: Root directory containing frame images (optional, uses default from data_loader.py)
- `--batch_size`: Batch size for training (default: 4)
- `--num_workers`: Number of workers for data loading (default: 4)
- `--epochs`: Number of training epochs (default: 100)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--weight_decay`: Weight decay for optimizer (default: 1e-5)
- `--temperature`: Temperature for contrastive loss (default: 0.07)
- `--hidden_dim`: Hidden dimension for features (default: 1024)
- `--bert_model`: BERT model name (default: 'bert-base-uncased')
- `--freeze_bert`: Flag to freeze BERT parameters
- `--save_path`: Directory to save models (default: './saved_models')
- `--validate_files`: Flag to validate if frame files exist

### Feature Extraction and Evaluation

To extract features from the trained model:

```bash
python feature_extraction.py --model_path "saved_models/best_model_epoch_100.pth"
```

Or using run_extraction.py:

```bash
python run_extraction.py --model_path "saved_models/best_model_epoch_100.pth"
```

With custom parameters:

```bash
python feature_extraction.py --model_path "saved_models/best_model_epoch_100.pth" --csv_path "/path/to/gloss_csv.csv" --frames_root_dir "/path/to/frames/" --batch_size 8
```

#### Feature Extraction Parameters

- `--model_path`: Path to the trained model (required)
- `--csv_path`: Path to the CSV file with gloss annotations (optional)
- `--frames_root_dir`: Root directory containing frame images (optional)
- `--output_dir`: Directory to save results (default: './results')
- `--batch_size`: Batch size for data loading (default: 4)
- `--num_workers`: Number of workers for data loading (default: 4)
- `--mode`: Which features to extract ('visual', 'text', or 'both') (default: 'both')

### Analyzing Results

To analyze the alignment results:

```bash
python analyze_results.py --results_file "results/alignment_scores.txt" --output_dir "results/analysis"
```

## Model Architecture Details

### Visual Stream

1. **Input**: Sequence of video frames (T, 3, 224, 224)
2. **Resize**: Frames are resized to (T, 3, 112, 112)
3. **Dynamic Positional Encoding**: Applied to each frame
4. **ResNet18 with Multi-Head Attention**:
   - Layer 1: 2 attention heads
   - Layer 2: 4 attention heads
   - Layer 3: 4 attention heads
   - Layer 4: 8 attention heads
5. **Framewise Feature Extraction**: Shape (T, 1024)
6. **Bi-LSTM**: Processes features to capture temporal context
7. **L2 Normalization**: Applied to output features

### Text Stream

1. **Input**: Gloss sentence (tokenized)
2. **BERT Processing**: Generates contextual embeddings
3. **[CLS] Token Extraction**: Used as the sentence representation
4. **Projection Layer**: Maps to dimension 1024
5. **L2 Normalization**: Applied to output features

### Training Process

1. **Batch Formation**: Random sampling of visual-text pairs
2. **Forward Pass**:
   - Encode frames → visual embedding
   - Encode gloss text via BERT → text embedding
   - Compute similarity matrix across batch
3. **Loss Calculation**: Apply contrastive loss (NTXent)
4. **Backpropagation**: Update visual backbone only (keep BERT frozen)

## Data Format

The model expects data in the format provided by the data_loader.py script, which returns:

- `frames`: Tensor of shape [batch_size, max_frames, 3, 224, 224]
- `frame_mask`: Boolean mask for valid frames
- `gloss1` and `gloss2`: Lists of gloss text strings
- `folder_names`: List of folder names
- `num_frames`: Tensor containing the number of valid frames for each sample

## Results and Evaluation

The feature extraction process will:
1. Extract visual and text features from the trained model
2. Calculate alignment scores between visual and text features
3. Generate detailed reports and visualizations

The analysis script produces:
- Histograms of similarity scores
- Box plots comparing alignments
- Scatter plots showing correlation between gloss1 and gloss2 similarity
- List of best and worst performing samples

## Default Paths

The code uses the following default paths if none are provided:
- CSV Path: "/home/pvvkishore/Desktop/IEEE_THMS_May10/Phonix_Train/train_gloss_eng.csv"
- Frames Root Directory: "/home/pvvkishore/Desktop/IEEE_THMS_May10/Phonix_Train/train/"

## Troubleshooting

If you encounter errors related to data loading:
1. Make sure the paths to the CSV file and frames directory are correct
2. Verify that the CSV format matches what the data_loader.py expects
3. Check if frame images exist in the specified directories

For any "TypeError: create_dataloader() missing 2 required positional arguments" error:
- Make sure to always provide both csv_path and frames_root_dir parameters

## Citation

If you use this code, please cite our work:

```
@article{sign_language_model,
  title={Visual-Text Alignment for Sign Language Using Contrastive Learning},
  author={Your Name},
  journal={Conference/Journal Name},
  year={2023}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This project builds on several existing works in sign language processing and visual-text alignment.