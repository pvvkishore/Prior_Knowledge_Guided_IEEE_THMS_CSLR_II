#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 14 14:46:49 2025

@author: pvvkishore
"""

import os
import sys
import argparse
import logging
import datetime
import subprocess
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'logs/training_log_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

def run_training(args):
    """
    Run the training process
    
    Args:
        args: Command line arguments
    """
    # Create necessary directories
    os.makedirs('saved_models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Set PYTHONPATH
    sys.path.append(os.getcwd())
    
    # Import the training module
    from sign_language_training import main as training_main
    
    # Log the start of training
    logger.info("Starting Sign Language Model training")
    logger.info(f"Arguments: {args}")
    
    # Run the training
    training_main()
    
    logger.info("Training completed")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run Sign Language Model training')
    
    # Data paths
    parser.add_argument('--csv_path', type=str, 
                        default="/home/pvvkishore/Desktop/IEEE_THMS_May10/Phonix_Train/train_gloss_eng.csv",
                        help='Path to the CSV file')
    parser.add_argument('--frames_root_dir', type=str, 
                        default="/home/pvvkishore/Desktop/IEEE_THMS_May10/Phonix_Train/train/",
                        help='Root directory containing frames')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--temperature', type=float, default=0.07, help='Temperature for contrastive loss')
    
    # Model parameters
    parser.add_argument('--hidden_dim', type=int, default=1024, help='Hidden dimension')
    parser.add_argument('--bert_model', type=str, default='bert-base-uncased', help='BERT model name')
    parser.add_argument('--freeze_bert', action='store_true', help='Freeze BERT parameters')
    
    # Saving parameters
    parser.add_argument('--save_path', type=str, default='./saved_models', help='Path to save models')
    parser.add_argument('--validate_files', action='store_true', help='Validate if frame files exist')
    
    return parser.parse_args()

if __name__ == '__main__':
    # Parse arguments
    args = parse_arguments()
    
    # Set command line arguments for the training script
    sys.argv = [
        'sign_language_training.py',
        f'--csv_path={args.csv_path}',
        f'--frames_root_dir={args.frames_root_dir}',
        f'--batch_size={args.batch_size}',
        f'--num_workers={args.num_workers}',
        f'--epochs={args.epochs}',
        f'--learning_rate={args.learning_rate}',
        f'--weight_decay={args.weight_decay}',
        f'--temperature={args.temperature}',
        f'--hidden_dim={args.hidden_dim}',
        f'--bert_model={args.bert_model}',
        f'--save_path={args.save_path}'
    ]
    
    if args.freeze_bert:
        sys.argv.append('--freeze_bert')
    if args.validate_files:
        sys.argv.append('--validate_files')
    
    # Run training
    run_training(args)