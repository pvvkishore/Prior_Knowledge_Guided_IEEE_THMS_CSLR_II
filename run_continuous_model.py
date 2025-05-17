#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
run_cslr_train.py - Simplified script to train Continuous Sign Language Recognition models
This script provides an easy way to train CSLR models using the pretrained visual-text feature extractor.
"""

import os
import logging
import argparse
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import time
import json
import matplotlib.pyplot as plt
from pathlib import Path

# Import the necessary modules from run_continuous_model.py
# When using this script, ensure run_continuous_model.py is in the same directory
from run_continuous_model import (
    ContinuousSignLanguageModel,
    build_vocabulary,
    evaluate_continuous_model,
    plot_training_history,
    device
)

# Import data loader
from data_loader import create_dataloader

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_cslr_model(args):
    """
    Train a Continuous Sign Language Recognition model using a pretrained feature extractor
    
    Args:
        args: Command line arguments
    """
    # Print GPU information
    if torch.cuda.is_available():
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        logger.info(f"Memory cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    else:
        logger.info("Using CPU")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the pretrained model
    logger.info(f"Loading pretrained model from {args.pretrained_model}")
    try:
        checkpoint = torch.load(args.pretrained_model, map_location=device)
        
        # Import the model after checking that we can load the checkpoint
        from sign_language_training import SignLanguageModel
        pretrained_model = SignLanguageModel(hidden_dim=1024, freeze_bert=True)
        pretrained_model.load_state_dict(checkpoint['model_state_dict'])
        pretrained_model = pretrained_model.to(device)
        logger.info("Pretrained model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load pretrained model: {e}")
        return
    
    # Create training dataloader
    logger.info("Creating training dataloader...")
    try:
        train_loader = create_dataloader(
            csv_path=args.csv_path,
            frames_root_dir=args.frames_root_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            validate_files=True
        )
        logger.info(f"Training dataloader created with {len(train_loader.dataset)} samples")
    except Exception as e:
        logger.error(f"Failed to create training dataloader: {e}")
        return
    
    # Create validation dataloader if provided
    val_loader = None
    val_csv_path = args.val_csv_path if args.val_csv_path else args.csv_path.replace('train', 'val')
    val_frames_dir = args.val_frames_dir if args.val_frames_dir else args.frames_root_dir.replace('train', 'val')
    
    if os.path.exists(val_csv_path):
        logger.info(f"Found validation CSV: {val_csv_path}")
        try:
            val_loader = create_dataloader(
                csv_path=val_csv_path,
                frames_root_dir=val_frames_dir,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                shuffle=False,
                validate_files=True
            )
            logger.info(f"Validation dataloader created with {len(val_loader.dataset)} samples")
        except Exception as e:
            logger.error(f"Failed to create validation dataloader: {e}")
            logger.info("Training will continue without validation")
    else:
        # Create a validation split from training data
        logger.info("No validation CSV found. Creating a validation subset from training data.")
        
        # We'll use a simple approach - just take the last validation_ratio of the dataset
        dataset_size = len(train_loader.dataset)
        val_size = int(dataset_size * args.validation_ratio)
        train_size = dataset_size - val_size
        
        # Create new datasets with manual splits
        # Note: Since we can't use sampler with the original create_dataloader, we need a different approach
        
        # We'll create a temporary CSV with just the training indices
        import pandas as pd
        import numpy as np
        
        try:
            # Read the original CSV
            df = pd.read_csv(args.csv_path)
            
            # Shuffle the dataframe
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)
            
            # Split into train and validation
            train_df = df.iloc[:train_size]
            val_df = df.iloc[train_size:]
            
            # Create temporary CSV files
            temp_train_csv = os.path.join(args.output_dir, 'temp_train.csv')
            temp_val_csv = os.path.join(args.output_dir, 'temp_val.csv')
            
            train_df.to_csv(temp_train_csv, index=False)
            val_df.to_csv(temp_val_csv, index=False)
            
            logger.info(f"Created temporary training CSV with {len(train_df)} samples")
            logger.info(f"Created temporary validation CSV with {len(val_df)} samples")
            
            # Create new dataloaders
            train_loader = create_dataloader(
                csv_path=temp_train_csv,
                frames_root_dir=args.frames_root_dir,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                shuffle=True,
                validate_files=True
            )
            
            val_loader = create_dataloader(
                csv_path=temp_val_csv,
                frames_root_dir=args.frames_root_dir,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                shuffle=False,
                validate_files=True
            )
            
            logger.info(f"Created training dataloader with {len(train_loader.dataset)} samples")
            logger.info(f"Created validation dataloader with {len(val_loader.dataset)} samples")
            
        except Exception as e:
            logger.error(f"Failed to create validation split: {e}")
            logger.info("Training will continue with the full dataset and no validation")
            # Restore the original train_loader just in case
            train_loader = create_dataloader(
                csv_path=args.csv_path,
                frames_root_dir=args.frames_root_dir,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                shuffle=True,
                validate_files=True
            )
    
    # Build vocabulary from training data
    logger.info("Building vocabulary from training data...")
    try:
        vocab_to_idx, idx_to_vocab = build_vocabulary(
            train_loader,
            min_freq=args.min_token_freq,
            special_tokens=['<PAD>', '<UNK>', '< SOS >', '<EOS>']
        )
        logger.info(f"Vocabulary built with {len(vocab_to_idx)} tokens")
        
        # Save vocabulary
        vocab_file = os.path.join(args.output_dir, 'vocabulary.json')
        with open(vocab_file, 'w') as f:
            json.dump({
                'vocab_to_idx': vocab_to_idx,
                'idx_to_vocab': idx_to_vocab
            }, f)
        logger.info(f"Vocabulary saved to {vocab_file}")
    except Exception as e:
        logger.error(f"Failed to build vocabulary: {e}")
        return
    
    # Initialize the CSLR model
    logger.info("Initializing CSLR model...")
    try:
        model = ContinuousSignLanguageModel(
            pretrained_model=pretrained_model,
            vocab_size=len(vocab_to_idx),
            hidden_dim=args.hidden_dim,
            embedding_dim=args.embedding_dim,
            freeze_resnet=args.freeze_resnet,
            freeze_bert=args.freeze_bert,
            alpha=args.alpha,
            beta=args.beta,
            num_heads=args.num_heads,
            dropout=args.dropout
        )
        model = model.to(device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model initialized with {total_params:,} total parameters")
        logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    except Exception as e:
        logger.error(f"Failed to initialize CSLR model: {e}")
        return
    
    # Initialize optimizer
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Setup model save paths
    best_model_path = os.path.join(args.output_dir, 'best_model.pth')
    final_model_path = os.path.join(args.output_dir, 'final_model.pth')
    
    # Training history
    history = {
        'loss': [],
        'contrastive_loss': [],
        'ctc_loss': []
    }
    
    if val_loader is not None:
        history.update({
            'val_loss': [],
            'val_contrastive_loss': [],
            'val_ctc_loss': [],
            'val_wer': []
        })
    
    # Training loop
    logger.info(f"Starting training for {args.epochs} epochs...")
    best_loss = float('inf')
    best_wer = float('inf')
    
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        total_loss = 0
        total_contrastive_loss = 0
        total_ctc_loss = 0
        batch_count = 0
        
        start_time = time.time()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Get data
            frames = batch['frames'].to(device)
            frame_mask = batch['frame_mask'].to(device)
            gloss1 = batch['gloss1']  # List of gloss strings
            
            # Tokenize target glosses for CTC loss
            target_sequences = []
            target_lengths = []
            
            for gloss in gloss1:
                # Split gloss string into tokens and convert to indices
                tokens = gloss.split()
                indices = [vocab_to_idx.get(token, vocab_to_idx['<UNK>']) for token in tokens]
                target_sequences.extend(indices)
                target_lengths.append(len(indices))
                
            targets = torch.tensor(target_sequences, dtype=torch.long, device=device)
            target_lengths = torch.tensor(target_lengths, dtype=torch.long, device=device)
            
            # Get input lengths from frame mask
            input_lengths = frame_mask.sum(dim=1).long()
            
            # Forward pass
            outputs = model(frames, frame_mask, gloss1)
            
            # Calculate loss
            loss, contrastive_loss, ctc_loss = model.calculate_total_loss(
                outputs, targets, input_lengths, target_lengths
            )
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            if args.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                
            optimizer.step()
            
            # Update statistics
            total_loss += loss.item()
            total_contrastive_loss += contrastive_loss.item()
            total_ctc_loss += ctc_loss.item()
            batch_count += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Cont": f"{contrastive_loss.item():.4f}",
                "CTC": f"{ctc_loss.item():.4f}"
            })
            
            # Log memory usage periodically
            if batch_idx % 50 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        
        # Calculate average losses
        avg_loss = total_loss / batch_count if batch_count > 0 else float('inf')
        avg_contrastive_loss = total_contrastive_loss / batch_count if batch_count > 0 else float('inf')
        avg_ctc_loss = total_ctc_loss / batch_count if batch_count > 0 else float('inf')
        epoch_time = time.time() - start_time
        
        # Update history
        history['loss'].append(avg_loss)
        history['contrastive_loss'].append(avg_contrastive_loss)
        history['ctc_loss'].append(avg_ctc_loss)
        
        logger.info(f"Epoch {epoch+1}/{args.epochs}, "
                  f"Loss: {avg_loss:.4f}, "
                  f"Contrastive: {avg_contrastive_loss:.4f}, "
                  f"CTC: {avg_ctc_loss:.4f}, "
                  f"Time: {epoch_time:.2f}s")
        
        # Evaluation phase
        if val_loader is not None:
            metrics = evaluate_continuous_model(model, val_loader, vocab_to_idx, idx_to_vocab, device)
            
            # Update history
            history['val_loss'].append(metrics['loss'])
            history['val_contrastive_loss'].append(metrics['contrastive_loss'])
            history['val_ctc_loss'].append(metrics['ctc_loss'])
            history['val_wer'].append(metrics['wer'])
            
            # Save the best model (based on validation WER if specified, otherwise validation loss)
            if args.save_best_wer:
                # Save based on WER
                current_wer = metrics['wer']
                if current_wer < best_wer:
                    best_wer = current_wer
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'vocab_to_idx': vocab_to_idx,
                        'idx_to_vocab': idx_to_vocab,
                        'metrics': metrics,
                        'alpha': model.alpha,
                        'beta': model.beta
                    }, best_model_path)
                    logger.info(f"Best model saved with validation WER: {best_wer:.4f}")
            else:
                # Save based on loss
                current_loss = metrics['loss']
                if current_loss < best_loss:
                    best_loss = current_loss
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'vocab_to_idx': vocab_to_idx,
                        'idx_to_vocab': idx_to_vocab,
                        'metrics': metrics,
                        'alpha': model.alpha,
                        'beta': model.beta
                    }, best_model_path)
                    logger.info(f"Best model saved with validation loss: {best_loss:.4f}")
        else:
            # Save based on training loss if no validation set
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'vocab_to_idx': vocab_to_idx,
                    'idx_to_vocab': idx_to_vocab,
                    'loss': avg_loss,
                    'alpha': model.alpha,
                    'beta': model.beta
                }, best_model_path)
                logger.info(f"Best model saved with training loss: {best_loss:.4f}")
        
        # Save checkpoint at regular intervals
        if (epoch + 1) % args.checkpoint_interval == 0 or epoch == args.epochs - 1:
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'vocab_to_idx': vocab_to_idx,
                'idx_to_vocab': idx_to_vocab,
                'history': history,
                'alpha': model.alpha,
                'beta': model.beta
            }, checkpoint_path)
            logger.info(f"Checkpoint saved at epoch {epoch+1}")
        
        # Check for early stopping
        if args.early_stopping > 0 and val_loader is not None:
            if len(history['val_loss']) > args.early_stopping:
                # Check if validation loss hasn't improved for early_stopping epochs
                if args.save_best_wer:
                    recent_wers = history['val_wer'][-args.early_stopping:]
                    if min(recent_wers) > best_wer and epoch > args.early_stopping:
                        logger.info(f"Early stopping at epoch {epoch+1} as validation WER hasn't improved for {args.early_stopping} epochs")
                        break
                else:
                    recent_losses = history['val_loss'][-args.early_stopping:]
                    if min(recent_losses) > best_loss and epoch > args.early_stopping:
                        logger.info(f"Early stopping at epoch {epoch+1} as validation loss hasn't improved for {args.early_stopping} epochs")
                        break
    
    # Save final model
    torch.save({
        'epoch': args.epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'vocab_to_idx': vocab_to_idx,
        'idx_to_vocab': idx_to_vocab,
        'history': history,
        'alpha': model.alpha,
        'beta': model.beta
    }, final_model_path)
    logger.info(f"Final model saved to {final_model_path}")
    
    # Plot training history
    plot_path = os.path.join(args.output_dir, 'training_history.png')
    plot_training_history(history, save_path=plot_path)
    
    # Save training history to file
    history_file = os.path.join(args.output_dir, 'training_history.json')
    with open(history_file, 'w') as f:
        json.dump(history, f)
    
    # Final evaluation on the best model
    if val_loader is not None:
        logger.info("Evaluating best model on validation set...")
        best_checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(best_checkpoint['model_state_dict'])
        metrics = evaluate_continuous_model(model, val_loader, vocab_to_idx, idx_to_vocab, device)
        
        # Save final metrics
        metrics_file = os.path.join(args.output_dir, 'best_model_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f)
        
        logger.info(f"Best model metrics:")
        logger.info(f"  Loss: {metrics['loss']:.4f}")
        logger.info(f"  WER: {metrics['wer']:.4f}")
        logger.info(f"  TER: {metrics['token_error_rate']:.4f}")
    
    # Clean up temporary files if they were created
    temp_train_csv = os.path.join(args.output_dir, 'temp_train.csv')
    temp_val_csv = os.path.join(args.output_dir, 'temp_val.csv')
    if os.path.exists(temp_train_csv):
        os.remove(temp_train_csv)
    if os.path.exists(temp_val_csv):
        os.remove(temp_val_csv)
    
    logger.info("Training completed!"'val')
    val_frames_dir = args.val_frames_dir if args.val_frames_dir else args.frames_root_dir.replace('train', 'val')
    
    if os.path.exists(val_csv_path):
        logger.info(f"Found validation CSV: {val_csv_path}")
        try:
            val_loader = create_dataloader(
                csv_path=val_csv_path,
                frames_root_dir=val_frames_dir,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                shuffle=False,
                validate_files=True
            )
            logger.info(f"Validation dataloader created with {len(val_loader.dataset)} samples")
        except Exception as e:
            logger.error(f"Failed to create validation dataloader: {e}")
            logger.info("Training will continue without validation")
    else:
        # Create a validation split from training data
        logger.info("No validation CSV found. Creating a validation subset from training data.")
        
        # We'll use a simple approach - just take the last validation_ratio of the dataset
        dataset_size = len(train_loader.dataset)
        val_size = int(dataset_size * args.validation_ratio)
        train_size = dataset_size - val_size
        
        # Create new datasets with manual splits
        # Note: Since we can't use sampler with the original create_dataloader, we need a different approach
        
        # We'll create a temporary CSV with just the training indices
        import pandas as pd
        import numpy as np
        
        try:
            # Read the original CSV
            df = pd.read_csv(args.csv_path)
            
            # Shuffle the dataframe
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)
            
            # Split into train and validation
            train_df = df.iloc[:train_size]
            val_df = df.iloc[train_size:]
            
            # Create temporary CSV files
            temp_train_csv = os.path.join(args.output_dir, 'temp_train.csv')
            temp_val_csv = os.path.join(args.output_dir, 'temp_val.csv')
            
            train_df.to_csv(temp_train_csv, index=False)
            val_df.to_csv(temp_val_csv, index=False)
            
            logger.info(f"Created temporary training CSV with {len(train_df)} samples")
            logger.info(f"Created temporary validation CSV with {len(val_df)} samples")
            
            # Create new dataloaders
            train_loader = create_dataloader(
                csv_path=temp_train_csv,
                frames_root_dir=args.frames_root_dir,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                shuffle=True,
                validate_files=True
            )
            
            val_loader = create_dataloader(
                csv_path=temp_val_csv,
                frames_root_dir=args.frames_root_dir,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                shuffle=False,
                validate_files=True
            )
            
            logger.info(f"Created training dataloader with {len(train_loader.dataset)} samples")
            logger.info(f"Created validation dataloader with {len(val_loader.dataset)} samples")
            
        except Exception as e:
            logger.error(f"Failed to create validation split: {e}")
            logger.info("Training will continue with the full dataset and no validation")
            # Restore the original train_loader just in case
            train_loader = create_dataloader(
                csv_path=args.csv_path,
                frames_root_dir=args.frames_root_dir,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                shuffle=True,
                validate_files=True
            )
    
    # Build vocabulary from training data
    logger.info("Building vocabulary from training data...")
    try:
        vocab_to_idx, idx_to_vocab = build_vocabulary(
            train_loader,
            min_freq=args.min_token_freq,
            special_tokens=['<PAD>', '<UNK>', '< SOS >', '<EOS>']
        )
        logger.info(f"Vocabulary built with {len(vocab_to_idx)} tokens")
        
        # Save vocabulary
        vocab_file = os.path.join(args.output_dir, 'vocabulary.json')
        with open(vocab_file, 'w') as f:
            json.dump({
                'vocab_to_idx': vocab_to_idx,
                'idx_to_vocab': idx_to_vocab
            }, f)
        logger.info(f"Vocabulary saved to {vocab_file}")
    except Exception as e:
        logger.error(f"Failed to build vocabulary: {e}")
        return
    
    # Initialize the CSLR model
    logger.info("Initializing CSLR model...")
    try:
        model = ContinuousSignLanguageModel(
            pretrained_model=pretrained_model,
            vocab_size=len(vocab_to_idx),
            hidden_dim=args.hidden_dim,
            embedding_dim=args.embedding_dim,
            freeze_resnet=args.freeze_resnet,
            freeze_bert=args.freeze_bert,
            alpha=args.alpha,
            beta=args.beta,
            num_heads=args.num_heads,
            dropout=args.dropout
        )
        model = model.to(device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model initialized with {total_params:,} total parameters")
        logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    except Exception as e:
        logger.error(f"Failed to initialize CSLR model: {e}")
        return
    
    # Initialize optimizer
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Setup model save paths
    best_model_path = os.path.join(args.output_dir, 'best_model.pth')
    final_model_path = os.path.join(args.output_dir, 'final_model.pth')
    
    # Training history
    history = {
        'loss': [],
        'contrastive_loss': [],
        'ctc_loss': []
    }
    
    if val_loader is not None:
        history.update({
            'val_loss': [],
            'val_contrastive_loss': [],
            'val_ctc_loss': [],
            'val_wer': []
        })
    
    # Training loop
    logger.info(f"Starting training for {args.epochs} epochs...")
    best_loss = float('inf')
    best_wer = float('inf')
    
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        total_loss = 0
        total_contrastive_loss = 0
        total_ctc_loss = 0
        batch_count = 0
        
        start_time = time.time()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Get data
            frames = batch['frames'].to(device)
            frame_mask = batch['frame_mask'].to(device)
            gloss1 = batch['gloss1']  # List of gloss strings
            
            # Tokenize target glosses for CTC loss
            target_sequences = []
            target_lengths = []
            
            for gloss in gloss1:
                # Split gloss string into tokens and convert to indices
                tokens = gloss.split()
                indices = [vocab_to_idx.get(token, vocab_to_idx['<UNK>']) for token in tokens]
                target_sequences.extend(indices)
                target_lengths.append(len(indices))
                
            targets = torch.tensor(target_sequences, dtype=torch.long, device=device)
            target_lengths = torch.tensor(target_lengths, dtype=torch.long, device=device)
            
            # Get input lengths from frame mask
            input_lengths = frame_mask.sum(dim=1).long()
            
            # Forward pass
            outputs = model(frames, frame_mask, gloss1)
            
            # Calculate loss
            loss, contrastive_loss, ctc_loss = model.calculate_total_loss(
                outputs, targets, input_lengths, target_lengths
            )
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            if args.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                
            optimizer.step()
            
            # Update statistics
            total_loss += loss.item()
            total_contrastive_loss += contrastive_loss.item()
            total_ctc_loss += ctc_loss.item()
            batch_count += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Cont": f"{contrastive_loss.item():.4f}",
                "CTC": f"{ctc_loss.item():.4f}"
            })
            
            # Log memory usage periodically
            if batch_idx % 50 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        
        # Calculate average losses
        avg_loss = total_loss / batch_count if batch_count > 0 else float('inf')
        avg_contrastive_loss = total_contrastive_loss / batch_count if batch_count > 0 else float('inf')
        avg_ctc_loss = total_ctc_loss / batch_count if batch_count > 0 else float('inf')
        epoch_time = time.time() - start_time
        
        # Update history
        history['loss'].append(avg_loss)
        history['contrastive_loss'].append(avg_contrastive_loss)
        history['ctc_loss'].append(avg_ctc_loss)
        
        logger.info(f"Epoch {epoch+1}/{args.epochs}, "
                  f"Loss: {avg_loss:.4f}, "
                  f"Contrastive: {avg_contrastive_loss:.4f}, "
                  f"CTC: {avg_ctc_loss:.4f}, "
                  f"Time: {epoch_time:.2f}s")
        
        # Evaluation phase
        if val_loader is not None:
            metrics = evaluate_continuous_model(model, val_loader, vocab_to_idx, idx_to_vocab, device)
            
            # Update history
            history['val_loss'].append(metrics['loss'])
            history['val_contrastive_loss'].append(metrics['contrastive_loss'])
            history['val_ctc_loss'].append(metrics['ctc_loss'])
            history['val_wer'].append(metrics['wer'])
            
            # Save the best model (based on validation WER if specified, otherwise validation loss)
            if args.save_best_wer:
                # Save based on WER
                current_wer = metrics['wer']
                if current_wer < best_wer:
                    best_wer = current_wer
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'vocab_to_idx': vocab_to_idx,
                        'idx_to_vocab': idx_to_vocab,
                        'metrics': metrics,
                        'alpha': model.alpha,
                        'beta': model.beta
                    }, best_model_path)
                    logger.info(f"Best model saved with validation WER: {best_wer:.4f}")
            else:
                # Save based on loss
                current_loss = metrics['loss']
                if current_loss < best_loss:
                    best_loss = current_loss
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'vocab_to_idx': vocab_to_idx,
                        'idx_to_vocab': idx_to_vocab,
                        'metrics': metrics,
                        'alpha': model.alpha,
                        'beta': model.beta
                    }, best_model_path)
                    logger.info(f"Best model saved with validation loss: {best_loss:.4f}")
        else:
            # Save based on training loss if no validation set
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'vocab_to_idx': vocab_to_idx,
                    'idx_to_vocab': idx_to_vocab,
                    'loss': avg_loss,
                    'alpha': model.alpha,
                    'beta': model.beta
                }, best_model_path)
                logger.info(f"Best model saved with training loss: {best_loss:.4f}")
        
        # Save checkpoint at regular intervals
        if (epoch + 1) % args.checkpoint_interval == 0 or epoch == args.epochs - 1:
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'vocab_to_idx': vocab_to_idx,
                'idx_to_vocab': idx_to_vocab,
                'history': history,
                'alpha': model.alpha,
                'beta': model.beta
            }, checkpoint_path)
            logger.info(f"Checkpoint saved at epoch {epoch+1}")
        
        # Check for early stopping
        if args.early_stopping > 0 and val_loader is not None:
            if len(history['val_loss']) > args.early_stopping:
                # Check if validation loss hasn't improved for early_stopping epochs
                if args.save_best_wer:
                    recent_wers = history['val_wer'][-args.early_stopping:]
                    if min(recent_wers) > best_wer and epoch > args.early_stopping:
                        logger.info(f"Early stopping at epoch {epoch+1} as validation WER hasn't improved for {args.early_stopping} epochs")
                        break
                else:
                    recent_losses = history['val_loss'][-args.early_stopping:]
                    if min(recent_losses) > best_loss and epoch > args.early_stopping:
                        logger.info(f"Early stopping at epoch {epoch+1} as validation loss hasn't improved for {args.early_stopping} epochs")
                        break
    
    # Save final model
    torch.save({
        'epoch': args.epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'vocab_to_idx': vocab_to_idx,
        'idx_to_vocab': idx_to_vocab,
        'history': history,
        'alpha': model.alpha,
        'beta': model.beta
    }, final_model_path)
    logger.info(f"Final model saved to {final_model_path}")
    
    # Plot training history
    plot_path = os.path.join(args.output_dir, 'training_history.png')
    plot_training_history(history, save_path=plot_path)
    
    # Save training history to file
    history_file = os.path.join(args.output_dir, 'training_history.json')
    with open(history_file, 'w') as f:
        json.dump(history, f)
    
    # Final evaluation on the best model
    if val_loader is not None:
        logger.info("Evaluating best model on validation set...")
        best_checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(best_checkpoint['model_state_dict'])
        metrics = evaluate_continuous_model(model, val_loader, vocab_to_idx, idx_to_vocab, device)
        
        # Save final metrics
        metrics_file = os.path.join(args.output_dir, 'best_model_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f)
        
        logger.info(f"Best model metrics:")
        logger.info(f"  Loss: {metrics['loss']:.4f}")
        logger.info(f"  WER: {metrics['wer']:.4f}")
        logger.info(f"  TER: {metrics['token_error_rate']:.4f}")
    
    # Clean up temporary files if they were created
    temp_train_csv = os.path.join(args.output_dir, 'temp_train.csv')
    temp_val_csv = os.path.join(args.output_dir, 'temp_val.csv')
    if os.path.exists(temp_train_csv):
        os.remove(temp_train_csv)
    if os.path.exists(temp_val_csv):
        os.remove(temp_val_csv)
    
    logger.info("Training completed!"'val')
    val_frames_dir = args.val_frames_dir if args.val_frames_dir else args.frames_root_dir.replace('train', 'val')
    
    if os.path.exists(val_csv_path):
        logger.info(f"Found validation CSV: {val_csv_path}")
        try:
            val_loader = create_dataloader(
                csv_path=val_csv_path,
                frames_root_dir=val_frames_dir,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                shuffle=False,
                validate_files=True
            )
            logger.info(f"Validation dataloader created with {len(val_loader.dataset)} samples")
        except Exception as e:
            logger.error(f"Failed to create validation dataloader: {e}")
            logger.info("Training will continue without validation")
    else:
        logger.info("No validation CSV found. Will use training data for model selection.")
    
    # Build vocabulary from training data
    logger.info("Building vocabulary from training data...")
    try:
        vocab_to_idx, idx_to_vocab = build_vocabulary(
            train_loader,
            min_freq=args.min_token_freq,
            special_tokens=['<PAD>', '<UNK>', '<SOS>', '<EOS>']
        )
        logger.info(f"Vocabulary built with {len(vocab_to_idx)} tokens")
        
        # Save vocabulary
        vocab_file = os.path.join(args.output_dir, 'vocabulary.json')
        with open(vocab_file, 'w') as f:
            json.dump({
                'vocab_to_idx': vocab_to_idx,
                'idx_to_vocab': idx_to_vocab
            }, f)
        logger.info(f"Vocabulary saved to {vocab_file}")
    except Exception as e:
        logger.error(f"Failed to build vocabulary: {e}")
        return
    
    # Initialize the CSLR model
    logger.info("Initializing CSLR model...")
    try:
        model = ContinuousSignLanguageModel(
            pretrained_model=pretrained_model,
            vocab_size=len(vocab_to_idx),
            hidden_dim=args.hidden_dim,
            embedding_dim=args.embedding_dim,
            freeze_resnet=args.freeze_resnet,
            freeze_bert=args.freeze_bert,
            alpha=args.alpha,
            beta=args.beta,
            num_heads=args.num_heads,
            dropout=args.dropout
        )
        model = model.to(device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model initialized with {total_params:,} total parameters")
        logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    except Exception as e:
        logger.error(f"Failed to initialize CSLR model: {e}")
        return
    
    # Initialize optimizer
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Setup model save paths
    best_model_path = os.path.join(args.output_dir, 'best_model.pth')
    final_model_path = os.path.join(args.output_dir, 'final_model.pth')
    
    # Training history
    history = {
        'loss': [],
        'contrastive_loss': [],
        'ctc_loss': []
    }
    
    if val_loader is not None:
        history.update({
            'val_loss': [],
            'val_contrastive_loss': [],
            'val_ctc_loss': [],
            'val_wer': []
        })
    
    # Training loop
    logger.info(f"Starting training for {args.epochs} epochs...")
    best_loss = float('inf')
    best_wer = float('inf')
    
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        total_loss = 0
        total_contrastive_loss = 0
        total_ctc_loss = 0
        batch_count = 0
        
        start_time = time.time()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Get data
            frames = batch['frames'].to(device)
            frame_mask = batch['frame_mask'].to(device)
            gloss1 = batch['gloss1']  # List of gloss strings
            
            # Tokenize target glosses for CTC loss
            target_sequences = []
            target_lengths = []
            
            for gloss in gloss1:
                # Split gloss string into tokens and convert to indices
                tokens = gloss.split()
                indices = [vocab_to_idx.get(token, vocab_to_idx['<UNK>']) for token in tokens]
                target_sequences.extend(indices)
                target_lengths.append(len(indices))
                
            targets = torch.tensor(target_sequences, dtype=torch.long, device=device)
            target_lengths = torch.tensor(target_lengths, dtype=torch.long, device=device)
            
            # Get input lengths from frame mask
            input_lengths = frame_mask.sum(dim=1).long()
            
            # Forward pass
            outputs = model(frames, frame_mask, gloss1)
            
            # Calculate loss
            loss, contrastive_loss, ctc_loss = model.calculate_total_loss(
                outputs, targets, input_lengths, target_lengths
            )
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            if args.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                
            optimizer.step()
            
            # Update statistics
            total_loss += loss.item()
            total_contrastive_loss += contrastive_loss.item()
            total_ctc_loss += ctc_loss.item()
            batch_count += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Cont": f"{contrastive_loss.item():.4f}",
                "CTC": f"{ctc_loss.item():.4f}"
            })
            
            # Log memory usage periodically
            if batch_idx % 50 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        
        # Calculate average losses
        avg_loss = total_loss / batch_count if batch_count > 0 else float('inf')
        avg_contrastive_loss = total_contrastive_loss / batch_count if batch_count > 0 else float('inf')
        avg_ctc_loss = total_ctc_loss / batch_count if batch_count > 0 else float('inf')
        epoch_time = time.time() - start_time
        
        # Update history
        history['loss'].append(avg_loss)
        history['contrastive_loss'].append(avg_contrastive_loss)
        history['ctc_loss'].append(avg_ctc_loss)
        
        logger.info(f"Epoch {epoch+1}/{args.epochs}, "
                  f"Loss: {avg_loss:.4f}, "
                  f"Contrastive: {avg_contrastive_loss:.4f}, "
                  f"CTC: {avg_ctc_loss:.4f}, "
                  f"Time: {epoch_time:.2f}s")
        
        # Evaluation phase
        if val_loader is not None:
            metrics = evaluate_continuous_model(model, val_loader, vocab_to_idx, idx_to_vocab, device)
            
            # Update history
            history['val_loss'].append(metrics['loss'])
            history['val_contrastive_loss'].append(metrics['contrastive_loss'])
            history['val_ctc_loss'].append(metrics['ctc_loss'])
            history['val_wer'].append(metrics['wer'])
            
            # Save the best model (based on validation WER if specified, otherwise validation loss)
            if args.save_best_wer:
                # Save based on WER
                current_wer = metrics['wer']
                if current_wer < best_wer:
                    best_wer = current_wer
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'vocab_to_idx': vocab_to_idx,
                        'idx_to_vocab': idx_to_vocab,
                        'metrics': metrics,
                        'alpha': model.alpha,
                        'beta': model.beta
                    }, best_model_path)
                    logger.info(f"Best model saved with validation WER: {best_wer:.4f}")
            else:
                # Save based on loss
                current_loss = metrics['loss']
                if current_loss < best_loss:
                    best_loss = current_loss
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'vocab_to_idx': vocab_to_idx,
                        'idx_to_vocab': idx_to_vocab,
                        'metrics': metrics,
                        'alpha': model.alpha,
                        'beta': model.beta
                    }, best_model_path)
                    logger.info(f"Best model saved with validation loss: {best_loss:.4f}")
        else:
            # Save based on training loss if no validation set
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'vocab_to_idx': vocab_to_idx,
                    'idx_to_vocab': idx_to_vocab,
                    'loss': avg_loss,
                    'alpha': model.alpha,
                    'beta': model.beta
                }, best_model_path)
                logger.info(f"Best model saved with training loss: {best_loss:.4f}")
        
        # Save checkpoint at regular intervals
        if (epoch + 1) % args.checkpoint_interval == 0 or epoch == args.epochs - 1:
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'vocab_to_idx': vocab_to_idx,
                'idx_to_vocab': idx_to_vocab,
                'history': history,
                'alpha': model.alpha,
                'beta': model.beta
            }, checkpoint_path)
            logger.info(f"Checkpoint saved at epoch {epoch+1}")
        
        # Check for early stopping
        if args.early_stopping > 0 and val_loader is not None:
            if len(history['val_loss']) > args.early_stopping:
                # Check if validation loss hasn't improved for early_stopping epochs
                if args.save_best_wer:
                    recent_wers = history['val_wer'][-args.early_stopping:]
                    if min(recent_wers) > best_wer and epoch > args.early_stopping:
                        logger.info(f"Early stopping at epoch {epoch+1} as validation WER hasn't improved for {args.early_stopping} epochs")
                        break
                else:
                    recent_losses = history['val_loss'][-args.early_stopping:]
                    if min(recent_losses) > best_loss and epoch > args.early_stopping:
                        logger.info(f"Early stopping at epoch {epoch+1} as validation loss hasn't improved for {args.early_stopping} epochs")
                        break
    
    # Save final model
    torch.save({
        'epoch': args.epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'vocab_to_idx': vocab_to_idx,
        'idx_to_vocab': idx_to_vocab,
        'history': history,
        'alpha': model.alpha,
        'beta': model.beta
    }, final_model_path)
    logger.info(f"Final model saved to {final_model_path}")
    
    # Plot training history
    plot_path = os.path.join(args.output_dir, 'training_history.png')
    plot_training_history(history, save_path=plot_path)
    
    # Save training history to file
    history_file = os.path.join(args.output_dir, 'training_history.json')
    with open(history_file, 'w') as f:
        json.dump(history, f)
    
    # Final evaluation on the best model
    if val_loader is not None:
        logger.info("Evaluating best model on validation set...")
        best_checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(best_checkpoint['model_state_dict'])
        metrics = evaluate_continuous_model(model, val_loader, vocab_to_idx, idx_to_vocab, device)
        
        # Save final metrics
        metrics_file = os.path.join(args.output_dir, 'best_model_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f)
        
        logger.info(f"Best model metrics:")
        logger.info(f"  Loss: {metrics['loss']:.4f}")
        logger.info(f"  WER: {metrics['wer']:.4f}")
        logger.info(f"  TER: {metrics['token_error_rate']:.4f}")
    
    logger.info("Training completed!")


def main():
    parser = argparse.ArgumentParser(description='Train Continuous Sign Language Recognition model')
    
    # Data paths
    parser.add_argument('--pretrained_model', type=str, required=True,
                       help='Path to the pretrained sign language model')
    parser.add_argument('--csv_path', type=str, required=True,
                       help='Path to the training CSV file')
    parser.add_argument('--frames_root_dir', type=str, required=True,
                       help='Root directory containing training frames')
    parser.add_argument('--val_csv_path', type=str, default=None,
                       help='Path to the validation CSV file (optional)')
    parser.add_argument('--val_frames_dir', type=str, default=None,
                       help='Root directory containing validation frames (optional)')
    
    # Output settings
    parser.add_argument('--output_dir', type=str, default='./cslr_models',
                       help='Directory to save models and results')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for training (default: 4)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of workers for data loading (default: 4)')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs (default: 30)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate (default: 1e-4)')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay (default: 1e-5)')
    parser.add_argument('--gradient_clip', type=float, default=5.0,
                       help='Gradient clipping value (default: 5.0, 0 for no clipping)')
    
    # Model parameters
    parser.add_argument('--alpha', type=float, default=0.5,
                       help='Weight for contrastive loss (default: 0.5)')
    parser.add_argument('--beta', type=float, default=0.5,
                       help='Weight for CTC loss (default: 0.5)')
    parser.add_argument('--hidden_dim', type=int, default=1024,
                       help='Hidden dimension (default: 1024)')
    parser.add_argument('--embedding_dim', type=int, default=512,
                       help='Embedding dimension (default: 512)')
    parser.add_argument('--num_heads', type=int, default=8,
                       help='Number of attention heads (default: 8)')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate (default: 0.1)')
    
    # Vocabulary parameters
    parser.add_argument('--min_token_freq', type=int, default=2,
                       help='Minimum token frequency for vocabulary (default: 2)')
    
    # Model saving parameters
    parser.add_argument('--checkpoint_interval', type=int, default=5,
                       help='Save checkpoint every N epochs (default: 5)')
    parser.add_argument('--save_best_wer', action='store_true',
                       help='Save best model based on WER instead of loss')
    parser.add_argument('--early_stopping', type=int, default=10,
                       help='Stop training if no improvement for N epochs (default: 10, 0 to disable)')
    
    # Freezing options
    parser.add_argument('--freeze_resnet', action='store_true',
                       help='Freeze ResNet parameters')
    parser.add_argument('--freeze_bert', action='store_true',
                       help='Freeze BERT parameters')
    
    # Validation split ratio (if no validation set provided)
    parser.add_argument('--validation_ratio', type=float, default=0.1,
                       help='Ratio of training data to use for validation (default: 0.1)')
    
    args = parser.parse_args()
    
    # Print arguments
    logger.info("Training with the following parameters:")
    for arg in vars(args):
        logger.info(f"  {arg}: {getattr(args, arg)}")
    
    # Run training
    train_cslr_model(args)


if __name__ == "__main__":
    main()