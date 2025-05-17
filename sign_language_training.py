#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 15 17:22:01 2025

@author: pvvkishore
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader
import numpy as np
from transformers import BertModel, BertTokenizer
import math
from tqdm import tqdm
import logging
import argparse
import time
from pathlib import Path

# Import the data loader
from data_loader import create_dataloader

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#-------------------------
# Model Architecture Components
#-------------------------

# Dynamic Positional Encoding for Visual Stream
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=200):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # Register buffer (not a parameter, but part of the module)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x expected shape: (batch_size, seq_len, d_model)
        # Add positional encoding
        x = x + self.pe[:, :x.size(1), :]
        return x

# Multi-Head Attention Module
class MultiHeadAttention(nn.Module):
    def __init__(self, in_features, head_num):
        super(MultiHeadAttention, self).__init__()
        self.head_num = head_num
        self.head_dim = in_features // head_num
        assert in_features % head_num == 0, "in_features must be divisible by head_num"
        
        self.query = nn.Linear(in_features, in_features)
        self.key = nn.Linear(in_features, in_features)
        self.value = nn.Linear(in_features, in_features)
        self.out = nn.Linear(in_features, in_features)
        
    def forward(self, x):
        # x shape: (batch_size, channels, height, width)
        batch_size, channels, height, width = x.size()
        
        # Reshape for attention
        x_flat = x.view(batch_size, channels, -1).permute(0, 2, 1)  # (batch_size, h*w, channels)
        
        # Linear projections
        queries = self.query(x_flat).view(batch_size, -1, self.head_num, self.head_dim).permute(0, 2, 1, 3)
        keys = self.key(x_flat).view(batch_size, -1, self.head_num, self.head_dim).permute(0, 2, 1, 3)
        values = self.value(x_flat).view(batch_size, -1, self.head_num, self.head_dim).permute(0, 2, 1, 3)
        
        # Scaled dot-product attention
        energy = torch.matmul(queries, keys.permute(0, 1, 3, 2)) / (self.head_dim ** 0.5)
        attention = F.softmax(energy, dim=-1)
        out = torch.matmul(attention, values)
        
        # Reshape back
        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, channels)
        out = self.out(out)
        
        # Reshape to original format
        out = out.permute(0, 2, 1).view(batch_size, channels, height, width)
        
        return out

# Bottleneck Block for ResNet50 with Multi-Head Attention
class BottleneckWithAttention(nn.Module):
    expansion = 4  # Bottleneck blocks expand the channels by a factor of 4
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, num_heads=4):
        super(BottleneckWithAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.attention = MultiHeadAttention(out_channels * self.expansion, num_heads)
        
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        # Apply multi-head attention
        out = self.attention(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
            
        out += residual
        out = self.relu(out)
        
        return out

# Custom ResNet50 with Multi-Head Attention
class ResNet50WithAttention(nn.Module):
    def __init__(self, block=BottleneckWithAttention, num_blocks=[3, 4, 6, 3], num_heads=[2, 4, 8, 8]):
        super(ResNet50WithAttention, self).__init__()
        self.in_planes = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet50 has different block structure compared to ResNet18
        # For ResNet50, the blocks are "Bottleneck" blocks with 3 conv layers each
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, num_heads=num_heads[0])
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, num_heads=num_heads[1])
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, num_heads=num_heads[2])
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, num_heads=num_heads[3])
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, 1024)  # ResNet50 typically has 2048 channels (512*4) at the end
        
    def _make_layer(self, block, planes, num_blocks, stride, num_heads):
        downsample = None
        if stride != 1 or self.in_planes != planes * 4:  # *4 for Bottleneck
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * 4, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * 4)
            )
            
        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample, num_heads=num_heads))
        self.in_planes = planes * 4  # *4 for Bottleneck
        for _ in range(1, num_blocks):
            layers.append(block(self.in_planes, planes, num_heads=num_heads))
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # x shape: (batch_size, channels, height, width)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        
        return out

# Visual Stream
class VisualStream(nn.Module):
    def __init__(self, hidden_dim=1024):
        super(VisualStream, self).__init__()
        self.hidden_dim = hidden_dim
        
        # Dynamic Positional Encoding
        self.pos_encoder = PositionalEncoding(3 * 112 * 112)  # For input frames
        
        # ResNet50 with Multi-Head Attention instead of ResNet18
        self.resnet = ResNet50WithAttention()
        
        # Resize layer for 224x224 to 112x112
        self.resize = nn.AdaptiveAvgPool2d((112, 112))
        
        # Bi-LSTM for temporal context
        self.lstm = nn.LSTM(
            input_size=1024,  # Output from ResNet
            hidden_size=hidden_dim // 2,  # Bidirectional will double this
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
    def forward(self, x, frame_mask=None):
        # x shape: (batch_size, T, 3, 224, 224)
        batch_size, T, C, H, W = x.size()
        
        # Process each frame through ResNet
        frame_features = []
        for t in range(T):
            # Skip processing if all frames in the batch at this timestep are masked out
            if frame_mask is not None and frame_mask[:, t].sum() == 0:
                continue
                
            frame = x[:, t]  # (batch_size, C, H, W)
            
            # Resize to 112x112 as specified in the requirements
            frame = self.resize(frame)  # Now (batch_size, C, 112, 112)
            
            # Apply positional encoding (reshape to match expected dimensions)
            frame_flat = frame.view(batch_size, C * 112 * 112)
            frame_flat = frame_flat.unsqueeze(1)  # Add sequence dimension
            frame_pos = self.pos_encoder(frame_flat)
            frame_pos = frame_pos.squeeze(1).view(batch_size, C, 112, 112)
            
            # Pass through ResNet
            frame_feat = self.resnet(frame_pos)  # (batch_size, 1024)
            
            # Handle frame mask if provided
            if frame_mask is not None:
                # Zero out features for masked frames
                mask_t = frame_mask[:, t].float().view(batch_size, 1)
                frame_feat = frame_feat * mask_t
                
            frame_features.append(frame_feat.unsqueeze(1))  # Add time dimension
        
        # Handle edge case where no frames are valid
        if not frame_features:
            # Return zero tensor with correct shape
            return torch.zeros(batch_size, T, self.hidden_dim, device=x.device)
        
        # Concatenate all frame features
        all_frames = torch.cat(frame_features, dim=1)  # (batch_size, valid_T, 1024)
        
        # If we have fewer valid frames than total frames, pad with zeros
        if all_frames.size(1) < T:
            padding = torch.zeros(batch_size, T - all_frames.size(1), 1024, device=all_frames.device)
            all_frames = torch.cat([all_frames, padding], dim=1)
        
        # Create packed sequence if frame_mask is provided
        if frame_mask is not None:
            # Count valid frames per batch
            valid_frames_per_batch = frame_mask.sum(dim=1).cpu()
            
            # Skip LSTM if no valid frames
            if valid_frames_per_batch.sum() == 0:
                return torch.zeros(batch_size, T, self.hidden_dim, device=x.device)
                
            # Create packed sequence
            packed_frames = nn.utils.rnn.pack_padded_sequence(
                all_frames, 
                valid_frames_per_batch, 
                batch_first=True, 
                enforce_sorted=False
            )
            
            # Pass through Bi-LSTM
            packed_lstm_out, _ = self.lstm(packed_frames)
            
            # Unpack sequence
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                packed_lstm_out, 
                batch_first=True, 
                total_length=T
            )
        else:
            # Regular LSTM pass without packing
            lstm_out, _ = self.lstm(all_frames)
        
        # L2 normalize visual features
        visual_emb = F.normalize(lstm_out, p=2, dim=2)
        
        return visual_emb

# Text Stream using BERT
class TextStream(nn.Module):
    def __init__(self, hidden_dim=1024, bert_model='bert-base-uncased', freeze_bert=True):
        super(TextStream, self).__init__()
        self.hidden_dim = hidden_dim
        
        # Load pre-trained BERT model
        self.bert = BertModel.from_pretrained(bert_model)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)
        
        # Freeze BERT parameters if specified
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
                
        # Projection layer
        self.fc = nn.Linear(self.bert.config.hidden_size, hidden_dim)
        
    def forward(self, text_list):
        # text_list is a list of gloss sentences
        
        # Tokenize the input
        encoded_inputs = self.tokenizer(text_list, padding=True, truncation=True, return_tensors='pt')
        encoded_inputs = {k: v.to(device) for k, v in encoded_inputs.items()}
        
        # Forward pass through BERT
        outputs = self.bert(**encoded_inputs)
        
        # Use CLS token as sentence representation
        cls_tokens = outputs.last_hidden_state[:, 0]
        
        # Project to the same dimension as visual features
        text_emb = self.fc(cls_tokens)
        
        # L2 normalize text features
        text_emb = F.normalize(text_emb, p=2, dim=1)
        
        return text_emb

# Contrastive Loss
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, visual_emb, text_emb):
        # visual_emb: (batch_size, T, D')
        # text_emb: (batch_size, D')
        
        # For visual embedding, take the mean over time dimension
        visual_emb = torch.mean(visual_emb, dim=1)  # (batch_size, D')
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(visual_emb, text_emb.t()) / self.temperature
        
        # Labels: diagonal elements (positive pairs)
        labels = torch.arange(sim_matrix.size(0)).to(device)
        
        # Compute loss (NTXent loss)
        loss_v2t = self.criterion(sim_matrix, labels)  # Visual to text
        loss_t2v = self.criterion(sim_matrix.t(), labels)  # Text to visual
        
        # Total loss
        loss = (loss_v2t + loss_t2v) / 2
        
        return loss

# Combined Model
class SignLanguageModel(nn.Module):
    def __init__(self, hidden_dim=1024, bert_model='bert-base-uncased', freeze_bert=True):
        super(SignLanguageModel, self).__init__()
        
        # Visual and Text streams
        self.visual_stream = VisualStream(hidden_dim=hidden_dim)
        self.text_stream = TextStream(hidden_dim=hidden_dim, bert_model=bert_model, freeze_bert=freeze_bert)
        
    def forward(self, frames, frame_mask, gloss_text):
        # Process visual input with frame mask
        visual_emb = self.visual_stream(frames, frame_mask)
        
        # Process text input
        text_emb = self.text_stream(gloss_text)
        
        return visual_emb, text_emb

#-------------------------
# Training and Evaluation Functions
#-------------------------

def train(model, train_loader, val_loader, optimizer, criterion, epochs, save_path, device):
    """
    Train the sign language model
    
    Args:
        model: The sign language model
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer for training
        criterion: Loss function (NTXentLoss)
        epochs: Number of training epochs
        save_path: Path to save the model
        device: Device to train on
    """
    model.train()
    best_loss = float('inf')
    
    for epoch in range(epochs):
        total_loss = 0
        total_batches = 0
        
        start_time = time.time()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            frames = batch['frames'].to(device)
            frame_mask = batch['frame_mask'].to(device)
            
            # Get the text (use gloss1 for now)
            gloss_text = batch['gloss1']  # List of strings
            
            # Forward pass
            visual_emb, text_emb = model(frames, frame_mask, gloss_text)
            
            # Compute loss
            loss = criterion(visual_emb, text_emb)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update statistics
            total_loss += loss.item()
            total_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({"Loss": loss.item()})
            
            # Log every 10 batches
            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        # Calculate average loss for the epoch
        avg_loss = total_loss / total_batches
        epoch_time = time.time() - start_time
        
        logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s")
        
        # Evaluate on validation set
        if val_loader:
            val_loss = evaluate(model, val_loader, criterion, device)
            logger.info(f"Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss:.4f}")
            
            # Save the best model based on validation loss
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                }, os.path.join(save_path, f'best_model_epoch_{epoch+1}.pth'))
                logger.info(f"Best model saved with validation loss: {best_loss:.4f}")
        else:
            # Save the model based on training loss if no validation set
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                }, os.path.join(save_path, f'best_model_epoch_{epoch+1}.pth'))
                logger.info(f"Best model saved with training loss: {best_loss:.4f}")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, os.path.join(save_path, f'checkpoint_epoch_{epoch+1}.pth'))
            logger.info(f"Checkpoint saved at epoch {epoch+1}")

def evaluate(model, data_loader, criterion, device):
    """
    Evaluate the sign language model
    
    Args:
        model: The sign language model
        data_loader: Data loader
        criterion: Loss function (NTXentLoss)
        device: Device to evaluate on
    
    Returns:
        avg_loss: Average loss on the evaluation set
    """
    model.eval()
    total_loss = 0
    total_batches = 0
    
    with torch.no_grad():
        for batch in data_loader:
            # Move data to device
            frames = batch['frames'].to(device)
            frame_mask = batch['frame_mask'].to(device)
            
            # Get the text (use gloss1 for now)
            gloss_text = batch['gloss1']  # List of strings
            
            # Forward pass
            visual_emb, text_emb = model(frames, frame_mask, gloss_text)
            
            # Compute loss
            loss = criterion(visual_emb, text_emb)
            
            # Update statistics
            total_loss += loss.item()
            total_batches += 1
    
    # Calculate average loss
    avg_loss = total_loss / total_batches
    
    return avg_loss

#-------------------------
# Main Function
#-------------------------

def main():
    """Main function for training and evaluation"""
    
    parser = argparse.ArgumentParser(description='Train Sign Language Visual-Text Model')
    
    # Data paths (optional - will use defaults if not provided)
    parser.add_argument('--csv_path', type=str, 
                        default=None,
                        help='Path to the CSV file (optional - will use default from data_loader.py)')
    parser.add_argument('--frames_root_dir', type=str, 
                        default=None,
                        help='Root directory containing frames (optional - will use default from data_loader.py)')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--epochs', type=int, default=2, help='Number of training epochs')
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
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_path, exist_ok=True)
    
    # Log all arguments
    logger.info("Training with the following parameters:")
    for arg in vars(args):
        logger.info(f"  {arg}: {getattr(args, arg)}")
    
    # Use default paths if not provided
    if args.csv_path is None:
        args.csv_path = "/home/pvvkishore/Desktop/IEEE_THMS_May10/Phonix_Train/train_gloss_eng.csv"
        logger.info(f"No CSV path provided, using default: {args.csv_path}")
    
    if args.frames_root_dir is None:
        args.frames_root_dir = "/home/pvvkishore/Desktop/IEEE_THMS_May10/Phonix_Train/train/"
        logger.info(f"No frames root directory provided, using default: {args.frames_root_dir}")
    
    # Create data loaders
    logger.info("Creating dataloaders...")
    
    # Training dataloader
    train_loader = create_dataloader(
        csv_path=args.csv_path,
        frames_root_dir=args.frames_root_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        validate_files=args.validate_files
    )
    
    # Validation dataloader (if provided)
    val_loader = None
    
    # Try to find the validation CSV path
    val_csv_path = args.csv_path.replace('train', 'val')
    val_frames_dir = args.frames_root_dir.replace('train', 'val')
    
    # Create validation dataloader if path exists
    if os.path.exists(val_csv_path):
        logger.info(f"Found validation CSV: {val_csv_path}")
        
        val_loader = create_dataloader(
            csv_path=val_csv_path,
            frames_root_dir=val_frames_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            validate_files=args.validate_files
        )
    else:
        logger.info("No validation CSV found. Will use training loss for model selection.")
    
    # Initialize model
    logger.info("Initializing model...")
    model = SignLanguageModel(
        hidden_dim=args.hidden_dim, 
        bert_model=args.bert_model, 
        freeze_bert=args.freeze_bert
    )
    model = model.to(device)
    
    # Initialize loss function and optimizer
    criterion = NTXentLoss(temperature=args.temperature).to(device)
    
    # Only update visual backbone, keep BERT frozen
    optimizer = torch.optim.Adam(
        [p for n, p in model.named_parameters() if not n.startswith('text_stream.bert')],
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Train the model
    logger.info("Starting training...")
    train(model, train_loader, val_loader, optimizer, criterion, args.epochs, args.save_path, device)
    
    # Save the final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join(args.save_path, 'final_model.pth'))
    
    logger.info("Training completed!")

if __name__ == '__main__':
    main()