import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from transformers import BertModel, BertTokenizer
import math
from tqdm import tqdm
import logging
import argparse
import time
from pathlib import Path

# Handle Levenshtein import - try different packages
try:
    import Levenshtein
except ImportError:
    try:
        # Some installations use python-Levenshtein
        import levenshtein as Levenshtein
    except ImportError:
        # Fallback to editdistance
        try:
            import editdistance
            
            # Create a compatible interface
            class LevenshteinCompat:
                @staticmethod
                def distance(s1, s2):
                    return editdistance.eval(s1, s2)
            
            Levenshtein = LevenshteinCompat
        except ImportError:
            # If nothing works, we'll implement a basic version
            class BasicLevenshtein:
                @staticmethod
                def distance(s1, s2):
                    """
                    Simple Levenshtein distance implementation
                    """
                    # Convert lists to tuples for consistent hashing
                    if isinstance(s1, list):
                        s1 = tuple(s1)
                    if isinstance(s2, list):
                        s2 = tuple(s2)
                        
                    # Create a matrix
                    m, n = len(s1), len(s2)
                    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
                    
                    # Initialize
                    for i in range(m + 1):
                        dp[i][0] = i
                    for j in range(n + 1):
                        dp[0][j] = j
                    
                    # Fill the matrix
                    for i in range(1, m + 1):
                        for j in range(1, n + 1):
                            if s1[i-1] == s2[j-1]:
                                dp[i][j] = dp[i-1][j-1]
                            else:
                                dp[i][j] = 1 + min(dp[i-1][j],      # Deletion
                                                dp[i][j-1],      # Insertion
                                                dp[i-1][j-1])    # Substitution
                    
                    return dp[m][n]
            
            Levenshtein = BasicLevenshtein
            logging.warning("No Levenshtein module found. Using basic built-in implementation.")
            logging.warning("For better performance, install python-Levenshtein or editdistance package.")

# Import the components from stage 1
from sign_language_training import SignLanguageModel, VisualStream, TextStream, PositionalEncoding
from data_loader import create_dataloader

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#-------------------------
# Stage 2: CSLR Model Components
#-------------------------

class CrossModalAttention(nn.Module):
    """
    Cross-Modal Attention module for visual-text alignment
    """
    def __init__(self, visual_dim, text_dim, num_heads=8):
        super(CrossModalAttention, self).__init__()
        self.num_heads = num_heads
        self.visual_dim = visual_dim
        self.text_dim = text_dim
        self.head_dim = visual_dim // num_heads
        assert visual_dim % num_heads == 0, "visual_dim must be divisible by num_heads"
        
        # Projection layers - handle different dimensions between visual and text features
        self.query_proj = nn.Linear(visual_dim, visual_dim)
        self.key_proj = nn.Linear(text_dim, visual_dim)  # Project text to visual dimension
        self.value_proj = nn.Linear(text_dim, visual_dim)  # Project text to visual dimension
        self.output_proj = nn.Linear(visual_dim, visual_dim)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, visual_features, text_features, attention_mask=None):
        """
        Cross-modal attention with visual features as queries and text features as keys/values
        
        Args:
            visual_features: [batch_size, seq_len_v, visual_dim]
            text_features: [batch_size, seq_len_t, text_dim]
            attention_mask: [batch_size, seq_len_v, seq_len_t] (optional)
        
        Returns:
            attended_features: [batch_size, seq_len_v, visual_dim]
        """
        batch_size, seq_len_v, _ = visual_features.size()
        seq_len_t = text_features.size(1)
        
        # Project queries, keys, and values
        q = self.query_proj(visual_features).view(batch_size, seq_len_v, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.key_proj(text_features).view(batch_size, seq_len_t, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.value_proj(text_features).view(batch_size, seq_len_t, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand mask for multiple heads: [batch_size, 1, seq_len_v, seq_len_t]
            expanded_mask = attention_mask.unsqueeze(1)
            scores = scores.masked_fill(expanded_mask == 0, -1e9)
        
        # Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Compute attended values
        attended = torch.matmul(attn_weights, v)
        attended = attended.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len_v, self.visual_dim)
        
        # Final projection
        output = self.output_proj(attended)
        
        return output

class GlossTokenizer:
    """
    Tokenizer for gloss sequences
    """
    def __init__(self, gloss_file=None):
        self.gloss_to_id = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<blank>': 3}
        self.id_to_gloss = {0: '<pad>', 1: '<sos>', 2: '<eos>', 3: '<blank>'}
        self.num_glosses = 4  # Start with special tokens
        
        # Load gloss vocabulary if provided
        if gloss_file:
            self.load_gloss_vocab(gloss_file)
    
    def load_gloss_vocab(self, gloss_file):
        """
        Load gloss vocabulary from file
        """
        try:
            with open(gloss_file, 'r') as f:
                for line in f:
                    gloss = line.strip()
                    if gloss and gloss not in self.gloss_to_id:
                        self.gloss_to_id[gloss] = self.num_glosses
                        self.id_to_gloss[self.num_glosses] = gloss
                        self.num_glosses += 1
            logger.info(f"Loaded {self.num_glosses} glosses from {gloss_file}")
        except Exception as e:
            logger.error(f"Error loading gloss vocabulary: {e}")
    
    def build_vocab_from_data(self, gloss_data):
        """
        Build vocabulary from gloss data
        """
        for gloss_seq in gloss_data:
            # Split by space to get individual glosses
            glosses = gloss_seq.split()
            for gloss in glosses:
                if gloss and gloss not in self.gloss_to_id:
                    self.gloss_to_id[gloss] = self.num_glosses
                    self.id_to_gloss[self.num_glosses] = gloss
                    self.num_glosses += 1
        logger.info(f"Built vocabulary with {self.num_glosses} glosses")
    
    def encode(self, gloss_seq):
        """
        Encode a gloss sequence to token IDs
        """
        if isinstance(gloss_seq, str):
            # Split by space and convert to IDs
            glosses = gloss_seq.split()
            return [self.gloss_to_id.get(gloss, self.gloss_to_id['<sos>']) for gloss in glosses]
        else:
            # Assume list/batch of gloss sequences
            return [self.encode(seq) for seq in gloss_seq]
    
    def decode(self, token_ids, remove_special=True):
        """
        Decode token IDs to gloss sequence
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.cpu().numpy()
            
        if isinstance(token_ids[0], (list, np.ndarray)):
            # Batch of sequences
            return [self.decode(seq, remove_special) for seq in token_ids]
        
        # Single sequence
        glosses = []
        for token_id in token_ids:
            gloss = self.id_to_gloss.get(token_id, '<unk>')
            if remove_special and gloss in ['<pad>', '<sos>', '<eos>', '<blank>']:
                continue
            glosses.append(gloss)
        
        return ' '.join(glosses)

class CSLRModel(nn.Module):
    """
    End-to-end CSLR model with pretrained visual encoder and CTC loss
    """
    def __init__(self, pretrained_model_path=None, hidden_dim=1024, gloss_tokenizer=None, 
                 freeze_visual=True, freeze_text=True, unfreeze_attention=True):
        super(CSLRModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.gloss_tokenizer = gloss_tokenizer
        
        # Initialize the base model
        self.sign_model = SignLanguageModel(hidden_dim=hidden_dim)
        
        # Load pretrained weights if path is provided
        if pretrained_model_path is not None:
            try:
                pretrained_state_dict = torch.load(pretrained_model_path, map_location=device)
                
                # Get model state dict based on the format of the saved checkpoint
                if 'model_state_dict' in pretrained_state_dict:
                    pretrained_state_dict = pretrained_state_dict['model_state_dict']
                
                # Try to load pretrained weights (ignore mismatches)
                model_state_dict = self.sign_model.state_dict()
                pretrained_dict = {k: v for k, v in pretrained_state_dict.items() if k in model_state_dict}
                model_state_dict.update(pretrained_dict)
                self.sign_model.load_state_dict(model_state_dict)
                
                logger.info(f"Loaded {len(pretrained_dict)}/{len(model_state_dict)} pretrained model parameters")
            except Exception as e:
                logger.error(f"Error loading pretrained model: {e}")
                logger.warning("Initializing model with random weights")
        else:
            logger.info("No pretrained model path provided. Initializing with random weights.")
        
        # Freeze/unfreeze parts of the model
        # Visual Stream
        for name, param in self.sign_model.visual_stream.named_parameters():
            if freeze_visual:
                param.requires_grad = False
                # Optionally unfreeze multi-head attention layers if specified
                if unfreeze_attention and 'attention' in name:
                    param.requires_grad = True
            else:
                param.requires_grad = True
        
        # Text Stream
        for name, param in self.sign_model.text_stream.named_parameters():
            if freeze_text:
                param.requires_grad = False
            else:
                param.requires_grad = True
                
        # Get BERT embedding dimension
        self.bert_dim = self.sign_model.text_stream.bert.config.hidden_size  # Usually 768 for BERT base
                
        # Additional components for CSLR
        self.cross_modal_attn = CrossModalAttention(
            visual_dim=hidden_dim,  # Visual features dimension (1024)
            text_dim=self.bert_dim   # BERT features dimension (768)
        )
        
        # Gloss classifier for CTC
        if gloss_tokenizer:
            self.num_glosses = gloss_tokenizer.num_glosses
            self.ctc_fc = nn.Linear(hidden_dim, gloss_tokenizer.num_glosses)
        else:
            # Default to a reasonable number if tokenizer not provided yet
            self.num_glosses = 1000
            self.ctc_fc = nn.Linear(hidden_dim, self.num_glosses)
    
    def update_gloss_tokenizer(self, gloss_tokenizer):
        """
        Update the gloss tokenizer and classifier layer
        """
        self.gloss_tokenizer = gloss_tokenizer
        self.num_glosses = gloss_tokenizer.num_glosses
        
        # Create new classifier with correct output dimension
        old_fc = self.ctc_fc
        self.ctc_fc = nn.Linear(self.hidden_dim, self.num_glosses)
        
        # Initialize with existing weights if possible
        if old_fc.out_features <= self.num_glosses:
            with torch.no_grad():
                self.ctc_fc.weight.data[:old_fc.out_features] = old_fc.weight.data
                self.ctc_fc.bias.data[:old_fc.out_features] = old_fc.bias.data
    
    def forward(self, frames, frame_mask, gloss_text, return_embeddings=False):
        """
        Forward pass for the CSLR model
        
        Args:
            frames: [batch_size, max_frames, C, H, W] - Visual frames
            frame_mask: [batch_size, max_frames] - Mask for valid frames
            gloss_text: List of gloss sentences
            return_embeddings: Whether to return embeddings for contrastive loss
            
        Returns:
            logits: CTC logits [batch_size, max_frames, num_glosses]
            visual_emb: Visual embeddings [batch_size, max_frames, hidden_dim] (if return_embeddings=True)
            text_emb: Text embeddings [batch_size, hidden_dim] (if return_embeddings=True)
        """
        # Get embeddings from pretrained model
        visual_emb, text_emb = self.sign_model(frames, frame_mask, gloss_text)
        
        # Get token-level BERT embeddings for cross-modal attention
        # Use the text_stream to get contextualized embeddings
        tokenizer = self.sign_model.text_stream.tokenizer
        bert = self.sign_model.text_stream.bert
        
        encoded_inputs = tokenizer(gloss_text, padding=True, truncation=True, return_tensors='pt')
        encoded_inputs = {k: v.to(device) for k, v in encoded_inputs.items()}
        
        with torch.set_grad_enabled(not self.sign_model.text_stream.bert.training):
            bert_outputs = bert(**encoded_inputs)
            token_embeddings = bert_outputs.last_hidden_state
        
        # Apply cross-modal attention (visual as queries, text as keys/values)
        # Create attention mask from encoded_inputs
        attention_mask = encoded_inputs['attention_mask'].unsqueeze(1)  # [batch_size, 1, seq_len_t]
        attention_mask = attention_mask.expand(-1, visual_emb.size(1), -1)  # [batch_size, seq_len_v, seq_len_t]
        
        enhanced_visual = self.cross_modal_attn(visual_emb, token_embeddings, attention_mask)
        
        # Apply CTC classifier
        logits = self.ctc_fc(enhanced_visual)  # [batch_size, max_frames, num_glosses]
        
        if return_embeddings:
            return logits, visual_emb, text_emb
        else:
            return logits

class CombinedLoss(nn.Module):
    """
    Combined loss function with CTC and contrastive components
    """
    def __init__(self, ctc_weight=1.0, contrastive_weight=0.2, temperature=0.07):
        super(CombinedLoss, self).__init__()
        self.ctc_weight = ctc_weight
        self.contrastive_weight = contrastive_weight
        self.temperature = temperature
        self.ctc_loss = nn.CTCLoss(blank=3, reduction='mean', zero_infinity=True)  # blank=3 for <blank> token
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, logits, targets, target_lengths, visual_emb, text_emb, input_lengths=None):
        """
        Compute combined CTC and contrastive loss
        
        Args:
            logits: [batch_size, max_frames, num_glosses] - CTC logits
            targets: [batch_size, max_target_length] - Padded target gloss IDs
            target_lengths: [batch_size] - Actual lengths of target sequences
            visual_emb: [batch_size, max_frames, hidden_dim] - Visual embeddings
            text_emb: [batch_size, hidden_dim] - Text embeddings
            input_lengths: [batch_size] - Actual lengths of input sequences
            
        Returns:
            total_loss: Combined loss
            ctc_loss_val: CTC loss value
            contrastive_loss_val: Contrastive loss value
        """
        batch_size, max_frames, num_glosses = logits.size()
        
        # Prepare CTC inputs
        if input_lengths is None:
            # If not provided, assume all frames are valid
            input_lengths = torch.full((batch_size,), max_frames, device=logits.device)
        
        # Prepare logits for CTC (need [T, B, C] format)
        log_probs = F.log_softmax(logits, dim=2).permute(1, 0, 2)
        
        # Prepare targets for CTC (need flattened targets)
        flattened_targets = torch.cat([targets[b, :target_lengths[b]] for b in range(batch_size)])
        
        # Compute CTC loss
        ctc_loss_val = self.ctc_loss(log_probs, flattened_targets, input_lengths, target_lengths)
        
        # For contrastive loss, use mean of frame features
        if visual_emb.dim() == 3:  # [batch_size, max_frames, hidden_dim]
            # Average along time dimension
            visual_emb_avg = torch.mean(visual_emb, dim=1)  # [batch_size, hidden_dim]
        else:
            visual_emb_avg = visual_emb
            
        # Compute contrastive loss
        # Compute similarity matrix
        sim_matrix = torch.matmul(visual_emb_avg, text_emb.t()) / self.temperature
        
        # Labels: diagonal elements (positive pairs)
        labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
        
        # Compute contrastive loss (NT-Xent loss)
        loss_v2t = self.criterion(sim_matrix, labels)  # Visual to text
        loss_t2v = self.criterion(sim_matrix.t(), labels)  # Text to visual
        
        contrastive_loss_val = (loss_v2t + loss_t2v) / 2
        
        # Combine losses
        total_loss = self.ctc_weight * ctc_loss_val + self.contrastive_weight * contrastive_loss_val
        
        return total_loss, ctc_loss_val, contrastive_loss_val

#-------------------------
# Training and Evaluation Functions
#-------------------------

def ctc_decode(logits, input_lengths=None, beam_width=10):
    """
    Greedy CTC decoding
    
    Args:
        logits: [batch_size, max_frames, num_glosses] - CTC logits
        input_lengths: [batch_size] - Actual lengths of input sequences
        beam_width: Beam width for beam search decoding
        
    Returns:
        best_paths: [batch_size, max_decoded_length] - Best paths (token IDs)
    """
    batch_size, max_frames, num_glosses = logits.size()
    
    if input_lengths is None:
        input_lengths = torch.full((batch_size,), max_frames, device=logits.device)
    
    # Convert to log probabilities
    log_probs = F.log_softmax(logits, dim=2)
    
    # List to store results
    results = []
    
    # Process each sequence in the batch
    for b in range(batch_size):
        seq_len = input_lengths[b]
        seq_log_probs = log_probs[b, :seq_len]
        
        # Greedy decoding (take argmax at each step)
        best_path = torch.argmax(seq_log_probs, dim=1)
        
        # Remove repeated tokens
        condensed_path = []
        prev_token = -1
        
        for token in best_path:
            token_id = token.item()
            # Skip blanks and repeated tokens
            if token_id != 3 and token_id != prev_token:  # 3 is <blank>
                condensed_path.append(token_id)
            prev_token = token_id
        
        results.append(condensed_path)
    
    # Pad results to same length
    max_decoded_length = max(len(path) for path in results) if results else 0
    padded_results = []
    
    for path in results:
        # Pad with 0 (<pad>)
        padded_path = path + [0] * (max_decoded_length - len(path))
        padded_results.append(padded_path)
    
    return padded_results

def compute_wer(predictions, targets, tokenizer):
    """
    Compute Word Error Rate (WER) for gloss sequences
    
    Args:
        predictions: List of predicted gloss sequences (strings or token IDs)
        targets: List of target gloss sequences (strings or token IDs)
        tokenizer: GlossTokenizer instance for decoding
        
    Returns:
        wer: Word Error Rate
    """
    # Convert predictions and targets to strings if they are token IDs
    if not isinstance(predictions[0], str):
        predictions = tokenizer.decode(predictions, remove_special=True)
    
    if not isinstance(targets[0], str):
        targets = tokenizer.decode(targets, remove_special=True)
    
    # Calculate WER
    total_words = 0
    total_errors = 0
    
    for pred, target in zip(predictions, targets):
        # Calculate Levenshtein distance at word level
        pred_words = pred.split()
        target_words = target.split()
        
        total_words += len(target_words)
        distance = Levenshtein.distance(pred_words, target_words)
        total_errors += distance
    
    # Calculate WER
    wer = total_errors / total_words if total_words > 0 else 1.0
    
    return wer

def train_cslr(model, train_loader, val_loader, optimizer, loss_fn, 
              epochs, save_path, device, tokenizer=None, scheduler=None):
    """
    Train the CSLR model
    
    Args:
        model: CSLRModel instance
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer
        loss_fn: Loss function (CombinedLoss)
        epochs: Number of epochs
        save_path: Path to save models
        device: Device to train on
        tokenizer: GlossTokenizer instance
        scheduler: Learning rate scheduler (optional)
    """
    model.train()
    best_wer = float('inf')
    gradient_clip_value = 1.0
    
    for epoch in range(epochs):
        total_loss = 0
        total_ctc_loss = 0
        total_contrastive_loss = 0
        total_batches = 0
        
        start_time = time.time()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            frames = batch['frames'].to(device)
            frame_mask = batch['frame_mask'].to(device)
            
            # Get the text (use gloss1 for now)
            gloss_text = batch['gloss1']  # List of strings
            
            # Get target sequences
            target_sequences = tokenizer.encode(gloss_text)
            max_target_len = max(len(seq) for seq in target_sequences)
            
            # Pad target sequences
            padded_targets = [seq + [0] * (max_target_len - len(seq)) for seq in target_sequences]
            targets = torch.tensor(padded_targets, device=device)
            target_lengths = torch.tensor([len(seq) for seq in target_sequences], device=device)
            
            # Input lengths from frame_mask
            input_lengths = frame_mask.sum(dim=1).long()
            
            # Forward pass
            logits, visual_emb, text_emb = model(frames, frame_mask, gloss_text, return_embeddings=True)
            
            # Compute loss
            loss, ctc_loss_val, contrastive_loss_val = loss_fn(
                logits, targets, target_lengths, visual_emb, text_emb, input_lengths
            )
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_value)
            
            optimizer.step()
            
            # Update statistics
            total_loss += loss.item()
            total_ctc_loss += ctc_loss_val.item()
            total_contrastive_loss += contrastive_loss_val.item()
            total_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                "Loss": loss.item(),
                "CTC": ctc_loss_val.item(),
                "Contrastive": contrastive_loss_val.item()
            })
            
            # Log every 10 batches
            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, "
                          f"Loss: {loss.item():.4f}, CTC: {ctc_loss_val.item():.4f}, "
                          f"Contrastive: {contrastive_loss_val.item():.4f}")
        
        # Calculate average losses for the epoch
        avg_loss = total_loss / total_batches
        avg_ctc_loss = total_ctc_loss / total_batches
        avg_contrastive_loss = total_contrastive_loss / total_batches
        epoch_time = time.time() - start_time
        
        logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}, "
                  f"CTC: {avg_ctc_loss:.4f}, Contrastive: {avg_contrastive_loss:.4f}, "
                  f"Time: {epoch_time:.2f}s")
        
        # Step scheduler if provided
        if scheduler:
            scheduler.step()
        
        # Evaluate on validation set
        if val_loader:
            val_loss, val_wer = evaluate_cslr(model, val_loader, loss_fn, device, tokenizer)
            logger.info(f"Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss:.4f}, WER: {val_wer:.4f}")
            
            # Save the best model based on validation WER
            if val_wer < best_wer:
                best_wer = val_wer
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_loss,
                    'wer': val_wer,
                }, os.path.join(save_path, f'best_cslr_model_wer_{val_wer:.4f}.pth'))
                logger.info(f"Best model saved with validation WER: {best_wer:.4f}")
        else:
            # Save based on training loss if no validation set
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, os.path.join(save_path, f'cslr_model_epoch_{epoch+1}.pth'))
            logger.info(f"Model saved at epoch {epoch+1}")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, os.path.join(save_path, f'cslr_checkpoint_epoch_{epoch+1}.pth'))
            logger.info(f"Checkpoint saved at epoch {epoch+1}")

def evaluate_cslr(model, data_loader, loss_fn, device, tokenizer):
    """
    Evaluate the CSLR model
    
    Args:
        model: CSLRModel instance
        data_loader: Data loader
        loss_fn: Loss function (CombinedLoss)
        device: Device to evaluate on
        tokenizer: GlossTokenizer instance
        
    Returns:
        avg_loss: Average loss on the evaluation set
        wer: Word Error Rate
    """
    model.eval()
    total_loss = 0
    total_batches = 0
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            # Move data to device
            frames = batch['frames'].to(device)
            frame_mask = batch['frame_mask'].to(device)
            
            # Get the text (use gloss1 for now)
            gloss_text = batch['gloss1']  # List of strings
            
            # Get target sequences
            target_sequences = tokenizer.encode(gloss_text)
            max_target_len = max(len(seq) for seq in target_sequences)
            
            # Pad target sequences
            padded_targets = [seq + [0] * (max_target_len - len(seq)) for seq in target_sequences]
            targets = torch.tensor(padded_targets, device=device)
            target_lengths = torch.tensor([len(seq) for seq in target_sequences], device=device)
            
            # Input lengths from frame_mask
            input_lengths = frame_mask.sum(dim=1).long()
            
            # Forward pass
            logits, visual_emb, text_emb = model(frames, frame_mask, gloss_text, return_embeddings=True)
            
            # Compute loss
            loss, _, _ = loss_fn(
                logits, targets, target_lengths, visual_emb, text_emb, input_lengths
            )
            
            # Update statistics
            total_loss += loss.item()
            total_batches += 1
            
            # Decode predictions
            predictions = ctc_decode(logits, input_lengths)
            
            # Store predictions and targets for WER calculation
            all_predictions.extend(predictions)
            all_targets.extend([seq for seq in target_sequences])
    
    # Calculate average loss
    avg_loss = total_loss / total_batches if total_batches > 0 else float('inf')
    
    # Calculate WER
    wer = compute_wer(all_predictions, all_targets, tokenizer)
    
    return avg_loss, wer

#-------------------------
# Main Function
#-------------------------

def main():
    """Main function for training and evaluation"""
    
    parser = argparse.ArgumentParser(description='Train CSLR Model (Stage 2)')
    
    # Base model parameters
    parser.add_argument('--pretrained_model', type=str, required=True,
                        help='Path to pretrained Stage 1 model checkpoint')
    
    # Data paths (optional - will use defaults if not provided)
    parser.add_argument('--csv_path', type=str, 
                        default=None,
                        help='Path to the CSV file (optional - will use default from data_loader.py)')
    parser.add_argument('--frames_root_dir', type=str, 
                        default=None,
                        help='Root directory containing frames (optional - will use default from data_loader.py)')
    parser.add_argument('--gloss_vocab', type=str, default=None,
                        help='Path to gloss vocabulary file (optional)')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--temperature', type=float, default=0.07, help='Temperature for contrastive loss')
    parser.add_argument('--ctc_weight', type=float, default=1.0, help='Weight for CTC loss')
    parser.add_argument('--contrastive_weight', type=float, default=0.2, help='Weight for contrastive loss')
    
    # Model parameters
    parser.add_argument('--hidden_dim', type=int, default=1024, help='Hidden dimension')
    parser.add_argument('--freeze_visual', action='store_true', help='Freeze visual backbone')
    parser.add_argument('--freeze_text', action='store_true', help='Freeze text backbone')
    parser.add_argument('--unfreeze_attention', action='store_true', 
                        help='Unfreeze attention layers in visual backbone')
    
    # Training phases
    parser.add_argument('--phase', type=int, default=1, choices=[1, 2], 
                        help='Training phase: 1=freeze backbones, 2=fine-tune')
    
    # Saving parameters
    parser.add_argument('--save_path', type=str, default='./saved_models/cslr', 
                        help='Path to save models')
    parser.add_argument('--validate_files', action='store_true', help='Validate if frame files exist')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set up paths
    if args.csv_path is None:
        args.csv_path = "/home/pvvkishore/Desktop/IEEE_THMS_May10/Phonix_Train/train_gloss_eng.csv"
        logger.info(f"No CSV path provided, using default: {args.csv_path}")
    
    if args.frames_root_dir is None:
        args.frames_root_dir = "/home/pvvkishore/Desktop/IEEE_THMS_May10/Phonix_Train/train/"
        logger.info(f"No frames root directory provided, using default: {args.frames_root_dir}")
    
    # Create save directory
    os.makedirs(args.save_path, exist_ok=True)
    
    # Log all arguments
    logger.info("Training CSLR model with the following parameters:")
    for arg in vars(args):
        logger.info(f"  {arg}: {getattr(args, arg)}")
    
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
    
    # Validation dataloader (if available)
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
    
    # Create gloss tokenizer
    logger.info("Creating gloss tokenizer...")
    tokenizer = GlossTokenizer(gloss_file=args.gloss_vocab)
    
    # If no vocabulary file provided, build vocabulary from the training data
    if args.gloss_vocab is None:
        logger.info("No gloss vocabulary file provided. Building vocabulary from data...")
        # Extract all gloss1 entries from the data
        gloss_data = []
        for batch in tqdm(train_loader, desc="Building vocabulary"):
            gloss_data.extend(batch['gloss1'])
            
            # Add gloss2 if available
            if 'gloss2' in batch:
                gloss_data.extend(batch['gloss2'])
                
            # Only process the first few batches to save time
            if len(gloss_data) > 5000:
                break
                
        # Build vocabulary
        tokenizer.build_vocab_from_data(gloss_data)
    
    # Initialize CSLR model
    logger.info(f"Initializing CSLR model from pretrained model: {args.pretrained_model}")
    
    # For Phase 1: Freeze both backbones
    # For Phase 2: Optionally unfreeze components
    freeze_visual = args.freeze_visual or (args.phase == 1)
    freeze_text = args.freeze_text or (args.phase == 1)
    unfreeze_attention = args.unfreeze_attention or (args.phase == 1)
    
    model = CSLRModel(
        pretrained_model_path=args.pretrained_model,
        hidden_dim=args.hidden_dim,
        gloss_tokenizer=tokenizer,
        freeze_visual=freeze_visual,
        freeze_text=freeze_text,
        unfreeze_attention=unfreeze_attention
    )
    
    # Update number of classes if needed
    model.update_gloss_tokenizer(tokenizer)
    
    # Move model to device
    model = model.to(device)
    
    # Log model parameters status
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params}")
    logger.info(f"Trainable parameters: {trainable_params} ({trainable_params/total_params*100:.2f}%)")
    
    # Initialize loss function
    loss_fn = CombinedLoss(
        ctc_weight=args.ctc_weight,
        contrastive_weight=args.contrastive_weight,
        temperature=args.temperature
    ).to(device)
    
    # Initialize optimizer
    # We use separate parameter groups for different components
    param_groups = [
        {'params': [p for n, p in model.named_parameters() if 'cross_modal_attn' in n], 'lr': args.learning_rate},
        {'params': [p for n, p in model.named_parameters() if 'ctc_fc' in n], 'lr': args.learning_rate}
    ]
    
    # Add visual backbone params if not frozen
    if not freeze_visual:
        visual_params = [p for n, p in model.named_parameters() 
                         if 'visual_stream' in n and p.requires_grad]
        if visual_params:
            param_groups.append({'params': visual_params, 'lr': args.learning_rate * 0.1})
    
    # Add text backbone params if not frozen
    if not freeze_text:
        text_params = [p for n, p in model.named_parameters() 
                       if 'text_stream' in n and p.requires_grad]
        if text_params:
            param_groups.append({'params': text_params, 'lr': args.learning_rate * 0.01})
    
    optimizer = torch.optim.Adam(param_groups, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.1
    )
    
    # Train the model
    logger.info("Starting CSLR training...")
    train_cslr(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=args.epochs,
        save_path=args.save_path,
        device=device,
        tokenizer=tokenizer,
        scheduler=scheduler
    )
    
    # Save the final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'tokenizer': tokenizer.__dict__,  # Save tokenizer state
    }, os.path.join(args.save_path, 'final_cslr_model.pth'))
    
    logger.info("CSLR Training completed!")

if __name__ == '__main__':
    main()