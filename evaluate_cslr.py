import os
import argparse
import logging
import json
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import confusion_matrix

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

# Import model and utilities
from cslr_model import CSLRModel, GlossTokenizer, ctc_decode, compute_wer
from data_loader import create_dataloader

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define functions that were previously imported from inference
def load_cslr_model(model_path, device):
    """
    Load a trained CSLR model
    
    Args:
        model_path: Path to the model checkpoint
        device: Device to load the model on
        
    Returns:
        model: Loaded CSLRModel
        tokenizer: GlossTokenizer
    """
    logger.info(f"Loading model from {model_path}")
    
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        raise FileNotFoundError(f"Could not find model file at {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Check if tokenizer information is saved in the checkpoint
    if 'tokenizer' in checkpoint:
        tokenizer_dict = checkpoint['tokenizer']
        tokenizer = GlossTokenizer()
        tokenizer.__dict__.update(tokenizer_dict)
        logger.info(f"Loaded tokenizer with {tokenizer.num_glosses} glosses")
    else:
        logger.warning("No tokenizer found in checkpoint. Creating a default one.")
        tokenizer = GlossTokenizer()
    
    # Create a model instance without loading pretrained weights
    model = CSLRModel(
        pretrained_model_path=None,  # Not needed as we'll load state dict directly
        gloss_tokenizer=tokenizer
    )
    
    # Now load the state dict from the checkpoint
    if 'model_state_dict' in checkpoint:
        model_state_dict = checkpoint['model_state_dict']
        # Try to load state dict (handle missing keys)
        try:
            model.load_state_dict(model_state_dict)
            logger.info("Model state dict loaded successfully")
        except Exception as e:
            logger.warning(f"Error loading model state dict: {e}")
            # Try partial loading
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in model_state_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            logger.info(f"Loaded {len(pretrained_dict)}/{len(model_dict)} parameters")
    else:
        logger.warning("No model_state_dict found in checkpoint. Using random weights.")
    
    # Move to device and set to evaluation mode
    model = model.to(device)
    model.eval()
    
    return model, tokenizer

def run_inference(model, data_loader, tokenizer, device, save_dir=None, conf_threshold=0.9):
    """
    Run inference with the CSLR model
    
    Args:
        model: CSLRModel instance
        data_loader: Data loader for testing
        tokenizer: GlossTokenizer instance
        device: Device to run inference on
        save_dir: Directory to save results (optional)
        conf_threshold: Confidence threshold for predictions
        
    Returns:
        results: Dictionary with prediction results
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_folder_names = []
    
    # Create save directory if specified
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Running inference")):
            # Move data to device
            frames = batch['frames'].to(device)
            frame_mask = batch['frame_mask'].to(device)
            
            # Get the text (use gloss1 for now)
            gloss_text = batch['gloss1']  # List of strings
            folder_names = batch['folder_names']  # List of folder names
            
            # Input lengths from frame_mask
            input_lengths = frame_mask.sum(dim=1).long()
            
            # Forward pass
            logits = model(frames, frame_mask, gloss_text)
            
            # Decode predictions
            predictions = ctc_decode(logits, input_lengths)
            
            # Get confidence scores
            log_probs = F.log_softmax(logits, dim=2)
            confidences = []
            
            for b in range(frames.size(0)):
                seq_len = input_lengths[b]
                seq_log_probs = log_probs[b, :seq_len]
                
                # Get max probability at each step
                max_probs = torch.exp(torch.max(seq_log_probs, dim=1)[0])
                avg_conf = max_probs.mean().item()
                confidences.append(avg_conf)
            
            # Get target sequences
            target_sequences = tokenizer.encode(gloss_text)
            
            # Store predictions and targets
            decoded_preds = tokenizer.decode(predictions, remove_special=True)
            decoded_targets = tokenizer.decode(target_sequences, remove_special=True)
            
            # Save results for each sample
            for i in range(len(decoded_preds)):
                folder_name = folder_names[i]
                pred = decoded_preds[i]
                target = decoded_targets[i]
                confidence = confidences[i]
                
                all_predictions.append(pred)
                all_targets.append(target)
                all_folder_names.append(folder_name)
                
                # Save detailed results to file
                if save_dir:
                    with open(os.path.join(save_dir, f"{folder_name}.json"), 'w') as f:
                        json.dump({
                            'folder': folder_name,
                            'prediction': pred,
                            'target': target,
                            'confidence': confidence
                        }, f, indent=2)
    
    # Calculate overall metrics
    wer = compute_wer(all_predictions, all_targets, tokenizer)
    logger.info(f"Overall WER: {wer:.4f}")
    
    # Save results to JSON
    if save_dir:
        results_file = os.path.join(save_dir, "inference_results.json")
        with open(results_file, 'w') as f:
            json.dump({
                'wer': wer,
                'sample_results': [
                    {
                        'folder': folder,
                        'prediction': pred,
                        'target': target
                    }
                    for folder, pred, target in zip(all_folder_names, all_predictions, all_targets)
                ]
            }, f, indent=2)
        
        logger.info(f"Results saved to {results_file}")
    
    # Return all results
    return {
        'wer': wer,
        'predictions': all_predictions,
        'targets': all_targets,
        'folder_names': all_folder_names
    }

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def analyze_errors(predictions, targets, tokenizer, save_path=None):
    """
    Analyze prediction errors
    
    Args:
        predictions: List of predicted gloss sequences
        targets: List of target gloss sequences
        tokenizer: GlossTokenizer instance
        save_path: Path to save the analysis (optional)
        
    Returns:
        error_stats: Dictionary with error statistics
    """
    error_stats = {}
    
    # Process predictions and targets
    pred_glosses = []
    target_glosses = []
    
    for pred, target in zip(predictions, targets):
        pred_words = pred.split()
        target_words = target.split()
        
        pred_glosses.extend(pred_words)
        target_glosses.extend(target_words)
    
    # Count gloss frequencies
    target_counter = {}
    for gloss in target_glosses:
        if gloss in target_counter:
            target_counter[gloss] += 1
        else:
            target_counter[gloss] = 1
    
    # Sort by frequency
    sorted_glosses = sorted(target_counter.items(), key=lambda x: x[1], reverse=True)
    
    # Get most common glosses
    common_glosses = [g[0] for g in sorted_glosses[:30]]
    
    # Prepare gloss-level confusion matrix
    gloss_to_idx = {gloss: i for i, gloss in enumerate(common_glosses)}
    
    # Initialize confusion matrix for common glosses
    y_true = []
    y_pred = []
    
    # Collect data for confusion matrix
    for pred, target in zip(predictions, targets):
        pred_words = pred.split()
        target_words = target.split()
        
        # Calculate aligned sequences
        # This is a simplified alignment - a real implementation would use
        # dynamic programming for sequence alignment
        min_len = min(len(pred_words), len(target_words))
        
        for i in range(min_len):
            t_gloss = target_words[i]
            p_gloss = pred_words[i]
            
            if t_gloss in common_glosses:
                y_true.append(t_gloss)
                y_pred.append(p_gloss)
    
    # Create confusion matrix
    confmat = None
    if y_true and y_pred:
        # Convert to indices for confusion matrix
        y_true_idx = [gloss_to_idx[g] for g in y_true if g in gloss_to_idx]
        y_pred_idx = []
        
        for i, g in enumerate(y_pred):
            if y_true[i] in gloss_to_idx:
                if g in gloss_to_idx:
                    y_pred_idx.append(gloss_to_idx[g])
                else:
                    # Use a special "other" category for predictions outside common glosses
                    y_pred_idx.append(-1)
        
        if y_true_idx and y_pred_idx:
            # Create confusion matrix
            conf_labels = list(range(len(common_glosses)))
            confmat = confusion_matrix(y_true_idx, y_pred_idx, labels=conf_labels)
            
            # Normalize by true positives
            row_sums = confmat.sum(axis=1)
            norm_confmat = confmat / row_sums[:, np.newaxis]
            
            # Plot confusion matrix
            plt.figure(figsize=(12, 10))
            sns.heatmap(
                norm_confmat, 
                annot=False, 
                fmt='.2f', 
                cmap='Blues',
                xticklabels=common_glosses,
                yticklabels=common_glosses
            )
            plt.title('Normalized Confusion Matrix (Top 30 Glosses)')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Confusion matrix saved to {save_path}")
            
            plt.close()
    
    # Calculate per-sample statistics
    sample_stats = []
    
    for i, (pred, target) in enumerate(zip(predictions, targets)):
        pred_words = pred.split()
        target_words = target.split()
        
        # Calculate WER
        total_words = len(target_words)
        distance = 0
        
        if total_words > 0:
            distance = Levenshtein.distance(pred_words, target_words)
            sample_wer = distance / total_words
        else:
            sample_wer = 0.0
        
        # Calculate correct, insertions, deletions, substitutions
        correct = 0
        insertions = 0
        deletions = 0
        substitutions = 0
        
        # Use Dynamic Programming to calculate edit operations
        if pred_words and target_words:
            # Create edit distance matrix
            m, n = len(pred_words), len(target_words)
            dp = [[0 for _ in range(n+1)] for _ in range(m+1)]
            
            # Initialize base cases
            for i in range(m+1):
                dp[i][0] = i
            for j in range(n+1):
                dp[0][j] = j
            
            # Fill the matrix
            for i in range(1, m+1):
                for j in range(1, n+1):
                    if pred_words[i-1] == target_words[j-1]:
                        dp[i][j] = dp[i-1][j-1]
                    else:
                        dp[i][j] = min(
                            dp[i-1][j] + 1,      # Deletion
                            dp[i][j-1] + 1,      # Insertion
                            dp[i-1][j-1] + 1     # Substitution
                        )
            
            # Backtrack to find operations
            i, j = m, n
            while i > 0 and j > 0:
                if pred_words[i-1] == target_words[j-1]:
                    correct += 1
                    i -= 1
                    j -= 1
                elif dp[i][j] == dp[i-1][j-1] + 1:
                    substitutions += 1
                    i -= 1
                    j -= 1
                elif dp[i][j] == dp[i-1][j] + 1:
                    deletions += 1
                    i -= 1
                elif dp[i][j] == dp[i][j-1] + 1:
                    insertions += 1
                    j -= 1
                else:
                    # Should not happen
                    i -= 1
                    j -= 1
            
            # Handle remaining cases
            while i > 0:
                deletions += 1
                i -= 1
            while j > 0:
                insertions += 1
                j -= 1
        
        sample_stats.append({
            'index': i,
            'prediction': pred,
            'target': target,
            'wer': sample_wer,
            'correct': correct,
            'substitutions': substitutions,
            'insertions': insertions,
            'deletions': deletions,
            'target_length': len(target_words)
        })
    
    # Calculate overall statistics
    total_words = sum(len(t.split()) for t in targets)
    correct_words = sum(s['correct'] for s in sample_stats)
    substitutions = sum(s['substitutions'] for s in sample_stats)
    insertions = sum(s['insertions'] for s in sample_stats)
    deletions = sum(s['deletions'] for s in sample_stats)
    
    accuracy = correct_words / total_words if total_words > 0 else 0
    
    error_stats = {
        'total_samples': len(predictions),
        'total_words': total_words,
        'correct_words': correct_words,
        'substitutions': substitutions,
        'insertions': insertions,
        'deletions': deletions,
        'accuracy': accuracy,
        'wer': 1.0 - accuracy,
        'sample_stats': sample_stats
    }
    
    # Save statistics to file
    if save_path:
        base_path = os.path.splitext(save_path)[0]
        stats_path = f"{base_path}_stats.json"
        
        with open(stats_path, 'w') as f:
            # Convert numpy values to Python native types for JSON serialization
            json_compatible_stats = {
                'total_samples': int(error_stats['total_samples']),
                'total_words': int(error_stats['total_words']),
                'correct_words': int(error_stats['correct_words']),
                'substitutions': int(error_stats['substitutions']),
                'insertions': int(error_stats['insertions']),
                'deletions': int(error_stats['deletions']),
                'accuracy': float(error_stats['accuracy']),
                'wer': float(error_stats['wer']),
            }
            
            json.dump(json_compatible_stats, f, indent=2)
            logger.info(f"Error statistics saved to {stats_path}")
    
    return error_stats

def evaluate_model_on_dataset(model, data_loader, tokenizer, device, save_dir=None):
    """
    Evaluate model on a dataset
    
    Args:
        model: CSLRModel instance
        data_loader: Data loader for evaluation
        tokenizer: GlossTokenizer instance
        device: Device to run evaluation on
        save_dir: Directory to save results (optional)
        
    Returns:
        metrics: Dictionary with evaluation metrics
    """
    # Run inference
    results = run_inference(
        model=model,
        data_loader=data_loader,
        tokenizer=tokenizer,
        device=device,
        save_dir=save_dir
    )
    
    # Analyze errors
    if save_dir:
        confmat_path = os.path.join(save_dir, 'confusion_matrix.png')
    else:
        confmat_path = None
        
    error_stats = analyze_errors(
        predictions=results['predictions'],
        targets=results['targets'],
        tokenizer=tokenizer,
        save_path=confmat_path
    )
    
    # Create performance summary
    metrics = {
        'wer': results['wer'],
        'accuracy': error_stats['accuracy'],
        'substitutions': error_stats['substitutions'],
        'insertions': error_stats['insertions'],
        'deletions': error_stats['deletions'],
        'total_words': error_stats['total_words']
    }
    
    # Print summary
    logger.info("\n===== Evaluation Summary =====")
    logger.info(f"Word Error Rate (WER): {metrics['wer']:.4f}")
    logger.info(f"Word Recognition Rate: {metrics['accuracy']:.4f}")
    logger.info(f"Total Words: {metrics['total_words']}")
    logger.info(f"Substitutions: {metrics['substitutions']} ({metrics['substitutions']/metrics['total_words']*100:.2f}%)")
    logger.info(f"Insertions: {metrics['insertions']} ({metrics['insertions']/metrics['total_words']*100:.2f}%)")
    logger.info(f"Deletions: {metrics['deletions']} ({metrics['deletions']/metrics['total_words']*100:.2f}%)")
    
    if save_dir:
        # Save metrics to file
        metrics_path = os.path.join(save_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
            logger.info(f"Metrics saved to {metrics_path}")
        
        # Create visualization for error distribution
        plt.figure(figsize=(10, 6))
        labels = ['Correct', 'Substitutions', 'Insertions', 'Deletions']
        values = [
            error_stats['correct_words'],
            error_stats['substitutions'],
            error_stats['insertions'],
            error_stats['deletions']
        ]
        
        colors = ['#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
        plt.bar(labels, values, color=colors)
        plt.title('Word Error Distribution')
        plt.ylabel('Count')
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on top of each bar
        for i, v in enumerate(values):
            plt.text(i, v + 5, str(v), ha='center')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'error_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Evaluate CSLR model')
    
    # Model and data paths
    parser.add_argument('--model_path', type=str, required=True, help='Path to CSLR model checkpoint')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to test CSV file')
    parser.add_argument('--frames_root_dir', type=str, required=True, help='Root directory for test frames')
    
    # Evaluation parameters
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    
    # Output parameters
    parser.add_argument('--save_dir', type=str, default='./evaluation_results', help='Directory to save results')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load model and tokenizer
    model, tokenizer = load_cslr_model(args.model_path, device)
    
    # Create data loader
    logger.info(f"Creating data loader for {args.csv_path}")
    
    test_loader = create_dataloader(
        csv_path=args.csv_path,
        frames_root_dir=args.frames_root_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        validate_files=True
    )
    
    # Evaluate model
    logger.info("Starting evaluation...")
    metrics = evaluate_model_on_dataset(
        model=model,
        data_loader=test_loader,
        tokenizer=tokenizer,
        device=device,
        save_dir=args.save_dir
    )
    
    logger.info("Evaluation completed!")

if __name__ == '__main__':
    main()