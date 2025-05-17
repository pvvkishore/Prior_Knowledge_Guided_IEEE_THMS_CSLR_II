import torch
import os
import logging
import argparse
from transformers import BertTokenizer
from tqdm import tqdm
from data_loader import create_dataloader
from pathlib import Path

# Import the model
from sign_language_training import SignLanguageModel

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def extract_features(model, data_loader, output_dir, mode='visual', device=device):
    """
    Extract features from the trained model
    
    Args:
        model: The trained sign language model
        data_loader: Data loader
        output_dir: Directory to save extracted features
        mode: 'visual', 'text', or 'both'
        device: Device to run extraction on
    """
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectories
    visual_dir = os.path.join(output_dir, 'visual_features')
    text_dir = os.path.join(output_dir, 'text_features')
    
    if mode in ['visual', 'both']:
        os.makedirs(visual_dir, exist_ok=True)
    if mode in ['text', 'both']:
        os.makedirs(text_dir, exist_ok=True)
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_loader, desc=f"Extracting {mode} features")):
            # Get folder names (for saving)
            folder_names = batch['folder_names']
            
            # Move data to device
            frames = batch['frames'].to(device)
            frame_mask = batch['frame_mask'].to(device)
            
            # Get text data
            gloss1 = batch['gloss1']
            gloss2 = batch['gloss2']
            
            # Extract features based on mode
            if mode in ['visual', 'both']:
                # Extract visual features
                visual_emb = model.visual_stream(frames, frame_mask)
                
                # Save visual features for each sample in the batch
                for j, folder_name in enumerate(folder_names):
                    # Create sample directory
                    sample_dir = os.path.join(visual_dir, folder_name)
                    os.makedirs(sample_dir, exist_ok=True)
                    
                    # Save features
                    feature_path = os.path.join(sample_dir, 'visual_features.pt')
                    torch.save(visual_emb[j].cpu(), feature_path)
                    
                    # Save metadata
                    with open(os.path.join(sample_dir, 'metadata.txt'), 'w') as f:
                        f.write(f"Folder: {folder_name}\n")
                        f.write(f"Gloss1: {gloss1[j]}\n")
                        f.write(f"Gloss2: {gloss2[j]}\n")
                        f.write(f"Num frames: {batch['num_frames'][j].item()}\n")
            
            if mode in ['text', 'both']:
                # Extract text features for gloss1
                text_emb1 = model.text_stream(gloss1)
                
                # Extract text features for gloss2
                text_emb2 = model.text_stream(gloss2)
                
                # Save text features for each sample
                for j, folder_name in enumerate(folder_names):
                    # Create sample directory
                    sample_dir = os.path.join(text_dir, folder_name)
                    os.makedirs(sample_dir, exist_ok=True)
                    
                    # Save features
                    feature_path1 = os.path.join(sample_dir, 'text_features_gloss1.pt')
                    feature_path2 = os.path.join(sample_dir, 'text_features_gloss2.pt')
                    torch.save(text_emb1[j].cpu(), feature_path1)
                    torch.save(text_emb2[j].cpu(), feature_path2)
                    
                    # Save metadata
                    with open(os.path.join(sample_dir, 'metadata.txt'), 'w') as f:
                        f.write(f"Folder: {folder_name}\n")
                        f.write(f"Gloss1: {gloss1[j]}\n")
                        f.write(f"Gloss2: {gloss2[j]}\n")
            
            # Log progress
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {(i + 1) * len(folder_names)} samples")

def calculate_similarity(visual_features, text_features):
    """
    Calculate similarity between visual and text features
    
    Args:
        visual_features: Visual features tensor
        text_features: Text features tensor
        
    Returns:
        similarity: Cosine similarity score
    """
    # Normalize features (should already be normalized, but just to be safe)
    visual_features = torch.nn.functional.normalize(visual_features, p=2, dim=-1)
    text_features = torch.nn.functional.normalize(text_features, p=2, dim=-1)
    
    # For visual features, take mean over temporal dimension if needed
    if len(visual_features.shape) > 1 and visual_features.shape[0] > 1:
        visual_features = torch.mean(visual_features, dim=0)
    
    # Calculate cosine similarity
    similarity = torch.matmul(visual_features.flatten(), text_features.flatten())
    
    return similarity.item()

def evaluate_alignment(model_path, csv_path, frames_root_dir, output_dir, batch_size=4, num_workers=4):
    """
    Evaluate the alignment between visual and text features
    
    Args:
        model_path: Path to the trained model
        csv_path: Path to the CSV file
        frames_root_dir: Root directory containing frames
        output_dir: Directory to save results
        batch_size: Batch size for the dataloader
        num_workers: Number of worker threads for loading data
    """
    # Load the model
    logger.info(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    model = SignLanguageModel(hidden_dim=1024, freeze_bert=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Create dataloader
    logger.info("Creating dataloader with parameters:")
    logger.info(f"  csv_path: {csv_path}")
    logger.info(f"  frames_root_dir: {frames_root_dir}")
    logger.info(f"  batch_size: {batch_size}")
    logger.info(f"  num_workers: {num_workers}")
        
    data_loader = create_dataloader(
        csv_path=csv_path,
        frames_root_dir=frames_root_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        validate_files=True
    )
    
    # Extract features
    logger.info("Extracting features")
    features_dir = os.path.join(output_dir, 'features')
    extract_features(model, data_loader, features_dir, mode='both', device=device)
    
    # Calculate alignment scores
    logger.info("Calculating alignment scores")
    visual_dir = os.path.join(features_dir, 'visual_features')
    text_dir = os.path.join(features_dir, 'text_features')
    
    results_file = os.path.join(output_dir, 'alignment_scores.txt')
    
    with open(results_file, 'w') as f:
        f.write("Folder\tGloss1_Similarity\tGloss2_Similarity\n")
        
        # Get all folders
        folders = [folder for folder in os.listdir(visual_dir) if os.path.isdir(os.path.join(visual_dir, folder))]
        
        for folder in tqdm(folders, desc="Calculating alignment scores"):
            # Load visual features
            visual_path = os.path.join(visual_dir, folder, 'visual_features.pt')
            
            # Load text features
            text_path1 = os.path.join(text_dir, folder, 'text_features_gloss1.pt')
            text_path2 = os.path.join(text_dir, folder, 'text_features_gloss2.pt')
            
            # Skip if any file is missing
            if not (os.path.exists(visual_path) and os.path.exists(text_path1) and os.path.exists(text_path2)):
                logger.warning(f"Missing features for folder {folder}")
                continue
            
            # Load features
            visual_features = torch.load(visual_path)
            text_features1 = torch.load(text_path1)
            text_features2 = torch.load(text_path2)
            
            # Calculate similarities
            similarity1 = calculate_similarity(visual_features, text_features1)
            similarity2 = calculate_similarity(visual_features, text_features2)
            
            # Write to file
            f.write(f"{folder}\t{similarity1:.4f}\t{similarity2:.4f}\n")
    
    logger.info(f"Alignment scores saved to {results_file}")
    
    # Run analysis
    try:
        # Import the analysis module
        from analyze_results import analyze_alignment_results
        
        # Run analysis
        analysis_dir = os.path.join(output_dir, 'analysis')
        analyze_alignment_results(results_file, analysis_dir)
        logger.info(f"Analysis completed and saved to {analysis_dir}")
    except ImportError:
        logger.info("analyze_results.py not found. Skipping analysis.")
        logger.info(f"To analyze results, run: python analyze_results.py --results_file {results_file} --output_dir {output_dir}/analysis")
    except Exception as e:
        logger.error(f"Error running analysis: {e}")
        logger.info(f"To analyze results manually, run: python analyze_results.py --results_file {results_file} --output_dir {output_dir}/analysis")

def main():
    """Main function for feature extraction and evaluation"""
    
    parser = argparse.ArgumentParser(description='Extract features and evaluate alignment')
    
    # Model path
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    
    # Data paths
    parser.add_argument('--csv_path', type=str, 
                        default="/home/pvvkishore/Desktop/IEEE_THMS_May10/Phonix_Train/train_gloss_eng.csv",
                        help='Path to the CSV file')
    parser.add_argument('--frames_root_dir', type=str, 
                        default="/home/pvvkishore/Desktop/IEEE_THMS_May10/Phonix_Train/train/",
                        help='Root directory containing frames')
    
    # Output directory
    parser.add_argument('--output_dir', type=str, default='./results', help='Directory to save results')
    
    # Extraction parameters
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for data loading')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--mode', type=str, default='both', choices=['visual', 'text', 'both'], 
                       help='Which features to extract')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Log all arguments
    logger.info("Extracting features with the following parameters:")
    for arg in vars(args):
        logger.info(f"  {arg}: {getattr(args, arg)}")
    
    # Evaluate alignment
    evaluate_alignment(
        model_path=args.model_path,
        csv_path=args.csv_path,
        frames_root_dir=args.frames_root_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

if __name__ == '__main__':
    main()