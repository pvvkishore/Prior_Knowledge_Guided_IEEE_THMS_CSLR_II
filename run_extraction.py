import os
import sys
import argparse
import logging
import datetime
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'logs/extraction_log_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

def run_extraction(args):
    """
    Run the feature extraction process
    
    Args:
        args: Command line arguments
    """
    # Create necessary directories
    os.makedirs('results', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Set PYTHONPATH
    sys.path.append(os.getcwd())
    
    # Import the feature extraction module
    from feature_extraction import main as extraction_main
    
    # Log the start of extraction
    logger.info("Starting feature extraction and evaluation")
    logger.info(f"Arguments: {args}")
    
    # Use default paths if not provided
    if args.csv_path is None:
        args.csv_path = "/home/pvvkishore/Desktop/IEEE_THMS_May10/Phonix_Train/train_gloss_eng.csv"
        logger.info(f"No CSV path provided, using default: {args.csv_path}")
    
    if args.frames_root_dir is None:
        args.frames_root_dir = "/home/pvvkishore/Desktop/IEEE_THMS_May10/Phonix_Train/train/"
        logger.info(f"No frames root directory provided, using default: {args.frames_root_dir}")
    
    # Set command line arguments for the extraction script
    sys.argv = [
        'feature_extraction.py',
        f'--model_path={args.model_path}',
        f'--csv_path={args.csv_path}',
        f'--frames_root_dir={args.frames_root_dir}',
        f'--output_dir={args.output_dir}',
        f'--batch_size={args.batch_size}',
        f'--num_workers={args.num_workers}',
        f'--mode={args.mode}'
    ]
    
    # Run the extraction
    extraction_main()
    
    logger.info("Feature extraction on training set completed")
    
    # Check if test set exists
    test_csv_path = args.csv_path.replace('train', 'test')
    test_frames_dir = args.frames_root_dir.replace('train', 'test')
    
    if os.path.exists(test_csv_path):
        logger.info("Test set found. Running evaluation on test set...")
        
        # Create test output directory
        test_output_dir = os.path.join(args.output_dir, 'test')
        os.makedirs(test_output_dir, exist_ok=True)
        
        # Set command line arguments for test set
        sys.argv = [
            'feature_extraction.py',
            f'--model_path={args.model_path}',
            f'--csv_path={test_csv_path}',
            f'--frames_root_dir={test_frames_dir}',
            f'--output_dir={test_output_dir}',
            f'--batch_size={args.batch_size}',
            f'--num_workers={args.num_workers}',
            f'--mode={args.mode}'
        ]
        
        # Run the extraction on test set
        extraction_main()
        logger.info("Feature extraction on test set completed")
    else:
        logger.info("Test set not found. Skipping test evaluation.")
    
    logger.info("All feature extraction and evaluation completed")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run feature extraction and evaluation')
    
    # Model path
    parser.add_argument('--model_path', type=str, 
                        default="./saved_models/best_model_epoch_45.pth",
                        help='Path to the trained model')
    
    # Data paths (optional - will use defaults from data_loader.py if not provided)
    parser.add_argument('--csv_path', type=str, 
                        default=None,
                        help='Path to the CSV file (optional - will use default from data_loader.py)')
    parser.add_argument('--frames_root_dir', type=str, 
                        default=None,
                        help='Root directory containing frames (optional - will use default from data_loader.py)')
    
    # Output directory
    parser.add_argument('--output_dir', type=str, default='./results', 
                        help='Directory to save results')
    
    # Extraction parameters
    parser.add_argument('--batch_size', type=int, default=4, 
                        help='Batch size for data loading')
    parser.add_argument('--num_workers', type=int, default=4, 
                        help='Number of workers for data loading')
    parser.add_argument('--mode', type=str, default='both',
                        choices=['visual', 'text', 'both'],
                        help='Which features to extract')
    
    return parser.parse_args()

if __name__ == '__main__':
    # Parse arguments
    args = parse_arguments()
    
    # Set command line arguments for the extraction script
    sys.argv = ['feature_extraction.py']
    
    # Add model path
    sys.argv.append(f'--model_path={args.model_path}')
    
    # Add data paths only if provided
    if args.csv_path:
        sys.argv.append(f'--csv_path={args.csv_path}')
    if args.frames_root_dir:
        sys.argv.append(f'--frames_root_dir={args.frames_root_dir}')
    
    # Add other parameters
    sys.argv.extend([
        f'--output_dir={args.output_dir}',
        f'--batch_size={args.batch_size}',
        f'--num_workers={args.num_workers}',
        f'--mode={args.mode}'
    ])
    
    # Run extraction
    run_extraction(args)