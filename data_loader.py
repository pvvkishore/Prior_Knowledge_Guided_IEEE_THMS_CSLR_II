import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VisionLanguageDataset(Dataset):
    """
    Dataset class for loading frame images and their corresponding text glosses.
    
    Args:
        csv_path (str): Path to the CSV file containing annotations
        frames_root_dir (str): Root directory containing the 'train' folder with subfolders
        transform (callable, optional): Transform to be applied to the images
        image_format (str, optional): Format of the image files (default: 'jpg')
        validate_files (bool, optional): Whether to validate if files exist (default: True)
    """
    def __init__(self, csv_path, frames_root_dir, transform=None, image_format=None, validate_files=True):
        self.annotations = pd.read_csv(csv_path)
        self.frames_root_dir = Path(frames_root_dir)
        self.train_dir = self.frames_root_dir / 'train'  # All subfolders are in the train folder
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # If image_format is None, we'll try multiple formats
        self.image_format = image_format
        self.image_formats = ['jpg', 'jpeg', 'png'] if image_format is None else [image_format]
        
        # Basic validation of the CSV structure
        required_columns = 3
        if len(self.annotations.columns) < required_columns:
            raise ValueError(f"CSV file should have at least {required_columns} columns: folder path, gloss1, gloss2")
        
        # Name the columns for clarity
        if self.annotations.columns[0] == '0' and self.annotations.columns[1] == '1' and self.annotations.columns[2] == '2':
            self.annotations.columns = ['folder_path', 'gloss1', 'gloss2'] + list(self.annotations.columns[3:])
        
        # Create a mapping from folder names to their paths
        self.folder_path_map = self._create_folder_path_map()
        
        # Validate that files exist
        if validate_files:
            self._validate_files()
    
    def _create_folder_path_map(self):
        """Create a mapping from folder names to their full paths."""
        folder_path_map = {}
        
        # Make sure train directory exists
        if not self.train_dir.exists():
            logger.error(f"Train directory not found: {self.train_dir}")
            logger.info(f"Available directories in {self.frames_root_dir}: {[d.name for d in self.frames_root_dir.iterdir() if d.is_dir()]}")
            
            # Special handling: if 'train' doesn't exist but there's only one directory, use that
            root_dirs = [d for d in self.frames_root_dir.iterdir() if d.is_dir()]
            if len(root_dirs) == 1:
                logger.info(f"Using {root_dirs[0]} as train directory")
                self.train_dir = root_dirs[0]
            else:
                # If train directory doesn't exist, try using the frames_root_dir directly
                logger.info(f"Falling back to using root directory: {self.frames_root_dir}")
                self.train_dir = self.frames_root_dir
        
        # Get all subdirectories in the train folder
        try:
            subdirs = [d for d in self.train_dir.iterdir() if d.is_dir()]
            logger.info(f"Found {len(subdirs)} subdirectories in {self.train_dir}")
            
            # Print a few examples
            if subdirs:
                logger.info(f"Example subdirectories: {[d.name for d in subdirs[:5]]}")
            
            for subdir in subdirs:
                folder_name = subdir.name
                folder_path_map[folder_name] = subdir
        except Exception as e:
            logger.error(f"Error listing subdirectories in {self.train_dir}: {e}")
        
        if not folder_path_map:
            logger.warning("No folders found! Attempting direct mapping from CSV entries")
            # If no folders were found, try direct mapping
            for idx, row in self.annotations.iterrows():
                folder_name = row[0]  # First column contains folder name
                # Try both with and without train directory
                potential_paths = [
                    self.train_dir / folder_name,
                    self.frames_root_dir / folder_name,
                    Path(folder_name)  # Try absolute path
                ]
                
                for path in potential_paths:
                    if path.exists() and path.is_dir():
                        folder_path_map[folder_name] = path
                        break
            
            logger.info(f"Created {len(folder_path_map)} direct folder mappings")
        
        return folder_path_map
    
    def _get_all_image_files(self, directory):
        """Get all image files in a directory recursively."""
        all_files = []
        
        # Try to list contents of the directory
        try:
            contents = list(directory.iterdir())
            logger.debug(f"Directory {directory} contains {len(contents)} items")
        except Exception as e:
            logger.error(f"Cannot list contents of {directory}: {e}")
            return all_files
        
        # Helper function for recursive search
        def search_directory(dir_path, depth=0):
            if depth > 3:  # Limit recursion depth
                return []
                
            files = []
            try:
                # Try all common image formats if none specified
                formats = self.image_formats if self.image_formats else ['jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG']
                
                for fmt in formats:
                    # Try both lowercase and uppercase extensions
                    pattern = f"*.{fmt}"
                    files.extend(list(dir_path.glob(pattern)))
                
                # Check subdirectories
                for item in dir_path.iterdir():
                    if item.is_dir():
                        subdir_files = search_directory(item, depth + 1)
                        files.extend(subdir_files)
            except Exception as e:
                logger.error(f"Error searching directory {dir_path}: {e}")
            
            return files
        
        # Start recursive search
        all_files = search_directory(directory)
        
        # If we found files, log some information
        if all_files:
            logger.debug(f"Found {len(all_files)} image files in {directory}")
            if len(all_files) > 0:
                logger.debug(f"First few images: {[f.name for f in all_files[:3]]}")
        else:
            # If no files found, try to see what's in the directory
            try:
                all_items = list(directory.glob("*"))
                if all_items:
                    logger.info(f"Directory {directory} contains {len(all_items)} items but no images")
                    logger.info(f"Examples: {[item.name for item in all_items[:5]]}")
                    
                    # Check if there are subdirectories
                    subdirs = [item for item in all_items if item.is_dir()]
                    if subdirs:
                        logger.info(f"Found {len(subdirs)} subdirectories")
                        # Check first few subdirectories for images
                        for subdir in subdirs[:3]:
                            try:
                                subdir_items = list(subdir.glob("*"))
                                if subdir_items:
                                    logger.info(f"Subdir {subdir.name} contains: {[item.name for item in subdir_items[:5]]}")
                            except Exception as e:
                                logger.error(f"Error listing subdir {subdir}: {e}")
            except Exception as e:
                logger.error(f"Error listing all items in {directory}: {e}")
        
        return all_files
    
    def _validate_files(self):
        """Check if all frame files exist for the given folder paths in the CSV."""
        logger.info("Validating frame files exist...")
        missing_folders = []
        found_folders = []
        empty_folders = []
        
        # Check all folder names in the CSV against our mapping
        for idx, row in self.annotations.iterrows():
            folder_name = row[0]  # First column contains folder name
            
            if folder_name not in self.folder_path_map:
                missing_folders.append(folder_name)
                continue
                
            # Verify if frames exist
            frame_dir = self.folder_path_map[folder_name]
            frame_files = self._get_all_image_files(frame_dir)
            
            if not frame_files:
                empty_folders.append(folder_name)
            else:
                found_folders.append(folder_name)
                # Log details for the first few folders
                if len(found_folders) <= 5:
                    logger.info(f"Folder '{folder_name}' contains {len(frame_files)} frames")
                    logger.info(f"Example frames: {[f.name for f in frame_files[:3]]}")
        
        if missing_folders:
            logger.warning(f"Found {len(missing_folders)} missing directories in CSV")
            logger.warning(f"First few missing: {missing_folders[:5]}")
        
        if empty_folders:
            logger.warning(f"Found {len(empty_folders)} folders with no frames")
            logger.warning(f"First few empty: {empty_folders[:5]}")
        
        # Print some statistics
        logger.info(f"Found {len(found_folders)} folders with frames out of {len(self.annotations)} entries in CSV")
    
    def __len__(self):
        return len(self.annotations)
    
    def _get_frames_from_folder(self, folder_name):
        """Get all frame files from a folder, sorted by frame number."""
        if folder_name not in self.folder_path_map:
            # If the folder is not in our mapping, try to find it on-the-fly
            frame_dir = None
            potential_paths = [
                self.train_dir / folder_name,
                self.frames_root_dir / folder_name,
                Path(folder_name)  # Try absolute path
            ]
            
            for path in potential_paths:
                if path.exists() and path.is_dir():
                    frame_dir = path
                    self.folder_path_map[folder_name] = path  # Update mapping
                    logger.info(f"Found folder on-the-fly: {folder_name} -> {path}")
                    break
            
            if frame_dir is None:
                logger.error(f"Folder not found in mapping or on disk: {folder_name}")
                return []
        else:
            frame_dir = self.folder_path_map[folder_name]
        
        # Get all image files with any of the supported formats
        frame_files = self._get_all_image_files(frame_dir)
        
        # Log detailed info for debugging
        logger.debug(f"Found {len(frame_files)} frames in {folder_name} at {frame_dir}")
        if not frame_files:
            # Try to list all files in the directory to see what's there
            try:
                all_files = list(frame_dir.glob("*"))
                logger.info(f"Directory contents for {folder_name}: {[f.name for f in all_files[:10]]}")
                
                # Check subdirectories
                subdirs = [d for d in frame_dir.iterdir() if d.is_dir()]
                if subdirs:
                    logger.info(f"Found {len(subdirs)} subdirectories in {folder_name}")
                    for subdir in subdirs[:3]:  # Check first 3 subdirectories
                        subdir_files = list(subdir.glob("*"))
                        logger.info(f"Files in {subdir.name}: {[f.name for f in subdir_files[:5]]}")
            except Exception as e:
                logger.error(f"Error listing directory contents: {e}")
        
        # Sort frames by number when possible
        frame_files = sorted(frame_files, key=lambda x: int(x.stem) if x.stem.isdigit() else x.stem)
        return frame_files
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Returns:
            dict: A dictionary containing:
                - 'frames': tensor of shape [num_frames, channels, height, width]
                - 'gloss1': first text gloss
                - 'gloss2': second text gloss
                - 'folder_name': name of the frame folder
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get row data
        row = self.annotations.iloc[idx]
        folder_name = row[0]  # First column contains folder name
        gloss1 = row[1]       # Second column contains first gloss
        gloss2 = row[2]       # Third column contains second gloss
        
        # Get frames from the folder
        frame_files = self._get_frames_from_folder(folder_name)
        
        if not frame_files:
            logger.warning(f"No frames found for folder: {folder_name}")
            # Return an empty placeholder with debugging info
            empty_tensor = torch.zeros((1, 3, 224, 224))
            return {
                'frames': empty_tensor,
                'gloss1': gloss1,
                'gloss2': gloss2,
                'folder_name': folder_name,
                'num_frames': 0,
                'error': f"No frames found for folder: {folder_name}"
            }
        
        # Print verification info for debugging
        if idx < 5 or idx % 1000 == 0:  # First few and occasional samples
            logger.info(f"Loading sample {idx}: Folder={folder_name}, Gloss1={gloss1[:30]}...")
            logger.info(f"  Found {len(frame_files)} frames")
            logger.info(f"  First frame path: {frame_files[0]}")
        
        # Load and transform frames
        frames = []
        frame_paths = []  # Store frame paths for verification
        
        for frame_file in frame_files:
            try:
                image = Image.open(frame_file).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                frames.append(image)
                frame_paths.append(str(frame_file))
            except Exception as e:
                logger.error(f"Error loading image {frame_file}: {e}")
                # Continue with other frames if one fails
                continue
        
        # Stack frames into a tensor
        if frames:
            frames_tensor = torch.stack(frames)
        else:
            # Return an empty tensor if no frames were loaded
            frames_tensor = torch.zeros((1, 3, 224, 224))
            logger.warning(f"No valid frames in {folder_name}, returning empty tensor")
        
        # Verification step: Check if loaded frames match the expected folder
        if frame_paths:
            first_frame_path = Path(frame_paths[0])
            actual_folder_name = first_frame_path.parent.name
            
            # If the folder name doesn't match, go up one level to check parent directory
            if actual_folder_name != folder_name:
                # Try parent directory
                parent_folder_name = first_frame_path.parent.parent.name
                if parent_folder_name == folder_name:
                    actual_folder_name = parent_folder_name
            
            # Debug print for verification
            if idx < 5:
                logger.info(f"  Verification - CSV Folder: {folder_name}, Actual Folder: {actual_folder_name}")
            
            if actual_folder_name != folder_name:
                logger.info(f"Folder name mismatch! CSV: {folder_name}, Actual: {actual_folder_name}")
                logger.info(f"Full path: {first_frame_path}")
        
        return {
            'frames': frames_tensor,
            'gloss1': gloss1,
            'gloss2': gloss2,
            'folder_name': folder_name,
            'num_frames': len(frames),
            'frame_paths': frame_paths[:5] if frames else []  # Include paths for first 5 frames for debugging
        }


def create_dataloader(csv_path, frames_root_dir, batch_size=8, num_workers=4, 
                     shuffle=True, image_format=None, validate_files=True):
    """
    Create a DataLoader for the vision-language dataset.
    
    Args:
        csv_path (str): Path to the CSV file containing annotations
        frames_root_dir (str): Root directory containing the 'train' folder with frame subfolders
        batch_size (int): Batch size for the dataloader
        num_workers (int): Number of worker threads for loading data
        shuffle (bool): Whether to shuffle the dataset
        image_format (str): Format of the image files (if None, tries multiple formats)
        validate_files (bool): Whether to validate if files exist
        
    Returns:
        DataLoader: A PyTorch DataLoader for the dataset
    """
    # Set higher logging level for initial scan
    logger.setLevel(logging.INFO)
    
    logger.info(f"Creating dataset with csv_path={csv_path}, frames_root_dir={frames_root_dir}")
    
    # First try to check if the paths exist
    csv_file = Path(csv_path)
    frames_dir = Path(frames_root_dir)
    
    if not csv_file.exists():
        logger.error(f"CSV file not found: {csv_file}")
        available_files = list(Path('.').glob("*.csv"))
        if available_files:
            logger.info(f"Available CSV files: {[f.name for f in available_files]}")
    
    if not frames_dir.exists():
        logger.error(f"Frames directory not found: {frames_dir}")
        available_dirs = [d for d in Path('.').iterdir() if d.is_dir()]
        if available_dirs:
            logger.info(f"Available directories: {[d.name for d in available_dirs]}")
    
    # Create the dataset
    dataset = VisionLanguageDataset(
        csv_path=csv_path,
        frames_root_dir=frames_root_dir,
        image_format=image_format,
        validate_files=validate_files
    )
    
    # Check if dataset is empty
    if len(dataset) == 0:
        logger.warning("Dataset is empty! Please check your paths and data.")
    
    # Create the dataloader with robust error handling
    logger.info(f"Creating DataLoader with batch_size={batch_size}, num_workers={num_workers}")
    
    def safe_collate(batch):
        """Wrapper for collate function to handle unexpected errors"""
        try:
            return collate_variable_length(batch)
        except Exception as e:
            logger.error(f"Error in collate function: {e}")
            # Return a minimal valid batch structure
            return {
                'frames': torch.zeros((0, 1, 3, 224, 224)),
                'frame_mask': torch.zeros((0, 1), dtype=torch.bool),
                'gloss1': [],
                'gloss2': [],
                'folder_names': [],
                'num_frames': torch.zeros(0, dtype=torch.int),
                'error': str(e)
            }
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=safe_collate
    )
    
    logger.info(f"Created dataloader with {len(dataset)} samples")
    return dataloader


def collate_variable_length(batch):
    """
    Custom collate function to handle variable number of frames.
    Pads or truncates the frame sequences to a fixed length.
    
    Args:
        batch (list): List of samples from the dataset
        
    Returns:
        dict: A dictionary with batched data
    """
    # Filter out samples with errors
    valid_batch = [sample for sample in batch if 'error' not in sample]
    
    if not valid_batch:
        # Return empty batch with expected structure
        return {
            'frames': torch.zeros((0, 0, 3, 224, 224)),
            'frame_mask': torch.zeros((0, 0), dtype=torch.bool),
            'gloss1': [],
            'gloss2': [],
            'folder_names': [],
            'num_frames': torch.zeros(0, dtype=torch.int),
        }
    
    # Find max number of frames in this batch
    max_frames = max([sample['num_frames'] for sample in valid_batch])
    
    # Set a reasonable maximum to avoid excessive memory usage
    max_frames = min(max_frames, 8)  # Limit to 32 frames
    
    # Initialize tensors
    batch_size = len(valid_batch)
    frame_shape = valid_batch[0]['frames'].shape[1:]  # [channels, height, width]
    
    # Create tensors
    frames_tensor = torch.zeros((batch_size, max_frames, *frame_shape))
    frame_mask = torch.zeros((batch_size, max_frames), dtype=torch.bool)
    
    # Collect text data
    gloss1 = []
    gloss2 = []
    folder_names = []
    num_frames = []
    
    for i, sample in enumerate(valid_batch):
        sample_frames = sample['frames']
        n_frames = min(sample['num_frames'], max_frames)
        
        # Copy frames to output tensor (either all or truncated)
        frames_tensor[i, :n_frames] = sample_frames[:n_frames]
        frame_mask[i, :n_frames] = 1  # Mark valid frames
        
        # Collect text data
        gloss1.append(sample['gloss1'])
        gloss2.append(sample['gloss2'])
        folder_names.append(sample['folder_name'])
        num_frames.append(n_frames)
    
    return {
        'frames': frames_tensor,          # Shape: [batch_size, max_frames, channels, height, width]
        'frame_mask': frame_mask,         # Shape: [batch_size, max_frames]
        'gloss1': gloss1,                 # List of strings
        'gloss2': gloss2,                 # List of strings
        'folder_names': folder_names,     # List of strings
        'num_frames': torch.tensor(num_frames)  # Tensor of frame counts
    }


# Example usage
if __name__ == "__main__":
    # Example paths
    csv_path = "/home/pvvkishore/Desktop/IEEE_THMS_May10/Phonix_Train/train_gloss_eng.csv"
    frames_root_dir = "/home/pvvkishore/Desktop/IEEE_THMS_May10/Phonix_Train/train/"
    
    # Create dataloader with improved error handling
    dataloader = create_dataloader(
        csv_path=csv_path,
        frames_root_dir=frames_root_dir,
        batch_size=4,
        validate_files=True,
        image_format=None  # Try multiple formats
    )
    
    # Test the dataloader with extensive verification
    print("\n--- Testing DataLoader ---")
    for i, batch in enumerate(dataloader):
        print(f"Batch {i}:")
        print(f"  Frames shape: {batch['frames'].shape}")
        print(f"  Frame mask shape: {batch['frame_mask'].shape}")
        print(f"  Number of frames: {batch['num_frames']}")
        print(f"  First gloss examples: {batch['gloss1'][:2]}")
        print(f"  Second gloss examples: {batch['gloss2'][:2]}")
        print(f"  Folder names: {batch['folder_names'][:2]}")
        
        # Only check first batch
        if i == 0:
            break
            
    # Standalone verification function to check if CSV entries match folder structure
    def verify_dataset_structure(csv_path, frames_root_dir):
        """Verify that CSV entries match the actual folder structure."""
        print("\n--- Dataset Structure Verification ---")
        frames_root = Path(frames_root_dir)
        
        # Try to find train directory
        train_dir = frames_root / 'train'
        if not train_dir.exists():
            print(f"WARNING: Train directory not found at {train_dir}")
            print(f"Available directories in {frames_root}: {[d.name for d in frames_root.iterdir() if d.is_dir()]}")
            
            # If train directory doesn't exist but there's only one directory, use that
            root_dirs = [d for d in frames_root.iterdir() if d.is_dir()]
            if len(root_dirs) == 1:
                train_dir = root_dirs[0]
                print(f"Using {train_dir} as train directory")
            else:
                train_dir = frames_root
                print(f"Falling back to using root directory: {train_dir}")
        
        # Get all actual folders
        actual_folders = {}
        
        # Use a recursive approach to find all potential folders
        def scan_directory(directory, depth=0, max_depth=3):
            if depth > max_depth:
                return
                
            try:
                for item in directory.iterdir():
                    if item.is_dir():
                        # Store both the name and full path
                        actual_folders[item.name] = item
                        # Recursively scan subdirectories
                        scan_directory(item, depth + 1, max_depth)
            except Exception as e:
                print(f"Error scanning directory {directory}: {e}")
        
        # Scan from the train directory
        scan_directory(train_dir)
        
        print(f"Found {len(actual_folders)} actual folders/subfolders")
        if actual_folders:
            print(f"Example folders: {list(actual_folders.keys())[:5]}")
        
        # Get all folders mentioned in CSV
        df = pd.read_csv(csv_path)
        csv_folders = set(df.iloc[:, 0])  # First column contains folder names
        print(f"Found {len(csv_folders)} folders mentioned in CSV")
        if csv_folders:
            print(f"Example CSV folders: {list(csv_folders)[:5]}")
        
        # Check overlap
        common = set(csv_folders).intersection(set(actual_folders.keys()))
        missing_in_disk = set(csv_folders) - set(actual_folders.keys())
        
        print(f"Folders present in both CSV and disk: {len(common)}")
        print(f"Folders in CSV but missing on disk: {len(missing_in_disk)}")
        
        if missing_in_disk and actual_folders:
            print("\nInvestigating missing folders...")
            for missing_folder in list(missing_in_disk)[:3]:  # Check first 3 missing folders
                print(f"\nChecking for folder similar to: {missing_folder}")
                
                # Check if the folder might exist with slight name variations
                close_matches = []
                for actual_folder in actual_folders:
                    # Simple similarity check (if one is a subset of the other)
                    if missing_folder in actual_folder or actual_folder in missing_folder:
                        close_matches.append((actual_folder, actual_folders[actual_folder]))
                
                if close_matches:
                    print(f"  Potential matches for {missing_folder}:")
                    for match, path in close_matches:
                        print(f"  - {match} at {path}")
                        
                        # Check if this folder contains image files
                        formats = ['jpg', 'jpeg', 'png']
                        found_images = False
                        for fmt in formats:
                            images = list(path.glob(f"*.{fmt}"))
                            if images:
                                print(f"    Found {len(images)} {fmt} images")
                                found_images = True
                                break
                        
                        if not found_images:
                            print(f"    No images found with formats: {formats}")
                            # Check subdirectories
                            subdirs = [d for d in path.iterdir() if d.is_dir()]
                            if subdirs:
                                print(f"    Contains {len(subdirs)} subdirectories")
                                for subdir in subdirs[:2]:  # Check first 2
                                    for fmt in formats:
                                        images = list(subdir.glob(f"*.{fmt}"))
                                        if images:
                                            print(f"    - {subdir.name}: {len(images)} {fmt} images")
                else:
                    print(f"  No similar folders found for {missing_folder}")