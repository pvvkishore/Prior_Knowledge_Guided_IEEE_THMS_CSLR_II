#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 14 14:57:04 2025

@author: pvvkishore
"""

import os
import sys
import argparse
import logging
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def analyze_alignment_results(results_file, output_dir):
    """
    Analyze alignment results and create visualizations
    
    Args:
        results_file: Path to the alignment scores file
        output_dir: Directory to save visualizations
    """
    logger.info(f"Analyzing alignment results from {results_file}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the results file
    try:
        df = pd.read_csv(results_file, sep='\t')
        logger.info(f"Loaded {len(df)} alignment scores")
    except Exception as e:
        logger.error(f"Error reading results file: {e}")
        return
    
    # Basic statistics
    stats = {
        'gloss1_mean': df['Gloss1_Similarity'].mean(),
        'gloss1_std': df['Gloss1_Similarity'].std(),
        'gloss1_min': df['Gloss1_Similarity'].min(),
        'gloss1_max': df['Gloss1_Similarity'].max(),
        'gloss2_mean': df['Gloss2_Similarity'].mean(),
        'gloss2_std': df['Gloss2_Similarity'].std(),
        'gloss2_min': df['Gloss2_Similarity'].min(),
        'gloss2_max': df['Gloss2_Similarity'].max(),
    }
    
    # Save statistics
    with open(os.path.join(output_dir, 'alignment_statistics.txt'), 'w') as f:
        f.write("Alignment Statistics\n")
        f.write("===================\n\n")
        f.write(f"Number of samples: {len(df)}\n\n")
        f.write("Gloss1 Similarity:\n")
        f.write(f"  Mean: {stats['gloss1_mean']:.4f}\n")
        f.write(f"  Std: {stats['gloss1_std']:.4f}\n")
        f.write(f"  Min: {stats['gloss1_min']:.4f}\n")
        f.write(f"  Max: {stats['gloss1_max']:.4f}\n\n")
        f.write("Gloss2 Similarity:\n")
        f.write(f"  Mean: {stats['gloss2_mean']:.4f}\n")
        f.write(f"  Std: {stats['gloss2_std']:.4f}\n")
        f.write(f"  Min: {stats['gloss2_min']:.4f}\n")
        f.write(f"  Max: {stats['gloss2_max']:.4f}\n")
    
    # Create visualizations
    
    # 1. Histogram of similarity scores
    plt.figure(figsize=(10, 6))
    plt.hist(df['Gloss1_Similarity'], bins=30, alpha=0.5, label='Gloss1')
    plt.hist(df['Gloss2_Similarity'], bins=30, alpha=0.5, label='Gloss2')
    plt.xlabel('Similarity Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Visual-Text Similarity Scores')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'similarity_histogram.png'), dpi=300)
    plt.close()
    
    # 2. Box plot comparison
    plt.figure(figsize=(8, 6))
    data = [df['Gloss1_Similarity'], df['Gloss2_Similarity']]
    plt.boxplot(data, labels=['Gloss1', 'Gloss2'])
    plt.ylabel('Similarity Score')
    plt.title('Comparison of Visual-Text Similarity')
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'similarity_boxplot.png'), dpi=300)
    plt.close()
    
    # 3. Scatter plot comparing Gloss1 vs Gloss2 similarity
    plt.figure(figsize=(8, 8))
    plt.scatter(df['Gloss1_Similarity'], df['Gloss2_Similarity'], alpha=0.5)
    plt.xlabel('Gloss1 Similarity')
    plt.ylabel('Gloss2 Similarity')
    plt.title('Gloss1 vs Gloss2 Similarity')
    plt.grid(alpha=0.3)
    # Add diagonal line for reference
    min_val = min(df['Gloss1_Similarity'].min(), df['Gloss2_Similarity'].min())
    max_val = max(df['Gloss1_Similarity'].max(), df['Gloss2_Similarity'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'similarity_scatter.png'), dpi=300)
    plt.close()
    
    # 4. Calculate which gloss is better for each sample
    df['Better_Gloss'] = np.where(df['Gloss1_Similarity'] > df['Gloss2_Similarity'], 'Gloss1', 'Gloss2')
    gloss1_better = (df['Better_Gloss'] == 'Gloss1').sum()
    gloss2_better = (df['Better_Gloss'] == 'Gloss2').sum()
    
    # Create a pie chart
    plt.figure(figsize=(8, 8))
    plt.pie([gloss1_better, gloss2_better], labels=['Gloss1 Better', 'Gloss2 Better'], 
            autopct='%1.1f%%', startangle=90, colors=['#ff9999', '#66b3ff'])
    plt.title('Which Gloss Has Better Alignment with Visual Features')
    plt.savefig(os.path.join(output_dir, 'better_gloss_pie.png'), dpi=300)
    plt.close()
    
    # Save top and bottom samples
    df_sorted_gloss1 = df.sort_values(by='Gloss1_Similarity', ascending=False)
    df_sorted_gloss2 = df.sort_values(by='Gloss2_Similarity', ascending=False)
    
    # Top 10 best aligned samples
    top10_gloss1 = df_sorted_gloss1.head(10)[['Folder', 'Gloss1_Similarity']]
    top10_gloss2 = df_sorted_gloss2.head(10)[['Folder', 'Gloss2_Similarity']]
    
    # Bottom 10 worst aligned samples
    bottom10_gloss1 = df_sorted_gloss1.tail(10)[['Folder', 'Gloss1_Similarity']]
    bottom10_gloss2 = df_sorted_gloss2.tail(10)[['Folder', 'Gloss2_Similarity']]
    
    # Save to CSV
    top10_gloss1.to_csv(os.path.join(output_dir, 'top10_gloss1.csv'), index=False)
    top10_gloss2.to_csv(os.path.join(output_dir, 'top10_gloss2.csv'), index=False)
    bottom10_gloss1.to_csv(os.path.join(output_dir, 'bottom10_gloss1.csv'), index=False)
    bottom10_gloss2.to_csv(os.path.join(output_dir, 'bottom10_gloss2.csv'), index=False)
    
    logger.info(f"Analysis completed. Results saved to {output_dir}")

def main():
    """Main function for analyzing alignment results"""
    
    parser = argparse.ArgumentParser(description='Analyze alignment results')
    
    # Input file
    parser.add_argument('--results_file', type=str, required=True, 
                        help='Path to the alignment scores file')
    
    # Output directory
    parser.add_argument('--output_dir', type=str, default='./analysis', 
                        help='Directory to save analysis results')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Analyze results
    analyze_alignment_results(args.results_file, args.output_dir)

if __name__ == '__main__':
    main()