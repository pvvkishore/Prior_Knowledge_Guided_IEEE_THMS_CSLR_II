#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 10 10:44:51 2025

@author: pvvkishore
"""

import csv
import os

def process_csv_with_pipe_delimiter(input_file, output_file):
    """
    Processes a CSV file where cells contain pipe-delimited data and creates
    a new CSV with each pipe-separated value in its own cell.
    
    Args:
        input_file (str): Path to the input CSV file
        output_file (str): Path to the output CSV file
    """
    
    processed_rows = []
    
    # Read the input CSV file
    with open(input_file, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        
        for row in reader:
            # If the row has just one cell with pipe-delimited data
            if len(row) == 1 and '|' in row[0]:
                # Split by pipe and add to processed rows
                split_row = row[0].split('|')
                processed_rows.append(split_row)
            
            # If the row has multiple cells, check each for pipes
            elif len(row) > 1:
                new_row = []
                for cell in row:
                    if '|' in cell:
                        # If a cell contains pipes, we'll split it and extend the row
                        # Note: This might create rows with different lengths
                        split_cells = cell.split('|')
                        new_row.extend(split_cells)
                    else:
                        new_row.append(cell)
                processed_rows.append(new_row)
            
            # If the row has no pipes, keep it as-is
            else:
                processed_rows.append(row)
    
    # Write the processed data to a new CSV file
    with open(output_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(processed_rows)
    
    print(f"Successfully processed {len(processed_rows)} rows")
    print(f"Output saved to: {output_file}")

# Example usage:
if __name__ == "__main__":
    # Replace with your actual file paths
    input_csv = "/home/pvvkishore/Desktop/IEEE_THMS_May10/Phonix_Train/annotations/PHOENIX-2014-T.train.corpus.csv"  # Your input file
    output_csv = "/home/pvvkishore/Desktop/IEEE_THMS_May10/Phonix_Train/annotations/output_split.csv"  # Your output file
    
    # Check if input file exists
    if os.path.exists(input_csv):
        process_csv_with_pipe_delimiter(input_csv, output_csv)
    else:
        print(f"Input file '{input_csv}' not found. Please check the file path.")
        print("\nYou can also use this script by importing it:")
        print("from csv_pipe_splitter import process_csv_with_pipe_delimiter")
        print("process_csv_with_pipe_delimiter('your_input.csv', 'your_output.csv')")

# Alternative: Process a sample string directly
def process_sample_string():
    """Process the sample string you provided"""
    sample_data = "11August_2010_Wednesday_tagesschau-1|11August_2010_Wednesday_tagesschau-1/1/*.png|-1|-1|Signer08|JETZT WETTER MORGEN DONNERSTAG ZWOELF FEBRUAR|und nun die wettervorhersage für morgen donnerstag den zwölften august"
    
    # Split by pipe
    split_data = sample_data.split('|')
    
    # Display the result
    print("\nSample data split into columns:")
    for i, cell in enumerate(split_data, 1):
        print(f"Column {i}: {cell}")
    
    # Save as CSV
    with open('sample_output.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(split_data)
    
    print("\nSample data saved to 'sample_output.csv'")

# Uncomment the next line to test with your sample string
# process_sample_string()