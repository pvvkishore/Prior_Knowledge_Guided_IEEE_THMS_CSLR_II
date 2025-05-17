#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 10 11:34:05 2025

@author: pvvkishore
"""
import csv
import os

def create_train_gloss_csv(input_file="/home/pvvkishore/Desktop/IEEE_THMS_May10/Phonix_Train/annotations/output_split.csv", 
                           output_file="/home/pvvkishore/Desktop/IEEE_THMS_May10/Phonix_Train/annotations/train_gloss.csv"):
    """
    Processes all rows in output_split.csv and creates train_gloss.csv
    
    Args:
        input_file (str): Path to the split CSV file (default: "output_split.csv")
        output_file (str): Path to the output train gloss file (default: "train_gloss.csv")
    """
    
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        print("Please make sure the output_split.csv file exists.")
        return False
    
    processed_rows = []
    headers = ["Extracted_B", "Column_F", "Column_G"]
    skipped_rows = 0
    
    # Read the split CSV file
    with open(input_file, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        
        for row_num, row in enumerate(reader, 1):
            if len(row) >= 7:  # Make sure we have at least 7 columns (G column)
                # Extract from column B (index 1)
                column_b = row[1]
                
                # Extract the part before the '/' delimiter
                if '/' in column_b:
                    extracted_b = column_b.split('/')[0]
                else:
                    extracted_b = column_b  # If no '/', keep the whole value
                
                # Extract column F (index 5) and column G (index 6)
                column_f = row[5]
                column_g = row[6]
                
                # Create new row with extracted data
                new_row = [extracted_b, column_f, column_g]
                processed_rows.append(new_row)
            else:
                skipped_rows += 1
                print(f"Skipping row {row_num} (insufficient columns): {row}")
    
    # Write the processed data to train_gloss.csv
    with open(output_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        # Write headers
        writer.writerow(headers)
        
        # Write all data rows
        writer.writerows(processed_rows)
    
    print(f"\nProcessing completed!")
    print(f"Successfully processed {len(processed_rows)} rows")
    print(f"Skipped {skipped_rows} rows due to insufficient columns")
    print(f"Output saved to: {output_file}")
    
    # Display first few rows for verification
    print(f"\nFirst 5 rows of {output_file}:")
    with open(output_file, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for i, row in enumerate(reader):
            if i <= 5:  # Show headers and first 5 data rows
                print(f"Row {i}: {row}")
            else:
                break
    
    print("...")
    return True

def verify_output(output_file="train_gloss.csv"):
    """Verify the output file and show statistics"""
    if not os.path.exists(output_file):
        print(f"Output file '{output_file}' not found.")
        return
    
    with open(output_file, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        rows = list(reader)
        
    total_rows = len(rows) - 1  # Exclude header
    
    print(f"\nVerification of {output_file}:")
    print(f"Total rows (excluding headers): {total_rows}")
    print(f"Columns: {rows[0] if rows else 'None'}")
    
    # Show a few sample rows
    if total_rows > 0:
        print(f"\nLast 3 rows:")
        for i in range(max(1, len(rows) - 3), len(rows)):
            print(f"Row {i}: {rows[i]}")

# Example function to view specific columns
def view_column_samples(output_file="train_gloss.csv", column_index=1, num_samples=10):
    """View samples of a specific column"""
    if not os.path.exists(output_file):
        print(f"Output file '{output_file}' not found.")
        return
    
    column_names = ["Extracted_B", "Column_F", "Column_G"]
    
    with open(output_file, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        header = next(reader)  # Skip header
        
        print(f"\nSample values from {column_names[column_index]} column:")
        for i, row in enumerate(reader):
            if i < num_samples and len(row) > column_index:
                print(f"{i+1}. {row[column_index]}")
            elif i >= num_samples:
                break

# Main execution
if __name__ == "__main__":
    # Process the split CSV and create train_gloss.csv
    success = create_train_gloss_csv()
    
    if success:
        # Verify the output
        verify_output()
        
        # Show some sample values from each column
        print("\n" + "="*50)
        view_column_samples(column_index=0, num_samples=5)  # Extracted_B samples
        view_column_samples(column_index=1, num_samples=5)  # Column_F samples
        view_column_samples(column_index=2, num_samples=5)  # Column_G samples

# Standalone usage
"""
To use this script:
1. Make sure output_split.csv exists in the same directory
2. Run: python create_train_gloss.py

The script will:
- Process all rows in output_split.csv
- Extract the required data from columns B, F, and G
- Create train_gloss.csv with the processed data
"""