#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 10 11:55:25 2025

@author: pvvkishore
"""

import csv
import os
import time
from typing import List, Dict

# Option 1: Using Google Translate API (requires googletrans library)
try:
    from googletrans import Translator
    GOOGLETRANS_AVAILABLE = True
except ImportError:
    GOOGLETRANS_AVAILABLE = False
    print("googletrans not installed. Install with: pip install googletrans==4.0.0-rc1")

# Option 2: Using DeepL API (requires deepl library)
try:
    import deepl
    DEEPL_AVAILABLE = True
except ImportError:
    DEEPL_AVAILABLE = False
    print("deepl not installed. Install with: pip install deepl")

def translate_with_googletrans(text: str, source_lang='de', target_lang='en') -> str:
    """
    Translate text using Google Translate (free)
    
    Args:
        text (str): Text to translate
        source_lang (str): Source language code (default: 'de' for German)
        target_lang (str): Target language code (default: 'en' for English)
    
    Returns:
        str: Translated text
    """
    if not GOOGLETRANS_AVAILABLE:
        return text  # Return original if library not available
    
    try:
        translator = Translator()
        result = translator.translate(text, src=source_lang, dest=target_lang)
        time.sleep(0.1)  # Small delay to avoid rate limiting
        return result.text
    except Exception as e:
        print(f"Translation error: {e}")
        return text  # Return original text if translation fails

def translate_with_deepl(text: str, auth_key: str, source_lang='DE', target_lang='EN') -> str:
    """
    Translate text using DeepL API (requires API key)
    
    Args:
        text (str): Text to translate
        auth_key (str): DeepL API authentication key
        source_lang (str): Source language code (default: 'DE' for German)
        target_lang (str): Target language code (default: 'EN' for English)
    
    Returns:
        str: Translated text
    """
    if not DEEPL_AVAILABLE:
        return text  # Return original if library not available
    
    try:
        translator = deepl.Translator(auth_key)
        result = translator.translate_text(text, source_lang=source_lang, target_lang=target_lang)
        return result.text
    except Exception as e:
        print(f"Translation error: {e}")
        return text  # Return original text if translation fails

def translate_gloss_file(input_file: str = "/home/pvvkishore/Desktop/IEEE_THMS_May10/Phonix_Train/annotations/train_gloss.csv", 
                        output_file: str = "/home/pvvkishore/Desktop/IEEE_THMS_May10/Phonix_Train/annotations/train_gloss_eng.csv",
                        translator_type: str = "googletrans",
                        deepl_api_key: str = None) -> bool:
    """
    Translate German text in train_gloss.csv to English and create train_gloss_eng.csv
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to output CSV file
        translator_type (str): Type of translator ('googletrans' or 'deepl')
        deepl_api_key (str): API key for DeepL (required if using DeepL)
    
    Returns:
        bool: True if successful, False otherwise
    """
    
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return False
    
    translated_rows = []
    headers = ["Extracted_B", "Column_F_English", "Column_G_English"]
    total_rows = 0
    
    # Read the input CSV file
    with open(input_file, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        header_row = next(reader)  # Skip header
        
        print(f"Processing {input_file}...")
        print("Translating German text to English...")
        
        for row_num, row in enumerate(reader, 1):
            if len(row) >= 3:
                # Column A (Extracted_B) - keep as is (identifier)
                extracted_b = row[0]
                
                # Column B (Column_F) - translate from German to English
                column_f_german = row[1]
                if translator_type == "deepl" and deepl_api_key:
                    column_f_english = translate_with_deepl(column_f_german, deepl_api_key)
                else:
                    column_f_english = translate_with_googletrans(column_f_german)
                
                # Column C (Column_G) - translate from German to English
                column_g_german = row[2]
                if translator_type == "deepl" and deepl_api_key:
                    column_g_english = translate_with_deepl(column_g_german, deepl_api_key)
                else:
                    column_g_english = translate_with_googletrans(column_g_german)
                
                # Create new row with translated data
                new_row = [extracted_b, column_f_english, column_g_english]
                translated_rows.append(new_row)
                
                # Progress indicator
                if row_num % 10 == 0:
                    print(f"Processed {row_num} rows...")
                
                total_rows += 1
    
    # Write the translated data to a new CSV file
    with open(output_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        # Write headers
        writer.writerow(headers)
        
        # Write all translated rows
        writer.writerows(translated_rows)
    
    print(f"\nTranslation completed!")
    print(f"Successfully translated {total_rows} rows")
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
    
    return True

def compare_translations(input_file: str = "/home/pvvkishore/Desktop/IEEE_THMS_May10/Phonix_Train/annotations/train_gloss.csv", 
                        output_file: str = "/home/pvvkishore/Desktop/IEEE_THMS_May10/Phonix_Train/annotations/train_gloss_eng.csv"):
    """
    Show a side-by-side comparison of original German and translated English text
    """
    if not os.path.exists(input_file) or not os.path.exists(output_file):
        print("Input or output file not found.")
        return
    
    # Read both files
    with open(input_file, 'r', encoding='utf-8') as f1, \
         open(output_file, 'r', encoding='utf-8') as f2:
        
        german_reader = csv.reader(f1)
        english_reader = csv.reader(f2)
        
        # Skip headers
        next(german_reader)
        next(english_reader)
        
        print("\nSide-by-side comparison (first 5 rows):")
        print("-" * 100)
        
        for i, (german_row, english_row) in enumerate(zip(german_reader, english_reader)):
            if i >= 5:
                break
            
            print(f"\nRow {i+1}:")
            print(f"German F:  {german_row[1]}")
            print(f"English F: {english_row[1]}")
            print(f"German G:  {german_row[2]}")
            print(f"English G: {english_row[2]}")
            print("-" * 50)

# Main execution
if __name__ == "__main__":
    # Option 1: Using Google Translate (free, no API key required)
    print("Starting translation process...")
    
    # Choose your translation method:
    # Method 1: Google Translate (free, but may have rate limits)
    success = translate_gloss_file(
        input_file="/home/pvvkishore/Desktop/IEEE_THMS_May10/Phonix_Train/annotations/train_gloss.csv",
        output_file="/home/pvvkishore/Desktop/IEEE_THMS_May10/Phonix_Train/annotations/train_gloss_eng.csv",
        translator_type="googletrans"
    )
    
    # Method 2: DeepL (better quality, requires API key)
    # Uncomment these lines if you want to use DeepL:
    """
    DEEPL_API_KEY = "your_deepl_api_key_here"  # Replace with your actual API key
    success = translate_gloss_file(
        input_file="train_gloss.csv",
        output_file="train_gloss_eng.csv",
        translator_type="deepl",
        deepl_api_key=DEEPL_API_KEY
    )
    """
    
    if success:
        # Show comparison of original and translated text
        compare_translations()

# Alternative: Manual translation dictionaries for common phrases
COMMON_TRANSLATIONS = {
    # Add common German phrases and their English translations
    "WETTER": "WEATHER",
    "MORGEN": "TOMORROW",
    "DONNERSTAG": "THURSDAY",
    "ZWOELF": "TWELVE",
    "FEBRUAR": "FEBRUARY",
    "JETZT": "NOW",
    "und nun": "and now",
    "die wettervorhersage": "the weather forecast",
    "für morgen": "for tomorrow",
    "den zwölften august": "the twelfth of august",
    # Add more translations as needed
}

def manual_translate(text: str, translation_dict: Dict[str, str]) -> str:
    """
    Simple manual translation using a dictionary
    For cases where you don't want to use external APIs
    """
    translated = text
    for german, english in translation_dict.items():
        translated = translated.replace(german, english)
    return translated

# Usage example for manual translation
"""
def translate_manually():
    with open("train_gloss.csv", 'r', encoding='utf-8') as f_in, \
         open("train_gloss_eng_manual.csv", 'w', newline='', encoding='utf-8') as f_out:
        
        reader = csv.reader(f_in)
        writer = csv.writer(f_out)
        
        headers = ["Extracted_B", "Column_F_English", "Column_G_English"]
        writer.writerow(headers)
        
        next(reader)  # Skip header
        
        for row in reader:
            if len(row) >= 3:
                translated_row = [
                    row[0],  # Keep Extracted_B as is
                    manual_translate(row[1], COMMON_TRANSLATIONS),
                    manual_translate(row[2], COMMON_TRANSLATIONS)
                ]
                writer.writerow(translated_row)
"""