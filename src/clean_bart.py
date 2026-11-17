"""
clean_bart.py - Minimal Text Cleaning for BART Summarization

This module performs very light text cleaning before passing the transcript to BART.

IMPORTANT: BART is a powerful language model that understands natural language.
It needs the text to be readable and grammatically intact.

DO NOT apply traditional NLP preprocessing here:
- No stopword removal
- No lemmatization
- No aggressive tokenization
- No lowercasing

We only clean up formatting issues:
1. Remove unnecessary extra whitespace
2. Remove unwanted special characters
3. Preserve sentence structure, punctuation, and meaning
"""

import re


def clean_text_for_bart(text):
    """
    Perform minimal text cleaning suitable for BART input.
    
    This keeps the text natural and readable — exactly what BART needs.
    
    Parameters:
        text (str): The raw transcript text
    
    Returns:
        str: The cleaned text, ready for BART
    """
    # Step 1: Replace multiple spaces with a single space
    cleaned = re.sub(r" +", " ", text)
    
    # Step 2: Replace multiple newlines with a single space
    # (Transcripts sometimes have random line breaks)
    cleaned = re.sub(r"\n+", " ", cleaned)
    
    # Step 3: Remove unusual special characters that might confuse the model
    # Keep: letters, numbers, basic punctuation (.,!?;:'-"), and spaces
    cleaned = re.sub(r"[^\w\s.,!?;:'\"\-()]", "", cleaned)
    
    # Step 4: Fix spacing around punctuation
    # Remove space before punctuation marks
    cleaned = re.sub(r"\s+([.,!?;:])", r"\1", cleaned)
    
    # Step 5: Strip leading and trailing whitespace
    cleaned = cleaned.strip()
    
    # Step 6: Final cleanup — collapse any remaining multiple spaces
    cleaned = re.sub(r" +", " ", cleaned)
    
    print(f"Text cleaning complete. Length: {len(cleaned)} characters.")
    
    return cleaned


# ============================================================
# STANDALONE TESTING
# ============================================================
# Run this file directly to test cleaning:
#   py -3.12 src/clean_bart.py

if __name__ == "__main__":
    import os
    
    # Load the sample transcript
    transcript_path = os.path.join("data", "transcripts", "sample_meeting.txt")
    
    print("=" * 60)
    print("BART TEXT CLEANING TEST")
    print("=" * 60)
    
    try:
        with open(transcript_path, "r", encoding="utf-8") as file:
            text = file.read()
        
        print(f"\nOriginal text length: {len(text)} characters")
        
        # Clean the text
        cleaned = clean_text_for_bart(text)
        
        print(f"\n--- CLEANED TEXT (first 500 characters) ---")
        print(cleaned[:500])
        print("...")
        
        # Verify sentence structure is preserved
        sentence_count = cleaned.count(".") + cleaned.count("!") + cleaned.count("?")
        print(f"\nApproximate sentence endings preserved: {sentence_count}")
    
    except FileNotFoundError:
        print(f"ERROR: Transcript file not found at '{transcript_path}'")
    except Exception as error:
        print(f"ERROR: {error}")
