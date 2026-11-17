"""
bart_summarizer.py - Abstractive Summarization using DistilBART

This module generates a summary by using a pretrained BART model to write
a NEW summary in its own words (unlike TF-IDF which picks existing sentences).

Model: sshleifer/distilbart-cnn-12-6
- A distilled (compressed) version of facebook/bart-large-cnn
- About 2-3x faster and uses less memory (~600 MB vs ~1.6 GB)
- Same API and very similar quality
- Trained on CNN/DailyMail news articles for summarization

Handling BART's Token Limit:
- BART can process a maximum of 1024 input tokens (~700-800 words).
- If the transcript is longer, we split it into smaller chunks.
- Each chunk is summarized separately.
- The chunk summaries are then combined into the final summary.
"""

from transformers import BartTokenizer, BartForConditionalGeneration
import torch
from src.clean_bart import clean_text_for_bart


# ============================================================
# CONFIGURATION
# ============================================================
# Model to use for summarization.
# "sshleifer/distilbart-cnn-12-6" is fast and lightweight.
# You can change to "facebook/bart-large-cnn" for better quality (but slower).
BART_MODEL_NAME = "sshleifer/distilbart-cnn-12-6"

# Maximum tokens BART can accept as input
MAX_INPUT_TOKENS = 1024

# Summary length controls (in tokens, not words)
# These control how long the generated summary will be.
MIN_SUMMARY_LENGTH = 50
MAX_SUMMARY_LENGTH = 150

# How many tokens to overlap between chunks when splitting long text.
# Overlap helps preserve context between chunks.
CHUNK_OVERLAP_TOKENS = 50


def load_bart_model(model_name=BART_MODEL_NAME):
    """
    Load the BART model and tokenizer.
    
    The first time you run this, the model will be downloaded (~1.2 GB).
    After that, it uses the cached version.
    
    Parameters:
        model_name (str): Name of the Hugging Face model to load
    
    Returns:
        tuple: (model, tokenizer)
    """
    print(f"Loading BART model: {model_name}...")
    
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    
    # Set model to evaluation mode (not training)
    model.eval()
    
    print(f"BART model '{model_name}' loaded successfully.")
    return model, tokenizer


def split_text_into_chunks(text, tokenizer, max_tokens=MAX_INPUT_TOKENS, overlap=CHUNK_OVERLAP_TOKENS):
    """
    Split long text into chunks that fit within BART's token limit.
    
    If the text already fits, it returns a single chunk (the full text).
    
    Parameters:
        text (str): The cleaned transcript text
        tokenizer: The BART tokenizer
        max_tokens (int): Maximum tokens per chunk
        overlap (int): Number of overlapping tokens between chunks
    
    Returns:
        list of str: Text chunks, each within the token limit
    """
    # Tokenize the entire text to see how many tokens it has
    all_tokens = tokenizer.encode(text, add_special_tokens=False)
    total_tokens = len(all_tokens)
    
    print(f"Total tokens in text: {total_tokens}")
    
    # If text fits within the limit, return it as a single chunk
    if total_tokens <= max_tokens:
        print("Text fits within token limit. No chunking needed.")
        return [text]
    
    # Split tokens into overlapping chunks
    print(f"Text exceeds token limit. Splitting into chunks...")
    
    chunks = []
    step_size = max_tokens - overlap  # How far to move forward for each chunk
    start = 0
    
    while start < total_tokens:
        end = min(start + max_tokens, total_tokens)
        
        # Get the tokens for this chunk
        chunk_tokens = all_tokens[start:end]
        
        # Decode tokens back into text
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)
        
        # Move forward by step_size (not max_tokens, to create overlap)
        start += step_size
    
    print(f"Text split into {len(chunks)} chunks.")
    return chunks


def summarize_single_chunk(text, model, tokenizer, min_length=MIN_SUMMARY_LENGTH, max_length=MAX_SUMMARY_LENGTH):
    """
    Generate a summary for a single chunk of text.
    
    Parameters:
        text (str): Text to summarize (must fit within token limit)
        model: The loaded BART model
        tokenizer: The loaded BART tokenizer
        min_length (int): Minimum length of the summary in tokens
        max_length (int): Maximum length of the summary in tokens
    
    Returns:
        str: The generated summary
    """
    # Tokenize the input text
    inputs = tokenizer(
        text,
        return_tensors="pt",       # Return PyTorch tensors
        max_length=MAX_INPUT_TOKENS,
        truncation=True            # Truncate if still too long
    )
    
    # Generate the summary
    # no_grad() tells PyTorch we don't need gradients (saves memory)
    with torch.no_grad():
        summary_ids = model.generate(
            inputs["input_ids"],
            min_length=min_length,
            max_length=max_length,
            num_beams=4,            # Beam search for better quality
            length_penalty=2.0,     # Encourage longer summaries
            early_stopping=True     # Stop when all beams produce end token
        )
    
    # Decode the generated tokens back into text
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return summary


def generate_bart_summary(text, model=None, tokenizer=None):
    """
    Generate an abstractive summary from the given text using BART.
    
    This is the main function you should call from other modules.
    
    Parameters:
        text (str): The raw transcript text
        model: A preloaded BART model. If None, a new model will be loaded.
        tokenizer: A preloaded BART tokenizer. If None, loaded with the model.
    
    Returns:
        dict: A dictionary containing:
            - "summary" (str): The abstractive summary
            - "key_points" (list of str): Key points extracted from the summary
    """
    # --- Step 1: Clean the text ---
    cleaned_text = clean_text_for_bart(text)
    
    # --- Step 2: Load model if not provided ---
    if model is None or tokenizer is None:
        model, tokenizer = load_bart_model()
    
    # --- Step 3: Split into chunks if needed ---
    chunks = split_text_into_chunks(cleaned_text, tokenizer)
    
    # --- Step 4: Summarize each chunk ---
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        print(f"Summarizing chunk {i + 1} of {len(chunks)}...")
        chunk_summary = summarize_single_chunk(chunk, model, tokenizer)
        chunk_summaries.append(chunk_summary)
    
    # --- Step 5: Combine chunk summaries ---
    if len(chunk_summaries) == 1:
        # Only one chunk — use its summary directly
        summary = chunk_summaries[0]
    else:
        # Multiple chunks — combine their summaries
        combined = " ".join(chunk_summaries)
        
        # If the combined summary is short enough, use it as-is
        # Otherwise, summarize the combined summaries one more time
        combined_tokens = tokenizer.encode(combined, add_special_tokens=False)
        if len(combined_tokens) <= MAX_INPUT_TOKENS:
            print("Re-summarizing combined chunk summaries...")
            summary = summarize_single_chunk(combined, model, tokenizer)
        else:
            # Combined summaries are still too long — just join them
            summary = combined
    
    # --- Step 6: Extract key points from the summary ---
    # Split the summary into sentences to use as key points
    # Simple approach: split on period followed by space
    key_points = [s.strip() for s in summary.split(". ") if s.strip()]
    
    # Add period back if missing
    key_points = [point if point.endswith(".") else point + "." for point in key_points]
    
    print("BART summary generated successfully.")
    
    return {
        "summary": summary,
        "key_points": key_points
    }


# ============================================================
# STANDALONE TESTING
# ============================================================
# Run this file directly to test BART summarization:
#   py -3.12 -m src.bart_summarizer

if __name__ == "__main__":
    import os
    import time
    
    # Load the sample transcript
    transcript_path = os.path.join("data", "transcripts", "sample_meeting.txt")
    
    print("=" * 60)
    print("BART ABSTRACTIVE SUMMARIZATION TEST")
    print("=" * 60)
    
    try:
        with open(transcript_path, "r", encoding="utf-8") as file:
            text = file.read()
        
        # Measure time taken
        start_time = time.time()
        
        # Generate the summary
        result = generate_bart_summary(text)
        
        elapsed_time = time.time() - start_time
        
        # Display the summary
        print("\n" + "-" * 60)
        print("ABSTRACTIVE SUMMARY:")
        print("-" * 60)
        print(result["summary"])
        
        # Display key points
        print("\n" + "-" * 60)
        print("KEY POINTS:")
        print("-" * 60)
        for i, point in enumerate(result["key_points"], 1):
            print(f"  {i}. {point}")
        
        print(f"\nTime taken: {elapsed_time:.1f} seconds")
    
    except FileNotFoundError:
        print(f"ERROR: Transcript file not found at '{transcript_path}'")
    except Exception as error:
        print(f"ERROR: {error}")
