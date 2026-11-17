"""
tfidf_summarizer.py - Extractive Summarization using TF-IDF

This module generates a summary by selecting the most important sentences
from the original transcript using TF-IDF scores.

How it works:
1. Take the preprocessed sentences and compute a TF-IDF matrix.
2. Score each sentence by summing its TF-IDF word scores.
3. Rank sentences from highest to lowest score.
4. Select the top N sentences (configurable).
5. Re-order selected sentences in their original appearance order.
6. Return the final summary using the ORIGINAL (unprocessed) sentences.

What is TF-IDF?
- TF (Term Frequency): How often a word appears in a sentence.
- IDF (Inverse Document Frequency): How rare a word is across all sentences.
- TF-IDF = TF x IDF: Words that are frequent in one sentence but rare overall
  get high scores. This helps identify the most distinctive/important sentences.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from src.preprocess_tfidf import preprocess_transcript


# ============================================================
# CONFIGURATION
# ============================================================
# Number of sentences to include in the summary.
# You can adjust this based on how long you want the summary to be.
NUM_SUMMARY_SENTENCES = 5

# Number of key points to extract.
# Key points are the top-ranked sentences before re-ordering.
NUM_KEY_POINTS = 5


def calculate_tfidf_matrix(preprocessed_sentences):
    """
    Calculate the TF-IDF matrix for a list of preprocessed sentences.
    
    Each row in the matrix represents a sentence.
    Each column represents a unique word.
    Each cell value is the TF-IDF score of that word in that sentence.
    
    Parameters:
        preprocessed_sentences (list of str): Cleaned sentences from preprocessing
    
    Returns:
        tfidf_matrix: The TF-IDF matrix (sparse matrix from scikit-learn)
    """
    # Create the TF-IDF vectorizer
    # It converts a collection of text documents into a matrix of TF-IDF features
    vectorizer = TfidfVectorizer()
    
    # Fit and transform the preprocessed sentences into a TF-IDF matrix
    tfidf_matrix = vectorizer.fit_transform(preprocessed_sentences)
    
    return tfidf_matrix


def score_sentences(tfidf_matrix):
    """
    Score each sentence based on its TF-IDF values.
    
    The score of a sentence = sum of all TF-IDF values of its words.
    A higher score means the sentence contains more important/distinctive words.
    
    Parameters:
        tfidf_matrix: The TF-IDF matrix from calculate_tfidf_matrix()
    
    Returns:
        list of float: A score for each sentence (same order as input)
    """
    # Convert the sparse matrix to a dense array for easier manipulation
    dense_matrix = tfidf_matrix.toarray()
    
    # Sum the TF-IDF scores across all words for each sentence
    # axis=1 means sum across columns (words) for each row (sentence)
    sentence_scores = np.sum(dense_matrix, axis=1).tolist()
    
    return sentence_scores


def rank_sentences(sentence_scores):
    """
    Rank sentence indices from highest to lowest score.
    
    Parameters:
        sentence_scores (list of float): Score for each sentence
    
    Returns:
        list of int: Sentence indices sorted by score (highest first)
    
    Example:
        Input:  [0.5, 1.2, 0.8]
        Output: [1, 2, 0]  (sentence 1 scored highest, then sentence 2, then 0)
    """
    # Create a list of (index, score) pairs
    indexed_scores = list(enumerate(sentence_scores))
    
    # Sort by score in descending order (highest score first)
    sorted_scores = sorted(indexed_scores, key=lambda pair: pair[1], reverse=True)
    
    # Extract just the indices
    ranked_indices = [index for index, score in sorted_scores]
    
    return ranked_indices


def generate_tfidf_summary(text, num_sentences=NUM_SUMMARY_SENTENCES, num_key_points=NUM_KEY_POINTS):
    """
    Generate an extractive summary from the given text using TF-IDF.
    
    This is the main function you should call from other modules.
    
    Parameters:
        text (str): The raw transcript text
        num_sentences (int): How many sentences to include in the summary
        num_key_points (int): How many key points to extract
    
    Returns:
        dict: A dictionary containing:
            - "summary" (str): The extractive summary
            - "key_points" (list of str): The most important sentences
            - "sentence_scores" (list of tuples): All sentences with their scores
    """
    # --- Step 1: Preprocess the transcript ---
    # This gives us original sentences AND preprocessed versions
    original_sentences, preprocessed_sentences = preprocess_transcript(text)
    
    # Handle edge case: not enough sentences
    if len(original_sentences) <= num_sentences:
        print("Text is too short for summarization. Returning the full text.")
        return {
            "summary": text,
            "key_points": original_sentences,
            "sentence_scores": [(s, 1.0) for s in original_sentences]
        }
    
    # Filter out sentences that became empty after preprocessing
    # (This can happen if a sentence was only stopwords/punctuation)
    valid_indices = [i for i, s in enumerate(preprocessed_sentences) if s.strip()]
    
    if len(valid_indices) < num_sentences:
        print("Not enough meaningful sentences after preprocessing.")
        return {
            "summary": text,
            "key_points": original_sentences[:num_key_points],
            "sentence_scores": [(s, 1.0) for s in original_sentences]
        }
    
    # Keep only valid sentences for TF-IDF calculation
    valid_preprocessed = [preprocessed_sentences[i] for i in valid_indices]
    valid_original = [original_sentences[i] for i in valid_indices]
    
    # --- Step 2: Calculate TF-IDF matrix ---
    tfidf_matrix = calculate_tfidf_matrix(valid_preprocessed)
    
    # --- Step 3: Score each sentence ---
    sentence_scores = score_sentences(tfidf_matrix)
    
    # --- Step 4: Rank sentences by importance ---
    ranked_indices = rank_sentences(sentence_scores)
    
    # --- Step 5: Select the top N sentences ---
    top_indices = ranked_indices[:num_sentences]
    
    # --- Step 6: Extract key points (top sentences in ranked order) ---
    # Key points are shown in importance order (most important first)
    key_point_indices = ranked_indices[:num_key_points]
    key_points = [valid_original[i] for i in key_point_indices]
    
    # --- Step 7: Re-order selected sentences in original appearance order ---
    # This makes the summary read naturally, following the meeting flow
    top_indices_sorted = sorted(top_indices)
    summary_sentences = [valid_original[i] for i in top_indices_sorted]
    
    # --- Step 8: Join selected sentences into the final summary ---
    summary = " ".join(summary_sentences)
    
    # Prepare all sentence scores for display (optional)
    all_scores = [(valid_original[i], sentence_scores[i]) for i in range(len(valid_original))]
    all_scores.sort(key=lambda pair: pair[1], reverse=True)
    
    print(f"TF-IDF summary generated: {num_sentences} sentences selected from {len(valid_original)}.")
    
    return {
        "summary": summary,
        "key_points": key_points,
        "sentence_scores": all_scores
    }


# ============================================================
# STANDALONE TESTING
# ============================================================
# Run this file directly to test TF-IDF summarization:
#   py -3.12 src/tfidf_summarizer.py

if __name__ == "__main__":
    import os
    
    # Load the sample transcript
    transcript_path = os.path.join("data", "transcripts", "sample_meeting.txt")
    
    print("=" * 60)
    print("TF-IDF EXTRACTIVE SUMMARIZATION TEST")
    print("=" * 60)
    
    try:
        with open(transcript_path, "r", encoding="utf-8") as file:
            text = file.read()
        
        # Generate the summary
        result = generate_tfidf_summary(text)
        
        # Display the summary
        print("\n" + "-" * 60)
        print("EXTRACTIVE SUMMARY:")
        print("-" * 60)
        print(result["summary"])
        
        # Display key points
        print("\n" + "-" * 60)
        print("KEY POINTS:")
        print("-" * 60)
        for i, point in enumerate(result["key_points"], 1):
            print(f"  {i}. {point}")
        
        # Display top 5 sentence scores
        print("\n" + "-" * 60)
        print("TOP 5 SENTENCE SCORES:")
        print("-" * 60)
        for sentence, score in result["sentence_scores"][:5]:
            # Truncate long sentences for display
            display_text = sentence[:80] + "..." if len(sentence) > 80 else sentence
            print(f"  Score: {score:.4f} | {display_text}")
    
    except FileNotFoundError:
        print(f"ERROR: Transcript file not found at '{transcript_path}'")
    except Exception as error:
        print(f"ERROR: {error}")
