"""
preprocess_tfidf.py - Traditional NLP Preprocessing for TF-IDF Summarization

This module prepares the raw transcript for TF-IDF based extractive summarization.

What it does:
1. Splits the transcript into individual sentences.
2. Converts each sentence to lowercase.
3. Tokenizes sentences into words.
4. Removes stopwords (common words like "the", "is", "and").
5. Optionally applies lemmatization (converting words to their base form).

IMPORTANT:
- This preprocessing is ONLY used for calculating sentence importance scores.
- The final summary will use the ORIGINAL sentences (not the preprocessed ones).
- This ensures the summary remains readable and grammatically correct.
"""

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data files (only needed on first run)
# These are small data files, not models — they download in seconds.
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)


# ============================================================
# CONFIGURATION
# ============================================================
# Set to True if you want to apply lemmatization.
# Lemmatization converts words to their base form:
#   "running" -> "run", "completed" -> "completed", "issues" -> "issue"
# This can improve TF-IDF accuracy by grouping related words together.
USE_LEMMATIZATION = True


def split_into_sentences(text):
    """
    Split the transcript into a list of individual sentences.
    
    Parameters:
        text (str): The raw transcript text
    
    Returns:
        list: A list of sentences (original, unmodified strings)
    
    Example:
        Input:  "Hello everyone. Welcome to the meeting."
        Output: ["Hello everyone.", "Welcome to the meeting."]
    """
    sentences = sent_tokenize(text)
    
    # Remove any empty sentences or sentences that are just whitespace
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    
    return sentences


def preprocess_sentence(sentence, use_lemmatization=USE_LEMMATIZATION):
    """
    Preprocess a single sentence for TF-IDF calculation.
    
    Steps:
    1. Convert to lowercase
    2. Tokenize into individual words
    3. Keep only alphabetic words (removes numbers and punctuation)
    4. Remove stopwords
    5. Optionally apply lemmatization
    
    Parameters:
        sentence (str): A single sentence to preprocess
        use_lemmatization (bool): Whether to apply lemmatization
    
    Returns:
        str: The preprocessed sentence as a single string
    
    Example:
        Input:  "The engineering team has successfully migrated the system."
        Output: "engineering team successfully migrated system"
    """
    # Step 1: Convert to lowercase
    sentence_lower = sentence.lower()
    
    # Step 2: Tokenize into individual words
    words = word_tokenize(sentence_lower)
    
    # Step 3: Keep only alphabetic words (removes punctuation and numbers)
    words = [word for word in words if word.isalpha()]
    
    # Step 4: Remove stopwords
    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word not in stop_words]
    
    # Step 5: Optionally apply lemmatization
    if use_lemmatization:
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]
    
    # Join the words back into a single string
    preprocessed_sentence = " ".join(words)
    
    return preprocessed_sentence


def preprocess_transcript(text, use_lemmatization=USE_LEMMATIZATION):
    """
    Preprocess the entire transcript for TF-IDF summarization.
    
    This is the main function you should call from other modules.
    
    It returns TWO lists:
    1. original_sentences  - The original, unmodified sentences (for the final summary)
    2. preprocessed_sentences - The cleaned sentences (for TF-IDF scoring)
    
    Both lists have the same length and are aligned by index:
    - original_sentences[0] corresponds to preprocessed_sentences[0]
    - original_sentences[1] corresponds to preprocessed_sentences[1]
    - ... and so on
    
    Parameters:
        text (str): The raw transcript text
        use_lemmatization (bool): Whether to apply lemmatization
    
    Returns:
        tuple: (original_sentences, preprocessed_sentences)
            - original_sentences (list of str): Unmodified sentences
            - preprocessed_sentences (list of str): Cleaned sentences for TF-IDF
    """
    # Step 1: Split the transcript into individual sentences
    original_sentences = split_into_sentences(text)
    
    # Step 2: Preprocess each sentence
    preprocessed_sentences = []
    for sentence in original_sentences:
        preprocessed = preprocess_sentence(sentence, use_lemmatization)
        preprocessed_sentences.append(preprocessed)
    
    print(f"Preprocessing complete: {len(original_sentences)} sentences found.")
    
    return original_sentences, preprocessed_sentences


# ============================================================
# STANDALONE TESTING
# ============================================================
# Run this file directly to test preprocessing:
#   py -3.12 src/preprocess_tfidf.py

if __name__ == "__main__":
    import os
    
    # Load the sample transcript
    transcript_path = os.path.join("data", "transcripts", "sample_meeting.txt")
    
    print("=" * 60)
    print("TF-IDF PREPROCESSING TEST")
    print("=" * 60)
    
    try:
        with open(transcript_path, "r", encoding="utf-8") as file:
            text = file.read()
        
        # Run preprocessing
        original_sentences, preprocessed_sentences = preprocess_transcript(text)
        
        # Display the first 5 sentences: original vs preprocessed
        print(f"\nShowing first 5 of {len(original_sentences)} sentences:\n")
        
        for i in range(min(5, len(original_sentences))):
            print(f"--- Sentence {i + 1} ---")
            print(f"  ORIGINAL:      {original_sentences[i]}")
            print(f"  PREPROCESSED:  {preprocessed_sentences[i]}")
            print()
    
    except FileNotFoundError:
        print(f"ERROR: Transcript file not found at '{transcript_path}'")
    except Exception as error:
        print(f"ERROR: {error}")
