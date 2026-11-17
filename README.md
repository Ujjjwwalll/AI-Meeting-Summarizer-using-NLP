# 🎙️ AI Meeting Summarizer

An end-to-end, local Natural Language Processing (NLP) application that transcribes spoken meeting audio and generates concise, structured summaries and actionable key points using two distinct summarization paradigms: **TF-IDF (Extractive)** and **DistilBART (Abstractive)**.

Built entirely with open-source technologies, this application runs **100% locally on CPU** with **zero paid APIs or cloud dependencies**.

---

## 📌 Table of Contents
- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Summarization Approaches](#-summarization-approaches)
  - [Pipeline A: TF-IDF (Extractive)](#pipeline-a-tf-idf-extractive)
  - [Pipeline B: DistilBART (Abstractive)](#pipeline-b-distilbart-abstractive)
  - [Comparison: Extractive vs. Abstractive](#comparison-extractive-vs-abstractive)
- [Tech Stack](#-tech-stack)
- [Project Directory Structure](#-project-directory-structure)
- [Installation & Setup](#-installation--setup)
- [Running the Application](#-running-the-application)


---

## 🚀 Project Overview

In busy professional environments, long meetings generate hours of discussions that are difficult to review. Traditional manual note-taking is slow and prone to omission.

**AI Meeting Summarizer** solves this problem by providing an interactive Streamlit dashboard where users can:
1. Upload any meeting audio recording (`.mp3`, `.wav`, `.m4a`, etc.) or test with pre-loaded samples.
2. Transcribe the audio into accurate text using **OpenAI Whisper** running locally.
3. Edit or inspect the generated transcript with live word count and reading time metrics.
4. Select between **Extractive (TF-IDF)** and **Abstractive (DistilBART)** summarization techniques.
5. Extract key action points and export the summary as a `.txt` file with one click.

---

## ✨ Key Features

- **Local Audio Transcription**: Transcribe meeting audio offline via OpenAI Whisper without cloud API costs or latency.
- **Dual Summarization Techniques**:
  - **TF-IDF Extractive**: Preserves 100% factual accuracy by extracting the most salient verbatim sentences.
  - **DistilBART Abstractive**: Synthesizes ideas into fresh, natural phrasing using a deep Transformer model.
- **Configurable Controls**: Dynamic sliders to tune the summary length (number of sentences).
- **Executive Key Points**: Automatically extracts bulleted action items and decisions.
- **Polished UI/UX**: Modern card-based Streamlit interface with responsive audio playback, compression metrics, and export functionality.
- **Built-in Sample Meeting**: Pre-loaded 2-minute product standup recording so anyone can test the app immediately without uploading their own audio.

---

## 🏗️ System Architecture

```
                    ┌────────────────────────┐
                    │   Meeting Audio File   │
                    │ (.mp3, .wav, .m4a, etc)│
                    └───────────┬────────────┘
                                │
                                ▼
                    ┌────────────────────────┐
                    │  Local Whisper Engine  │
                    │      (tiny / base)     │
                    └───────────┬────────────┘
                                │
                                ▼
                    ┌────────────────────────┐
                    │     Raw Transcript     │
                    └───────────┬────────────┘
                                │
        ┌───────────────────────┴───────────────────────┐
        ▼                                               ▼
┌───────────────────────────────┐       ┌───────────────────────────────┐
│     PIPELINE A: TF-IDF        │       │    PIPELINE B: DistilBART     │
│         (Extractive)          │       │         (Abstractive)         │
├───────────────────────────────┤       ├───────────────────────────────┤
│ • Sentence Splitting          │       │ • Minimal Text Normalization  │
│ • Lowercase & Tokenization    │       │ • Preserve Syntax & Grammar   │
│ • Stopword Removal            │       │ • Context Chunking (<1024 tok)│
│ • Lemmatization (NLTK)        │       │ • Hugging Face Transformer    │
│ • TF-IDF Matrix Calculation   │       │   (sshleifer/distilbart-cnn)  │
│ • Sentence Salience Scoring   │       │ • Neural Text Generation      │
│ • Original-Order Extraction   │       │   (Beam Search Decoding)      │
└───────────────┬───────────────┘       └───────────────┬───────────────┘
                │                                       │
                └───────────────────┬───────────────────┘
                                    │
                                    ▼
                    ┌────────────────────────┐
                    │    Concise Summary     │
                    │    + Key Action Items  │
                    └───────────┬────────────┘
                                │
                                ▼
                    ┌────────────────────────┐
                    │ Streamlit UI Dashboard │
                    └────────────────────────┘
```

---

## 🧠 Summarization Approaches

### Pipeline A: TF-IDF (Extractive)
Extractive summarization identifies and extracts the most significant sentences directly from the source text.

1. **Preprocessing (`src/preprocess_tfidf.py`)**:
   - Tokenizes transcript into sentences and words using `nltk`.
   - Strips non-alphabetic tokens and common English stopwords.
   - Applies WordNet lemmatization to reduce words to their base forms (e.g., *"migrating"* $\rightarrow$ *"migrate"*).
2. **Scoring (`src/tfidf_summarizer.py`)**:
   - Computes the Term Frequency-Inverse Document Frequency (TF-IDF) matrix using `scikit-learn`.
   - Scores each sentence by summing the TF-IDF weights of its constituent words.
3. **Reordering**:
   - Selects the top $N$ ranked sentences.
   - **Crucial Step**: Re-orders selected sentences according to their original appearance in the meeting, and extracts from the **original unmodified sentences** to guarantee natural grammar and readability.

### Pipeline B: DistilBART (Abstractive)
Abstractive summarization reads the transcript and writes an entirely new summary in its own words.

1. **Cleaning (`src/clean_bart.py`)**:
   - Performs light formatting (collapsing whitespace and removing noisy characters).
   - **Does NOT remove stopwords or lowercase words**, as deep transformer attention heads rely on full syntax and grammatical structure.
2. **Generation (`src/bart_summarizer.py`)**:
   - Uses `sshleifer/distilbart-cnn-12-6` — a distilled transformer model that offers 2-3x faster inference and ~600 MB memory usage compared to standard `bart-large-cnn`.
   - Handles text exceeding 1024 tokens via an overlapping chunking strategy.
   - Decodes generated tokens using Beam Search (`num_beams=4`).

### Comparison: Extractive vs. Abstractive

| Dimension | TF-IDF (Extractive) | DistilBART (Abstractive) |
|---|---|---|
| **Core Method** | Statistical word importance weighting | Deep sequence-to-sequence transformer |
| **Output Type** | Direct sentences from the meeting | Freshly rephrased synthesis |
| **Factual Hallucination** | **0% (Zero)** | Low to moderate |
| **Speed (CPU)** | **< 10 milliseconds** | ~5-15 seconds |
| **Memory Needed** | < 10 MB | ~600 MB |
| **Best Application** | Legal/medical transcripts where exact quotes matter | Executive digests where brevity is prioritized |

---

## 🛠️ Tech Stack

- **Language**: Python 3.12+
- **Web Dashboard**: Streamlit
- **Speech-to-Text**: OpenAI Whisper (Local open-source engine)
- **Audio Processing**: FFmpeg
- **Traditional NLP**: NLTK (Tokenization, Stopwords, WordNet Lemmatizer)
- **Statistical Modeling**: Scikit-learn (`TfidfVectorizer`)
- **Deep Learning / Transformers**: Hugging Face Transformers (`DistilBART`), PyTorch (CPU)
- **Data Manipulation**: Pandas, NumPy

---

## 📂 Project Directory Structure

```
meeting-summarizer/
│
├── app.py                            # Streamlit web dashboard
│
├── src/
│   ├── transcribe.py                 # Whisper audio transcription & FFmpeg discovery
│   ├── clean_bart.py                 # Text normalization for Transformer input
│   ├── preprocess_tfidf.py           # Traditional NLP preprocessing for TF-IDF
│   ├── tfidf_summarizer.py           # TF-IDF calculation, sentence scoring & ranking
│   └── bart_summarizer.py            # DistilBART model loader, chunking & generation
│
├── data/
│   ├── audio/                        # Sample & uploaded audio files
│   │   └── sample_meeting.mp3        # Pre-loaded 2-min product meeting recording
│   └── transcripts/                  # Meeting transcripts
│       └── sample_meeting.txt        # Benchmark meeting transcript
│
├── .gitignore
│  
│
├── requirements.txt                  # Python dependencies
└── README.md                         # Documentation
```

---

## 💻 Installation & Setup

### 1. Prerequisites
- **Python 3.10+** (Tested on Python 3.12.1)
- **FFmpeg**: Required by Whisper for audio decoding.

#### Installing FFmpeg on Windows:
Using Windows Package Manager (cmd/PowerShell):
```powershell
winget install --id Gyan.FFmpeg -e
```
*(Note: `src/transcribe.py` contains automatic dynamic discovery that finds FFmpeg automatically on Windows without manual PATH adjustments!)*

### 2. Install Python Dependencies
From the project root:
```powershell
py -3.12 -m pip install -r requirements.txt
```

---

## 🏃 Running the Application

Launch the Streamlit web dashboard:
```powershell
py -3.12 -m streamlit run app.py
```

Open your browser and navigate to:
```
http://localhost:8501
```

### Quick Test Walkthrough:
1. Under **1. Meeting Audio Source**, select **"🎙️ Upload audio"**.
2. Click **"🚀 Transcribe Audio to Text"**. Whisper will transcribe the recording in seconds.
3. Under **3. Summarization Strategy**, choose **TF-IDF (Extractive)** or **BART (Abstractive)**.
4. Click **"✨ Generate Summary & Key Points"**.
5. View the summary card, compression percentage, and bulleted key points.
6. Click **"📥 Download Summary & Action Points (.txt)"** to export your results!

---
