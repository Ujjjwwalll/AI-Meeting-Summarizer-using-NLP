"""
app.py - Streamlit Web Dashboard for AI Meeting Summarizer

A clean, sequential, and beginner-friendly web application:
Step 1: Upload Meeting Audio (Local Whisper Transcription)
Step 2: Review & Edit Transcript
Step 3: Select Summarization Strategy (TF-IDF vs DistilBART)
Step 4: View Summary, Key Points & Export
"""

import os
import streamlit as st

# Import backend NLP modules
from src.transcribe import load_whisper_model, transcribe_audio
from src.tfidf_summarizer import generate_tfidf_summary
from src.bart_summarizer import load_bart_model, generate_bart_summary


# ============================================================
# PAGE CONFIGURATION & STYLING
# ============================================================
st.set_page_config(
    page_title="AI Meeting Summarizer",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished, high-contrast UI
st.markdown("""
<style>
    /* Metric Badges */
    .metric-box {
        background: #FFFFFF;
        border: 1px solid #CBD5E1;
        border-radius: 10px;
        padding: 14px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .metric-value {
        font-size: 24px;
        font-weight: 700;
        color: #1E293B;
    }
    .metric-label {
        font-size: 13px;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
    }

    /* Summary Card */
    .summary-card {
        background-color: #FFFFFF;
        border-left: 6px solid #2563EB;
        padding: 22px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.06);
        line-height: 1.8;
        font-size: 16px;
        color: #0F172A;
        border-top: 1px solid #E2E8F0;
        border-right: 1px solid #E2E8F0;
        border-bottom: 1px solid #E2E8F0;
    }

    /* Key Point Items */
    .key-point-item {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 10px;
        border-left: 5px solid #10B981;
        border: 1px solid #E2E8F0;
        border-left-width: 5px;
        border-left-color: #10B981;
        font-size: 15px;
        color: #0F172A;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# CACHED MODEL LOADERS (Loads once into memory)
# ============================================================
@st.cache_resource(show_spinner=False)
def get_whisper_model(model_size="tiny"):
    """Load Whisper model once and keep in memory."""
    return load_whisper_model(model_size)


@st.cache_resource(show_spinner=False)
def get_bart_model():
    """Load DistilBART model & tokenizer once and keep in memory."""
    return load_bart_model()


# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================
if "transcript" not in st.session_state:
    st.session_state.transcript = ""
if "summary_result" not in st.session_state:
    st.session_state.summary_result = None
if "current_method" not in st.session_state:
    st.session_state.current_method = ""
if "audio_path" not in st.session_state:
    st.session_state.audio_path = None


# ============================================================
# SIDEBAR: MODEL CONFIGURATION & MODEL COMPARISON GUIDE
# ============================================================
with st.sidebar:
    st.title("⚙️ Settings")
    st.markdown("---")
    
    st.subheader("Whisper Audio Model")
    whisper_size = st.selectbox(
        "Select Model Size",
        options=["tiny", "base"],
        index=0
    )
    
    st.markdown("---")
    st.subheader("💡 Model Guide: Tiny vs Base")
    st.markdown("""
    **`tiny` (Recommended)**
    - **Speed**: Ultra-fast (~5-10 sec)
    - **RAM**: ~1 GB
    - **When to use**: Clear audio, quick meeting summaries, standard PC hardware.

    **`base`**
    - **Speed**: Moderate (~20-30 sec)
    - **RAM**: ~1 GB
    - **When to use**: Audio with background noise, strong accents, or complex technical terms.
    """)


# ============================================================
# HEADER BANNER
# ============================================================
st.title("🎙️ AI Meeting Summarizer")
st.caption("Upload your meeting audio to generate an accurate transcript, concise executive summary, and actionable key points.")
st.markdown("---")


# ============================================================
# STEP 1: UPLOAD MEETING AUDIO (FULL SCREEN WIDTH)
# ============================================================
st.subheader("📁 Step 1: Upload Meeting Audio", divider="blue")

uploaded_file = st.file_uploader(
    "Choose a meeting audio recording (.mp3, .wav, .m4a, .ogg, .flac)",
    type=["mp3", "wav", "m4a", "ogg", "flac"],
    help="Upload your recorded meeting audio."
)

if uploaded_file is not None:
    # Save uploaded file locally
    save_dir = os.path.abspath(os.path.join("data", "audio"))
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.session_state.audio_path = save_path

# Display audio player and Transcribe button
if st.session_state.audio_path and os.path.exists(st.session_state.audio_path):
    st.audio(st.session_state.audio_path)
    
    if st.button("🚀 Transcribe Audio to Text", type="primary", use_container_width=True):
        with st.spinner("Transcribing meeting audio with Whisper... please wait"):
            try:
                whisper_model = get_whisper_model(whisper_size)
                transcript = transcribe_audio(st.session_state.audio_path, model=whisper_model)
                st.session_state.transcript = transcript
                st.session_state.summary_result = None
                st.success("Transcription completed successfully!")
            except Exception as e:
                st.error(f"Transcription error: {e}")


# ============================================================
# STEP 2: MEETING TRANSCRIPT (FULL SCREEN WIDTH)
# ============================================================
st.subheader("📝 Step 2: Meeting Transcript", divider="blue")

transcript_input = st.text_area(
    "Review, edit, or directly paste meeting transcript below:",
    value=st.session_state.transcript,
    height=220,
    placeholder="Your transcribed text will appear here automatically after transcription. You can also paste an existing transcript directly..."
)
st.session_state.transcript = transcript_input

# Transcript Stats Bar
if st.session_state.transcript.strip():
    words = len(st.session_state.transcript.split())
    chars = len(st.session_state.transcript)
    reading_time = max(1, round(words / 150))
    
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f"<div class='metric-box'><div class='metric-value'>{words}</div><div class='metric-label'>Total Words</div></div>", unsafe_allow_html=True)
    with m2:
        st.markdown(f"<div class='metric-box'><div class='metric-value'>{chars}</div><div class='metric-label'>Total Characters</div></div>", unsafe_allow_html=True)
    with m3:
        st.markdown(f"<div class='metric-box'><div class='metric-value'>~{reading_time} min</div><div class='metric-label'>Est. Reading Time</div></div>", unsafe_allow_html=True)


# ============================================================
# STEP 3: SUMMARIZATION STRATEGY (FULL SCREEN WIDTH)
# ============================================================
st.subheader("⚙️ Step 3: Select Summarization Strategy", divider="blue")

col_strat, col_slider = st.columns([1, 1], gap="medium")

with col_strat:
    method = st.radio(
        "Summarization Method:",
        options=["TF-IDF (Extractive)", "BART (Abstractive)"],
        index=0,
        horizontal=True
    )
    if method == "TF-IDF (Extractive)":
        st.caption("📌 **Extractive**: Ranks and selects the most important verbatim sentences directly from the transcript using statistical term frequency. Zero risk of hallucination.")
    else:
        st.caption("📌 **Abstractive**: Uses `sshleifer/distilbart-cnn-12-6` to rewrite and synthesize discussion points into fresh, concise sentences.")

with col_slider:
    if method == "TF-IDF (Extractive)":
        num_sentences = st.slider("Target Summary Length (Sentences)", min_value=2, max_value=8, value=4)
    else:
        st.info("ℹ️ DistilBART dynamically generates a concise abstractive paragraph based on natural meeting flow.")
        num_sentences = 4

# Summarize Action Button
if st.button("✨ Generate Summary & Action Items", type="primary", use_container_width=True):
    if not st.session_state.transcript.strip():
        st.warning("Please upload an audio file, transcribe it, or paste transcript text first.")
    else:
        with st.spinner(f"Generating {method} summary..."):
            try:
                if method == "TF-IDF (Extractive)":
                    result = generate_tfidf_summary(
                        st.session_state.transcript,
                        num_sentences=num_sentences,
                        num_key_points=5
                    )
                else:
                    bart_model, bart_tokenizer = get_bart_model()
                    result = generate_bart_summary(
                        st.session_state.transcript,
                        model=bart_model,
                        tokenizer=bart_tokenizer
                    )

                st.session_state.summary_result = result
                st.session_state.current_method = method
                st.success(f"{method} summary ready!")
            except Exception as e:
                st.error(f"Summarization error: {e}")


# ============================================================
# STEP 4: GENERATED SUMMARY & KEY POINTS (FULL SCREEN WIDTH)
# ============================================================
if st.session_state.summary_result is not None:
    st.subheader("📋 Step 4: Meeting Summary & Action Items", divider="blue")
    
    summary_text = st.session_state.summary_result["summary"]
    st.markdown(f"<div class='summary-card'>{summary_text}</div>", unsafe_allow_html=True)
    
    # Compression Metric
    orig_words = len(st.session_state.transcript.split())
    sum_words = len(summary_text.split())
    reduction = round((1 - (sum_words / max(1, orig_words))) * 100) if orig_words > 0 else 0
    st.caption(f"📊 **Compression Ratio**: {orig_words} words → {sum_words} words (**{reduction}% word reduction**)")

    # Key Action Points
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 🎯 Key Action Points")
    key_points = st.session_state.summary_result.get("key_points", [])
    
    for idx, pt in enumerate(key_points, 1):
        st.markdown(f"<div class='key-point-item'><b>{idx}.</b> {pt}</div>", unsafe_allow_html=True)

    # Export Download Button
    export_content = (
        f"AI MEETING SUMMARY ({st.session_state.current_method})\n"
        f"{'='*50}\n\n"
        f"{summary_text}\n\n"
        f"KEY ACTION ITEMS\n"
        f"{'-'*50}\n"
        + "\n".join([f"{i}. {p}" for i, p in enumerate(key_points, 1)])
    )
    
    st.download_button(
        label="📥 Download Summary & Action Items (.txt)",
        data=export_content,
        file_name="meeting_summary.txt",
        mime="text/plain",
        use_container_width=True
    )
