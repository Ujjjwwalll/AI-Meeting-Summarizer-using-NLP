"""
transcribe.py - Audio Transcription using OpenAI Whisper

This module handles converting audio files into text using the Whisper model.
Whisper runs locally on your computer — no API keys or internet needed (after model download).

How it works:
1. Ensure ffmpeg is accessible on system PATH (required for audio decoding).
2. Load the Whisper model (configurable size: tiny, base, small, etc.)
3. Accept an audio file path
4. Transcribe the audio into text
5. Return the transcript as a string
"""

import os
import shutil
import whisper

# ============================================================
# FFMPEG PATH RESOLUTION (Windows Support)
# ============================================================
# Whisper relies on ffmpeg for audio decoding. On Windows, winget or manual installs
# might not immediately reflect in running process environments.
def ensure_ffmpeg_in_path():
    """Ensure ffmpeg executable is discovered in os.environ['PATH']."""
    if shutil.which("ffmpeg") is not None:
        return

    # Check common Windows directories where ffmpeg gets installed
    candidate_dirs = [
        os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Links"),
        r"C:\ffmpeg\bin",
        r"C:\Program Files\ffmpeg\bin",
    ]

    # Search WinGet packages directory dynamically if installed via winget
    winget_pkg_dir = os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages")
    if os.path.exists(winget_pkg_dir):
        for root, _, files in os.walk(winget_pkg_dir):
            if "ffmpeg.exe" in files:
                candidate_dirs.insert(0, root)
                break

    for folder in candidate_dirs:
        if os.path.exists(folder):
            os.environ["PATH"] = folder + os.pathsep + os.environ.get("PATH", "")
            if shutil.which("ffmpeg") is not None:
                print(f"FFmpeg successfully located at: {folder}")
                return

# Automatically resolve ffmpeg on module import
ensure_ffmpeg_in_path()


# ============================================================
# CONFIGURATION
# ============================================================
WHISPER_MODEL_SIZE = "tiny"


def load_whisper_model(model_size=WHISPER_MODEL_SIZE):
    """
    Load the Whisper model into memory.
    
    Parameters:
        model_size (str): Size of the Whisper model to load.
                          Options: "tiny", "base", "small", "medium", "large"
    
    Returns:
        model: The loaded Whisper model object
    """
    ensure_ffmpeg_in_path()
    print(f"Loading Whisper model: {model_size}...")
    model = whisper.load_model(model_size)
    print(f"Whisper model '{model_size}' loaded successfully.")
    return model


def transcribe_audio(audio_file_path, model=None):
    """
    Transcribe an audio file into text using Whisper.
    
    Parameters:
        audio_file_path (str): Path to the audio file (mp3, wav, m4a, etc.)
        model: A preloaded Whisper model. If None, a new model will be loaded.
    
    Returns:
        str: The transcribed text from the audio file
    """
    # Verify file exists
    if not os.path.exists(audio_file_path):
        raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

    # Check extension
    supported_formats = [".mp3", ".wav", ".m4a", ".flac", ".ogg", ".webm", ".mp4"]
    file_extension = os.path.splitext(audio_file_path)[1].lower()
    
    if file_extension not in supported_formats:
        raise ValueError(
            f"Unsupported audio format: '{file_extension}'. "
            f"Supported formats: {', '.join(supported_formats)}"
        )

    # Ensure ffmpeg is available before running transcription
    ensure_ffmpeg_in_path()
    if shutil.which("ffmpeg") is None:
        raise FileNotFoundError(
            "FFmpeg was not found on your system. Please make sure FFmpeg is installed and accessible."
        )

    if model is None:
        model = load_whisper_model()

    print(f"Transcribing audio file: {os.path.basename(audio_file_path)}...")
    
    try:
        result = model.transcribe(audio_file_path)
        transcript = result["text"].strip()
        print("Transcription completed successfully.")
        return transcript
    
    except Exception as error:
        raise Exception(f"Error during transcription: {str(error)}")


if __name__ == "__main__":
    test_audio_path = os.path.join("data", "audio", "sample_meeting.mp3")
    print("=" * 60)
    print("WHISPER TRANSCRIPTION TEST")
    print("=" * 60)
    
    try:
        model = load_whisper_model()
        transcript = transcribe_audio(test_audio_path, model=model)
        print("\nTRANSCRIPT:\n", transcript)
    except Exception as error:
        print(f"ERROR: {error}")
