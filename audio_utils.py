# audio_utils.py
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import tempfile
from faster_whisper import WhisperModel
from TTS.api import TTS


def record_audio(duration=5, fs=16000):
    """
    Record audio from microphone for a given duration and sample rate.
    Returns the path to the temporary WAV file.
    """
    st = "Recording..."
    print(st)
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    print("Recording complete.")

    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    wav.write(temp_wav.name, fs, audio)
    return temp_wav.name

def transcribe_audio(audio_path):
    """
    Transcribe the audio file using faster-whisper and return the text.
    """
    model = WhisperModel(model_size_or_path="base", device="cpu", compute_type="int8")
    segments, info = model.transcribe(audio_path, beam_size=2)
    
    transcription = " ".join([segment.text for segment in segments])
    return transcription, info.language

tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False, gpu=False)
available_speakers = tts.speakers
default_speaker = available_speakers[0]  # Just pick the first one (or choose based on preference)

def synthesize_speech(text, output_path=None):
    """
    Synthesize speech from text using Coqui TTS and save it to a WAV file.
    Returns the file path.
    """
    print("Synthesizing speech...")

    if output_path is None:
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name

    # Synthesize and save
    tts.tts_to_file(text=text, speaker=default_speaker, language="en", file_path=output_path)

    print(f"Speech saved to {output_path}")
    return output_path
