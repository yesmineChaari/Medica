# audio_utils.py
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import tempfile
from faster_whisper import WhisperModel
# --- START OF BAND-AID PATCH ---
import torch
import transformers.pytorch_utils
# Recreate the missing function that the new transformers deleted
def isin_mps_friendly(elements, test_elements):
    return torch.isin(elements, test_elements)
# Inject it directly into transformers before TTS loads
transformers.pytorch_utils.isin_mps_friendly = isin_mps_friendly
# --- END OF BAND-AID PATCH ---

from TTS.api import TTS

_whisper_model = WhisperModel(model_size_or_path="small", device="cpu", compute_type="int8")
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
    segments, info = _whisper_model.transcribe(audio_path, beam_size=2)
    
    transcription = " ".join([segment.text for segment in segments])
    return transcription, info.language


def synthesize_speech(text: str, tts, speaker: str, output_path=None) -> str:
    import os
    if output_path is None:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        output_path = tmp.name
    try:
        tts.tts_to_file(text=text, speaker=speaker, language="en", file_path=output_path)
        return output_path
    except Exception:
        os.unlink(output_path)
        raise