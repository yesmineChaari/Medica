from functools import lru_cache
import logging
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import tempfile
import threading
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_whisper_model():
    return WhisperModel(model_size_or_path="small", device="cpu", compute_type="int8")

def record_audio(duration=5, fs=16000):
    """
    Record audio from microphone for a given duration and sample rate.
    Returns the path to the temporary WAV file.
    """
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()

    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    wav.write(temp_wav.name, fs, audio)
    return temp_wav.name


class AudioRecorder:
    def __init__(self, fs=16000, channels=1):
        self.fs = fs
        self.channels = channels
        self._chunks = []
        self._lock = threading.Lock()
        self._stream = sd.InputStream(
            samplerate=fs,
            channels=channels,
            dtype="int16",
            callback=self._callback,
        )
        self._stream.start()

    def _callback(self, indata, frames, time_info, status):
        if status:
            logger.warning("Audio input status: %s", status)
        with self._lock:
            self._chunks.append(indata.copy())

    def stop(self):
        self._stream.stop()
        self._stream.close()

        with self._lock:
            if not self._chunks:
                raise RuntimeError("No audio was captured.")
            audio = np.concatenate(self._chunks, axis=0)

        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_wav.close()
        wav.write(temp_wav.name, self.fs, audio)
        return temp_wav.name


def start_audio_recording(fs=16000):
    return AudioRecorder(fs=fs)


def stop_audio_recording(recorder):
    if recorder is None:
        raise RuntimeError("Recording was not started.")
    return recorder.stop()

def transcribe_audio(audio_path):
    """
    Transcribe the audio file using faster-whisper and return the text.
    """
    whisper_model = get_whisper_model()
    segments, info = whisper_model.transcribe(audio_path, beam_size=2)
    
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
