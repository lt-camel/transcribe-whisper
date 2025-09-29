import numpy as np
import noisereduce as nr
from src.audio_processing.preprocess_base import AudioPreprocessorBase

class AudioPreprocessorNR(AudioPreprocessorBase):
    def denoise(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        reduced_audio = nr.reduce_noise(y=audio, sr=sr)
        return reduced_audio

    def enhance(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        return audio 