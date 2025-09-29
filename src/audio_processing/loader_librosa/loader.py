import numpy as np
import librosa
from typing import Tuple
from src.audio_processing.loader_base import AudioLoaderBase

class AudioLoaderLibrosa(AudioLoaderBase):
    """
    基于librosa的音频加载实现。
    """
    def load_from_file(self, file_path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
        audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
        return audio, sr

    def load_from_array(self, audio_array: np.ndarray, sr: int = 16000) -> Tuple[np.ndarray, int]:
        return audio_array, sr 