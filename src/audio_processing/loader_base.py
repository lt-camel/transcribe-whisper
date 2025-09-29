from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple

class AudioLoaderBase(ABC):
    @abstractmethod
    def load_from_file(self, file_path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
        pass

    @abstractmethod
    def load_from_array(self, audio_array: np.ndarray, sr: int = 16000) -> Tuple[np.ndarray, int]:
        pass 