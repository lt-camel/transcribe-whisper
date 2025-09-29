from abc import ABC, abstractmethod
import numpy as np

class AudioPreprocessorBase(ABC):
    @abstractmethod
    def denoise(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        pass

    @abstractmethod
    def enhance(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        pass 