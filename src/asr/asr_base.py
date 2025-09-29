from abc import ABC, abstractmethod
import numpy as np
from typing import List

class ASRBase(ABC):
    @abstractmethod
    def transcribe(self, audio_list: List[np.ndarray], sr: int = 16000) -> List[str]:
        pass 