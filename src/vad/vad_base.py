from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from typing import List, Optional

@dataclass
class SpeechSegment:
    start_frame: int            # 起始帧下标
    end_frame: int              # 结束帧下标
    index: Optional[int] = None # 段索引，允许为空

class VADBase(ABC):
    @abstractmethod
    def process(self, audio: np.ndarray, sr: int = 16000) -> List[SpeechSegment]:
        """
        对音频进行帧级语音活动检测，返回每段的帧索引信息。
        """
        pass 