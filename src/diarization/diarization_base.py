from abc import ABC, abstractmethod
from src.vad.vad_base import SpeechSegment
from typing import List
import numpy as np

class DiarizationBase(ABC):
    @abstractmethod
    def load_model(self, model_path: str, window: str = "whole") -> None:
        """
        加载说话人嵌入模型。
        """
        pass

    @abstractmethod
    def extract_embeddings(self, audio: np.ndarray, sr: int, segments: List[SpeechSegment], frame_hop: int = 160) -> np.ndarray:
        """
        对每个帧区间提取说话人嵌入特征。
        返回 shape=(num_segments, embedding_dim) 的ndarray。
        """
        pass

    @abstractmethod
    def cluster_embeddings(self, embeddings: np.ndarray, min_clusters: int = 1, max_clusters: int = 5) -> np.ndarray:
        """
        对嵌入特征进行聚类，返回每个段的说话人标签。
        """
        pass

    @abstractmethod
    def diarize(self, audio: np.ndarray, sr: int, segments: List[SpeechSegment], num_speakers: int = 0, min_duration: float = 0.8, frame_hop: int = 160) -> List[int]:
        """
        完整说话人分离流程，返回每段的说话人标签和帧区间。
        """
        pass
    