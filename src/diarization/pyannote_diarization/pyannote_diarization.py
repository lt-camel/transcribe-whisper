from typing import List

import numpy as np
import torch
from pyannote.audio import Inference
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

from src.diarization.diarization_base import DiarizationBase
from src.utils.speech_segment_utils import get_segment_audio, duration_time, frame_time
from src.vad.vad_base import SpeechSegment


class PyannoteDiarization(DiarizationBase):
    def __init__(self):
        self.inference = None
        self.window = "whole"

    def load_model(self, model_path: str, window: str = "whole") -> None:
        """
        加载说话人嵌入模型。
        :param model_path: 模型文件路径
        :param window: 嵌入提取窗口类型（whole/sliding）
        """
        self.inference = Inference(model=model_path, window=window)
        self.window = window

    def extract_embeddings(self, audio: np.ndarray, sr: int, segments: List[SpeechSegment],
                           frame_hop: int = 160) -> np.ndarray:
        """
        对每个帧区间提取说话人嵌入特征。
        :param audio: 原始音频
        :param sr: 采样率
        :param segments: VAD输出的帧区间
        :param frame_hop: 帧移
        :return: shape=(num_segments, embedding_dim) 的嵌入特征矩阵
        """
        embeddings = []
        for seg in segments:
            seg_audio = get_segment_audio(audio, seg, frame_hop)
            waveform = torch.tensor(seg_audio).unsqueeze(0)
            embedding = self.inference({"waveform": waveform, "sample_rate": sr})
            embeddings.append(embedding)
        return np.stack(embeddings)

    def cluster_embeddings(self, embeddings: np.ndarray, min_clusters: int = 1, max_clusters: int = 5) -> np.ndarray:
        """
        对嵌入特征进行聚类，自动选择最佳聚类数。
        :param embeddings: 嵌入特征矩阵
        :param min_clusters: 最小聚类数
        :param max_clusters: 最大聚类数
        :return: 每个段的说话人标签数组
        """
        pass

    def diarize(self, audio: np.ndarray, sr: int, segments: List[SpeechSegment], num_speakers: int = 0,
                min_duration: float = 0.8, frame_hop: int = 160) -> List[int]:
        """
        只做说话人识别，返回segments的说话人标识集合（顺序与输入segments一致）。
        :param audio: 原始音频数据
        :param sr: 采样率
        :param segments: VAD输出的帧区间
        :param num_speakers: 说话人数（可选，若为0则自动聚类）
        :param min_duration: 长段最小时长（秒），短段将单独处理
        :param frame_hop: 帧移（采样点数）
        :return: 每段的说话人标签（List[int]，顺序与segments一致）
        """
        if self.inference is None:
            raise RuntimeError("未初始化模型")
        # 嵌入提取
        embeddings = self.extract_embeddings(audio, sr, segments, frame_hop)
        # 长短段分离
        durations = [duration_time(seg, sr, frame_hop) for seg in segments]
        filtered_idx = [i for i, d in enumerate(durations) if d >= min_duration]
        min_idx = [i for i, d in enumerate(durations) if d < min_duration]
        filtered_embeddings = embeddings[filtered_idx] if filtered_idx else np.empty((0, embeddings.shape[1]))
        min_embeddings = embeddings[min_idx] if min_idx else np.empty((0, embeddings.shape[1]))
        speaker_labels = [0] * len(segments)
        if len(filtered_embeddings) > 0:
            X = filtered_embeddings
            if num_speakers <= 0:
                num_speakers = 2
            clustering = AgglomerativeClustering(n_clusters=num_speakers)
            labels = clustering.fit_predict(X)
            for i, idx in enumerate(filtered_idx):
                speaker_labels[idx] = labels[i]
        else:
            labels = []

        for i, label in zip(filtered_idx, labels):
            seg = segments[i]
            start, end = frame_time(seg, sr, frame_hop)
            duration = duration_time(seg, sr, frame_hop)
            print(f"Segment {1}_{seg.index:>3} [{start:>6.3f}, {end:>6.3f}] [{duration:>6.3f}]: speaker={label:>2}")

        # 短段与聚类中心相似度归类
        if len(min_embeddings) > 0 and len(filtered_embeddings) > 0:
            min_labels = self.assign_short_segments_by_center(min_embeddings, X, labels)
            # min_labels = self.assign_short_segments_by_knn(min_embeddings, filtered_embeddings, labels)

            for i, min_label in enumerate(min_labels):
                speaker_labels[min_idx[i]] = min_label
                print(f"短段 {min_idx[i]} 归为说话人 {min_label}")

        return [int(label) for label in speaker_labels]

    def assign_short_segments_by_center(self, min_embeddings, X, long_labels):
        """
        原始中心法：短段与聚类中心余弦相似度最大者归类。
        :return: List[int]
        """
        assigned_labels = []
        centers = []
        for label in set(long_labels):
            centers.append(X[long_labels == label].mean(axis=0))
        centers = np.stack(centers)

        for emb_short in min_embeddings:
            sims = cosine_similarity([emb_short], centers)[0]
            best_label = np.argmax(sims)
            assigned_labels.append(best_label)
        return assigned_labels

    def assign_short_segments_by_knn(self, min_embeddings, long_embeddings, long_labels, k=3, sim_threshold=0.8):
        """
        K近邻投票法: 短段与所有长段embedding计算相似度，选取相似度高于阈值的样本进行投票。
        :param min_embeddings: 短语音段的嵌入特征
        :param long_embeddings: 长语音段的嵌入特征
        :param long_labels: 长语音段的说话人标签
        :param k: 至少考虑的近邻数量
        :param sim_threshold: 相似度阈值，只考虑相似度高于此阈值的样本
        :return: List[int] 分配的标签列表
        """
        print("使用[K近邻投票法]进行短音频归类")
        assigned_labels = []
        for emb in min_embeddings:
            print("===" * 20)
            sims = cosine_similarity([emb], long_embeddings)[0]

            # 获取相似度高于阈值的样本索引
            high_sim_idx = np.where(sims >= sim_threshold)[0]

            # 如果高相似度样本数量少于k，则取top-k
            if len(high_sim_idx) < k:
                high_sim_idx = np.argsort(sims)[-k:]

            selected_labels = [long_labels[i] for i in high_sim_idx]

            # 统计每个标签的出现次数
            label_counts = {}
            for label in selected_labels:
                label_counts[label] = label_counts.get(label, 0) + 1

            # 选择出现次数最多的标签
            best_label = max(label_counts.items(), key=lambda x: x[1])[0]

            assigned_labels.append(best_label)
        return assigned_labels
