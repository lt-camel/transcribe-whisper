import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.vad.ten_vad.ten_vad import TenVAD
from src.audio_processing.loader_librosa.loader import AudioLoaderLibrosa
from pyannote.audio import Inference
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from src.utils.speech_segment_utils import duration_time, split_audio, get_segment_audio, frame_time

# audio_file = "C:/Users/camel/Downloads/192.168.16.85/202507031500/X5MINI-0WS12_20250702_173636.aac"
# audio_file = "C:/Users/camel/Downloads/192.168.16.85/202507111803/merged.wav"
# audio_file = "C:/Users/camel/Downloads/192.168.16.85/202507111714/merged.wav"

# audio_file = "C:/Users/camel/Downloads/192.168.16.85/202507091804/_1752055133704_1752055153428_0.mp3"
# audio_file = "C:/Users/camel/Downloads/192.168.16.85/202507141626/device1_20250702_095000.aac"
audio_file = "C:/Users/camel/Downloads/192.168.16.85/device1_20250702_142451.aac"
# audio_file = "C:/Users/camel/Downloads/192.168.16.85/202507031501/X5MINI-0WS12_20250702_143038.aac"

# audio_file = "C:/Users/camel/Downloads/192.168.16.85/202507151116/device1_20250702_154231.aac"

embedding_model = "D:/workspace/project/ai-model/transcribe-whisper/models/embedding/pytorch_model.bin"
embedding_window = "whole"
# embedding_window =  "sliding"

print("开始加载 embedding 模型")
# 加载模型
inference = Inference(model=embedding_model, window=embedding_window)
print("模型加载完成")

# 初始化 TenVAD
hop_size = 128
vad = TenVAD(frame_size=hop_size, energy_threshold=0.65, overlap_sec=0.16)

# 加载音频
audioLoader = AudioLoaderLibrosa()
audio, sr = audioLoader.load_from_file(audio_file, 16000)

segments = vad.process(audio, sr)
print("===" * 20)
print(len(segments))
print("===" * 20)
for i, part in enumerate(segments):
    start, end = frame_time(part, sr, hop_size)
    duration = duration_time(part, sr, hop_size)
    print(i)
    print(f"start: {start}")
    print(f"end: {end}")
    print(f"duration: {duration}")
    print("===" * 20)

embeddings = []
for segment in segments:
    array = get_segment_audio(audio, segment, hop_size)
    waveform = torch.tensor(array).unsqueeze(0)
    embedding = inference({"waveform": waveform, "sample_rate": sr})
    embeddings.append(embedding)

# KMeans 聚类
# num_speakers = 2  # 可调整
# kmeans = KMeans(n_clusters=num_speakers, random_state=0)
# labels = kmeans.fit_predict(X)

# 为什么要丢弃部分短时间的音频。因为会扰乱识别效果
# TODO 将短时间的音频，单独做说话人识别，并与长时间的音频合并
min_duration = 0.8  # 秒
filtered_segments = [seg for seg in segments if duration_time(seg, sr, hop_size) >= min_duration]
filtered_embeddings = [embeddings[i] for i, seg in enumerate(segments) if duration_time(seg, sr, hop_size) >= min_duration]
print(f"过滤时长小于 {min_duration}s 的音频段，剩余音频个数: {len(filtered_segments)}")

X = np.stack(filtered_embeddings)

# best_score = -1
# best_k = 1
# for k in range(1, min(len(X), 3) + 1):  # 尝试1~3类
#     clustering = AgglomerativeClustering(n_clusters=k)
#     labels = clustering.fit_predict(X)
#     print(f"labels: {len(labels)}")
#     print(labels)
#     if k == 1:
#         continue  # silhouette_score不支持k=1
#     if len(set(labels)) < 2 or len(set(labels)) >= len(X):
#         continue  # 跳过不合法的聚类数
#     score = silhouette_score(X, labels)
#     print(f"score = {score}")
#     if score > best_score:
#         best_score = score
#         best_k = k

# print(f"最佳聚类数: {best_k}")

# num_speakers = best_k
num_speakers = 2
clustering = AgglomerativeClustering(n_clusters=num_speakers)
labels = clustering.fit_predict(X)
print(f"聚类数: {len(set(labels))}")

for i, segment in enumerate(filtered_segments):
    start, end = frame_time(segment, sr, hop_size)
    duration = duration_time(segment, sr, hop_size)
    print(f"Segment {i}_{segment.index}: start={start}, end={end}, duration={duration}, speaker={labels[i]}")

speaker_labels = [seg.index for seg in segments]
for i, label in enumerate(labels):
    speaker_labels[filtered_segments[i].index] = label

print("===" * 20)
min_segments = [seg for seg in segments if duration_time(seg, sr, hop_size) < min_duration]
min_embeddings = [embeddings[i] for i, seg in enumerate(segments) if duration_time(seg, sr, hop_size) < min_duration]

if min_embeddings and len(min_embeddings) > 0:
    for i, emb in enumerate(min_embeddings):
        if np.isnan(emb).any():
            print(f'min_embedding index: {i}, segments index: {segments[min_segments[i].index]}')
    min_X = np.stack(min_embeddings)
    min_clustering = AgglomerativeClustering(n_clusters=num_speakers)
    min_labels = min_clustering.fit_predict(min_X)
    print(f"聚类数: {len(set(min_labels))}")

    for i, segment in enumerate(min_segments):
        start, end = frame_time(segment, sr, hop_size)
        duration = duration_time(segment, sr, hop_size)
        print(f"Segment {i}_{segment.index}: start={start}, end={end}, duration={duration}, speaker={min_labels[i]}")
    print("===" * 20)

    centers = []
    for label in set(labels):
        centers.append(X[labels == label].mean(axis=0))
    centers = np.stack(centers)

    for i, emb_short in enumerate(min_X):
        sims = cosine_similarity([emb_short], centers)[0]
        best_label = np.argmax(sims)
        speaker_labels[min_segments[i].index] = best_label
        print(f"短段 {min_segments[i].index} 归为说话人 {best_label}")

    for i, segment in enumerate(segments):
        start, end = frame_time(segment, sr, hop_size)
        duration = duration_time(segment, sr, hop_size)
        print(f"Segment {i}_{segment.index}: start={start}, end={end}, duration={duration}, speaker={speaker_labels[i]}")
    print("===" * 20)

# 生成颜色
num_speakers = len(set(labels))
colors = plt.cm.get_cmap('tab10', num_speakers)

plt.figure(figsize=(12, 2))
for i, segment in enumerate(segments):
    plt.barh(
        y=0,
        width=(segment.end_frame - segment.start_frame) * hop_size / sr,
        left=segment.start_frame * hop_size / sr,
        height=0.5,
        color=colors(speaker_labels[i]),
        edgecolor='black',
        label=f"Speaker {speaker_labels[i]}" if f"Speaker {speaker_labels[i]}" not in plt.gca().get_legend_handles_labels()[1] else ""
    )
plt.xlabel("Time (s)")
plt.yticks([])
plt.title("Speaker Diarization Result")
# 去重 legend
handles, labels_ = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels_, handles))
plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.01, 1), loc='upper left')
plt.tight_layout()
plt.show()

# X_embedded = TSNE(n_components=2, random_state=0, perplexity=len(filtered_embeddings) - 1).fit_transform(X)
# plt.scatter(X_embedded[:,0], X_embedded[:,1], c=labels, cmap='tab10')
# plt.title('Speaker Embedding t-SNE')
# plt.show()
