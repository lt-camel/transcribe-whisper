import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.vad.ten_vad.ten_vad import TenVAD
from src.audio_processing.loader_librosa.loader import AudioLoaderLibrosa
import matplotlib.pyplot as plt
from src.diarization.pyannote_diarization.pyannote_diarization import PyannoteDiarization
from src.utils.speech_segment_utils import duration_time, merge_speaker, frame_time



audio_file = "C:/Users/camel/Downloads/192.168.16.85/202507211039/309F-H02_20250718_184521.aac"

embedding_model = "D:/workspace/project/ai-model/transcribe-whisper/models/embedding/pytorch_model.bin"
embedding_window = "whole"
# embedding_window =  "sliding"

print("开始加载 embedding 模型")
# 加载模型
p_diarization = PyannoteDiarization()
p_diarization.load_model(embedding_model, embedding_window)
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

# KMeans 聚类
# num_speakers = 2  # 可调整
# kmeans = KMeans(n_clusters=num_speakers, random_state=0)
# labels = kmeans.fit_predict(X)

# 为什么要丢弃部分短时间的音频。因为会扰乱识别效果
# TODO 将短时间的音频，单独做说话人识别，并与长时间的音频合并
min_duration = 0.8  # 秒
labels = p_diarization.diarize(audio, sr, segments, 3, min_duration, hop_size)

i = 0
for seg, label in zip(segments, labels):
    start, end = frame_time(seg, sr, hop_size)
    duration = duration_time(seg, sr, hop_size)
    print(f"Segment {i:>3}_{seg.index:>3} [{start:>6.3f}, {end:>6.3f}] [{duration:>6.3f}]: speaker={label:>2}")
    i += 1

print("根据 speaker_labels 合并 segments:")
merged_segments, merged_speaker_labels = merge_speaker(segments, labels)
print("合并完成")

for seg, label in zip(merged_segments, merged_speaker_labels):
    start, end = frame_time(seg, sr, hop_size)
    duration = duration_time(seg, sr, hop_size)
    print(f"Segment {i:>3}_{seg.index:>3} [{start:>6.3f}, {end:>6.3f}] [{duration:>6.3f}]: speaker={label:>2}")

# 生成颜色
# num_speakers = len(set(labels))
# colors = plt.cm.get_cmap('tab10', num_speakers)

# plt.figure(figsize=(12, 2))
# for i, segment in enumerate(segments):
#     plt.barh(
#         y=0,
#         width=(segment.end_frame - segment.start_frame) * hop_size / sr,
#         left=segment.start_frame * hop_size / sr,
#         height=0.5,
#         color=colors(labels[i]),
#         edgecolor='black',
#         label=f"Speaker {labels[i]}" if f"Speaker {labels[i]}" not in plt.gca().get_legend_handles_labels()[1] else ""
#     )
# plt.xlabel("Time (s)")
# plt.yticks([])
# plt.title("Speaker Diarization Result")
# # 去重 legend
# handles, labels_ = plt.gca().get_legend_handles_labels()
# by_label = dict(zip(labels_, handles))
# plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.01, 1), loc='upper left')
# plt.tight_layout()
# plt.show()
