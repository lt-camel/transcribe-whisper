import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.vad.ten_vad.ten_vad import TenVAD
from src.audio_processing.loader_librosa.loader import AudioLoaderLibrosa
from src.diarization.pyannote_diarization.pyannote_diarization import PyannoteDiarization


audio_file = "C:/Users/camel/Downloads/192.168.16.85/202507151116/device1_20250702_154231.aac"

embedding_model = "D:/workspace/project/ai-model/transcribe-whisper/models/embedding/pytorch_model.bin"
embedding_window = "whole"


diarization = PyannoteDiarization()
print("开始加载 embedding 模型")
diarization.load_model(embedding_model, embedding_window)
print("模型加载完成")


# 初始化 TenVAD
print("开始加载 TenVAD 模型")
vad = TenVAD(frame_size=128, energy_threshold=0.65, overlap_sec=0.15)
print("模型加载完成")


audioLoader = AudioLoaderLibrosa()


audio, sr = audioLoader.load_from_file(audio_file)

segments = vad.process(audio, sr)

MIN_DURATION = 0.8
result = diarization.diarize(audio, sr, segments, 2, MIN_DURATION)


for i, line in enumerate(result):
    print(f"{i:>3} {line["segment_index"]:>3} [{line["start"]:>6.3f} - {line["end"]:>6.3f}] [{line["duration"]:>6.3f}] : speaker = {line["speaker"]:>2} |\n")


