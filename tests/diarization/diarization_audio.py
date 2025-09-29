import sys
import os
print(os.path.abspath(os.path.join(os.path.dirname(__file__))))
print(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.vad.ten_vad.ten_vad import TenVAD
from src.audio_processing.loader_librosa.loader import AudioLoaderLibrosa
from src.diarization.pyannote_diarization.pyannote_diarization import PyannoteDiarization
from src.utils.speech_segment_utils import duration_time, merge_speaker, frame_time
import soundfile as sf


# audio_file = "C:/Users/camel/Downloads/192.168.16.85/202507031500/X5MINI-0WS12_20250702_173636.aac"
# audio_file = "C:/Users/camel/Downloads/192.168.16.85/202507111803/merged.wav"
# audio_file = "C:/Users/camel/Downloads/192.168.16.85/202507141626/device1_20250702_095000.aac"
# audio_file = "C:/Users/camel/Downloads/192.168.16.85/device1_20250702_142451.aac"
# audio_file = "C:/Users/camel/Downloads/192.168.16.85/202507111714/merged.wav"
audio_file = "C:/Users/camel/Downloads/192.168.16.85/202507031501/X5MINI-0WS12_20250702_143038.aac"
# audio_file = "C:/Users/camel/Downloads/192.168.16.85/202507151116/device1_20250702_154231.aac"
# audio_file = "C:/Users/camel/Downloads/192.168.16.85/202507091804/_1752055133704_1752055153428_0.mp3"
model_name = "C:/Users/camel/.cache/huggingface/hub/models--openai--whisper-small/snapshots/973afd24965f72e36ca33b3055d56a652f456b4d"
if sys.platform == "linux":
    os.environ["LD_LIBRARY_PATH"] = os.path.abspath("../../src/vad/ten_vad/lib/os") + ";" + os.environ["LD_LIBRARY_PATH"]
    model_name = "/data/whisper-finetune/whisper-finetune/Belle-zh-punct-finetune7/model7"
    # audio_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "X5MINI-0WS12_20250702_143038.aac"))
    audio_file = "/data/hls/75VRIRJSKUP7OD99/2025-07-09/1752055153366/_1752055133704_1752055153428_0.mp3"

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
vad = TenVAD(frame_size=hop_size, energy_threshold=0.65, overlap_sec=0.18)


# librosa加载音频
audioLoader = AudioLoaderLibrosa()
audio, sr = audioLoader.load_from_file(audio_file, 16000)
# 降噪
# from src.audio_processing.preprocess_nr.preprocess import AudioPreprocessorNR
# audioProcessor = AudioPreprocessorNR()
# audio = audioProcessor.denoise(audio, sr)
# audio = audioProcessor.enhance(audio, sr)


print("===" * 20)
print("===" * 20)
segments = vad.process(audio, sr)

labels = p_diarization.diarize(audio, sr, segments, 2, 0.8, hop_size)

segments, labels = merge_speaker(segments, labels)

for i, part in enumerate(segments):
    start, end = frame_time(part, sr, hop_size)
    duration = duration_time(part, sr, hop_size)
    print(i)
    print(f"start: {start}")
    print(f"end: {end}")
    print(f"duration: {duration}")
    print("===" * 20)
    if sys.platform == "win32":
        segment = audio[part.start_frame * hop_size:part.end_frame * hop_size]
        sf.write(f"gen_audio/segment_{i}.wav", segment, 16000)


