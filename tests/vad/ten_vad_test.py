import sys
import os
print(os.path.abspath(os.path.join(os.path.dirname(__file__))))
print(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.vad.ten_vad.ten_vad import TenVAD
from src.audio_processing.loader_librosa.loader import AudioLoaderLibrosa
import soundfile as sf




# audio_file = "C:/Users/camel/Downloads/192.168.16.85/device1_20250702_142451.aac"
audio_file = "C:/Users/camel/Downloads/192.168.16.85/202507211039/309F-H02_20250718_184521.aac"
model_name = "C:/Users/camel/.cache/huggingface/hub/models--openai--whisper-small/snapshots/973afd24965f72e36ca33b3055d56a652f456b4d"
if sys.platform == "linux":
    os.environ["LD_LIBRARY_PATH"] = os.path.abspath("../../src/vad/ten_vad/lib/os") + ";" + os.environ["LD_LIBRARY_PATH"]
    model_name = "/data/whisper-finetune/whisper-finetune/Belle-zh-punct-finetune7/model7"
    # audio_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "X5MINI-0WS12_20250702_143038.aac"))
    audio_file = "/data/hls/75VRIRJSKUP7OD99/2025-07-09/1752055153366/_1752055133704_1752055153428_0.mp3"

# 初始化 TenVAD
hop_size = 128
vad = TenVAD(frame_size=hop_size, energy_threshold=0.65, overlap_sec=0.16)


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

# print(segments)

for i, part in enumerate(segments):
    start = round(part.start_frame * hop_size / sr, 3)
    end = round(part.end_frame * hop_size / sr, 3)
    duration = round(end - start, 3)
    print(i)
    print(f"start: {start}")
    print(f"end: {end}")
    print(f"duration: {duration}")
    print("===" * 20)
    if sys.platform == "win32":
        segment = audio[part.start_frame * hop_size:part.end_frame * hop_size]
        sf.write(f"gen_audio/segment_{i}.wav", segment, 16000)

print(len(segments))

print("===" * 20)


if sys.platform == "linux":
    from src.asr.whisper_asr.whisper_asr import WhisperASR
    whisper = WhisperASR(model_name=model_name)
    print("模型加载完成")
    transcription = whisper.transcribe([part.array for part in segments], sr)

    print("===" * 20)

    print(transcription)
