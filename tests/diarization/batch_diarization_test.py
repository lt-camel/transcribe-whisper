import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.vad.ten_vad.ten_vad import TenVAD
from src.audio_processing.loader_librosa.loader import AudioLoaderLibrosa
from src.diarization.pyannote_diarization.pyannote_diarization import PyannoteDiarization
from src.utils.speech_segment_utils import duration_time, split_audio, get_segment_audio, frame_time, merge_speaker

# 配置
AUDIO_ROOT = "C:/Users/camel/Downloads/192.168.16.85"
EMBEDDING_MODEL = "models/embedding/pytorch_model.bin"
ASR_MODEL_PATH = "C:/Users/camel/.cache/huggingface/hub/models--openai--whisper-small/snapshots/973afd24965f72e36ca33b3055d56a652f456b4d"
EMBEDDING_WINDOW = "whole"
AUDIO_EXTS = {".wav", ".mp3", ".aac", ".flac", ".m4a", ".ogg"}
MIN_DURATION = 0.8  # 秒

transcribe = False

if sys.platform == 'linux':
    AUDIO_ROOT = "/data/whisper-finetune/dual_speaker_diarization/audio_data/mechanical_assembly"
    ASR_MODEL_PATH = "/data/whisper-finetune/whisper-finetune/Belle-zh-punct-finetune7/model7"
    transcribe = True
    AUDIO_EXTS = {".aac"}



OUTPUT_DIR = "./diarization_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 初始化模型
p_diarization = PyannoteDiarization()
p_diarization.load_model(EMBEDDING_MODEL, EMBEDDING_WINDOW)

hop_size = 128
vad = TenVAD(frame_size=hop_size, energy_threshold=0.65, overlap_sec=0.16)
audioLoader = AudioLoaderLibrosa()


if transcribe:
    from src.asr.whisper_asr.whisper_asr import WhisperASR
    asr = WhisperASR(model_name=ASR_MODEL_PATH)

# 遍历所有音频文件
audio_files = []

for root, dirs, files in os.walk(AUDIO_ROOT):
    for file in files:
        ext = os.path.splitext(file)[1].lower()
        if ext in AUDIO_EXTS and file.lower().startswith("device1"):
            audio_files.append(os.path.join(root, file))

print(f"共找到 {len(audio_files)} 个音频文件")

for wav_path in audio_files:
    print(f"处理: {wav_path}")
    try:
        import time
        t0 = time.time()

        audio, sr = audioLoader.load_from_file(wav_path, 16000)

        segments = vad.process(audio, sr)

        labels = p_diarization.diarize(audio, sr, segments, 2, MIN_DURATION, hop_size)

        print("根据 speaker_labels 合并 segments:")
        merged_segments, merged_speaker_labels =merge_speaker(segments, labels)
        print("合并完成")

        vad_embed_time = time.time() - t0
        print(f"VAD+embedding耗时: {vad_embed_time:.2f} 秒")

        # 保存结果
        # 格式为：index [start - end]: speaker = 0
        # 保存为 txt 文件
        rel_path = os.path.relpath(wav_path, AUDIO_ROOT)
        base, _ = os.path.splitext(rel_path)
        out_txt = os.path.join(OUTPUT_DIR, base + ".txt")
        os.makedirs(os.path.dirname(out_txt), exist_ok=True)
        asr_time = 0
        with open(out_txt, "w", encoding="utf-8") as f:
            for i, seg in enumerate(merged_segments):
                speaker = int(merged_speaker_labels[i]) if merged_speaker_labels[i] is not None else -1
                start, end = frame_time(seg, sr, hop_size)
                text = ""
                if transcribe:
                    try:
                        t1 = time.time()
                        text = asr.transcribe([get_segment_audio(audio, seg, hop_size)], sr)[0]  # 假设返回列表
                        asr_time += time.time() - t1
                    except Exception as e:
                        text = f"[ASR ERROR: {e}]"
                f.write(f"{i:>3} [{start:>8.3f} - {end:>8.3f}]: speaker = {speaker:>2} | {text}\n")
        if transcribe:
            print(f"ASR转录总耗时: {asr_time:.2f} 秒")
        print(f"保存结果到: {out_txt}")

    except Exception as e:
        import traceback
        print(f"处理 {wav_path} 时出错: ")
        traceback.print_exc() 