from pipeline.transcribe_pipeline import TranscribePipeline
from audio_processing.loader import AudioLoader
from audio_processing.preprocess import AudioPreprocessor
from vad.ten_vad import TenVAD
from asr.whisper_asr import WhisperASR
import sys

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python main.py <音频文件路径>")
        exit(1)
    file_path = sys.argv[1]
    # 按接口实例化各模块
    loader = AudioLoader()
    preprocessor = AudioPreprocessor()
    vad = TenVAD()
    asr = WhisperASR()
    pipeline = TranscribePipeline(loader, preprocessor, vad, asr)
    try:
        results = pipeline.run(file_path)
        for idx, text in enumerate(results):
            print(f"片段{idx+1}: {text}")
    except NotImplementedError:
        print("部分功能尚未实现，请完善各模块代码。")
