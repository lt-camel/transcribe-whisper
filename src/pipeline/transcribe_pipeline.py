from src.audio_processing.loader_base import AudioLoaderBase
from src.audio_processing.preprocess_base import AudioPreprocessorBase
from src.utils.speech_segment_utils import get_segment_audio
from src.vad.vad_base import VADBase
from src.asr.asr_base import ASRBase
from src.diarization.diarization_base import DiarizationBase
from src.utils.speech_segment_utils import merge_speaker
from typing import List

class TranscribePipeline:
    """
    串联音频加载、预处理、VAD切割、ASR转录的主流程。
    支持依赖注入，便于扩展和替换各模块实现。
    """
    def __init__(self, loader: AudioLoaderBase,
        preprocessor: AudioPreprocessorBase,
        vad: VADBase,
        diarization_model: DiarizationBase,
        asr: ASRBase,
    ):
        self.loader = loader
        self.preprocessor = preprocessor
        self.vad = vad
        self.diarization_model = diarization_model
        self.asr = asr

    def __call__(self, file_path: str) -> List[str]:
        hop_size = self.vad.hop_size
        # 1. 加载音频
        audio, sr = self.loader.load_from_file(file_path)
        # 2. 预处理
        # audio = self.preprocessor.denoise(audio, sr)
        # audio = self.preprocessor.enhance(audio, sr)
        # 3. VAD切割
        segments = self.vad.process(audio)
        labels = self.diarization_model.diarize(audio, sr, segments, 2, 0.8, hop_size)

        segments, labels =merge_speaker(segments, labels)

        audio_segments = [get_segment_audio(audio, seg, hop_size) for seg in segments]
        # 4. ASR转录
        texts = self.asr.transcribe(audio_segments, sr)
        return texts 