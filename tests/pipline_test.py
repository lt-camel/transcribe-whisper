import sys
import os


print(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.pipeline.transcribe_pipeline import TranscribePipeline
from src.asr.whisper_asr.whisper_asr import WhisperASR
from src.audio_processing.loader_librosa.loader import AudioLoaderLibrosa
from src.vad.ten_vad.ten_vad import TenVAD
from src.diarization.pyannote_diarization.pyannote_diarization import PyannoteDiarization


audio_file = "C:/Users/camel/Downloads/192.168.16.85/202507031501/X5MINI-0WS12_20250702_143038.aac"
model_name = "C:/Users/camel/.cache/huggingface/hub/models--openai--whisper-small/snapshots/973afd24965f72e36ca33b3055d56a652f456b4d"
EMBEDDING_MODEL = "models/embedding/pytorch_model.bin"
EMBEDDING_WINDOW = "whole"
lora_model = None
if sys.platform == "linux":
    os.environ["LD_LIBRARY_PATH"] = os.path.abspath("../../src/vad/ten_vad/lib/os") + ";" + os.environ["LD_LIBRARY_PATH"]
    # model_name = "/data/whisper-finetune/whisper-finetune/model/BELLE-2/Belle-whisper-large-v3-zh-punct"
    model_name = "/data/whisper-finetune/whisper-finetune/Belle-zh-punct-finetune7/model7"
    # model_name = "/data/whisper-finetune/whisper-finetune/lora-whisper-finetuned"
    audio_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "vad/X5MINI-0WS12_20250702_143038.aac"))
    # lora_model = '/data/whisper-finetune/whisper-finetune/lora-whisper-finetuned/checkpoint-500'
    EMBEDDING_MODEL = "models/embedding/pytorch_model.bin"
    EMBEDDING_WINDOW = "whole"
    

audio_loader = AudioLoaderLibrosa()
vad = TenVAD(frame_size=128, energy_threshold=0.65, overlap_sec=0.16)
diarization_model = PyannoteDiarization()
diarization_model.load_model(EMBEDDING_MODEL, EMBEDDING_WINDOW)
asr = WhisperASR(model_name, lora_model)
pip = TranscribePipeline(loader=audio_loader, preprocessor=None, vad=vad, diarization_model=diarization_model, asr=asr)

result = pip(audio_file)

print(result)