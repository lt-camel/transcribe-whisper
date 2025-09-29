import numpy as np
from typing import List
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline
from peft import PeftModel
from src.asr.asr_base import ASRBase

class WhisperASR(ASRBase):
    def __init__(self, model_name=None, lora_model='', task="transcribe", language="zh"):
        assert(model_name), "未指定模型"
        self.model_name = model_name
        self.task = task
        self.language = language
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        self.processor = WhisperProcessor.from_pretrained(model_name)

        if not lora_model and lora_model != '':
            print("使用 lora 模型")
            model = PeftModel.from_pretrained(model, lora_model)


        self.pipeline = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            max_new_tokens=128,
            torch_dtype=self.torch_dtype,
            device=self.device,
            # return_timestamps=True,
            # return_timestamps="word",
            # chunk_length_s=30,  # 分块长度(秒)
            # stride_length_s=[2, 1],  # 前后各重叠5秒
        )

    def transcribe(self, audio_list: List[np.ndarray], sr: int = 16000):
        # input_features = self.processor(
        #         audio_list,
        #         sampling_rate=sr,
        #         return_tensors="pt",
        #         truncation=False,
        #     ).input_features.to(self.device)

        # generate_kwargs = {
        #     "input_features": input_features,
        #     "task": self.task,
        #     "language": self.language,
        #     "return_timestamps": True,  # 启用时间戳
        # }
        # predicted_ids = self.model.generate(**generate_kwargs)
        # return self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        kwargs = {
            "generate_kwargs": {
                "language": self.language,
                "task": self.task,  # 明确指定任务类型
            }
        }
        return self.pipeline(audio_list, **kwargs)
