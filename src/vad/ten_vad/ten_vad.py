import numpy as np
from typing import List, Tuple
from src.vad.vad_base import VADBase, SpeechSegment
from ctypes import c_int, c_int32, c_float, c_size_t, CDLL, c_void_p, POINTER
import os
import sys

class TenVAD(VADBase):
    """
    基于能量门限的简单VAD实现（ten vad）。
    将音频分帧，检测每帧能量，连续高能量帧视为语音段。
    """
    def __init__(self, frame_size: int = 256, frame_shift: int = 160, energy_threshold: float = 0.5, min_speech_frames: int = 5, overlap_sec: float = 0.0):
        self.hop_size = frame_size
        self.threshold = energy_threshold
        self.overlap_sec = overlap_sec
        
        # 判断当前平台
        if sys.platform == "win32":
            self.vad_library = CDLL(
                os.path.join(
                    os.path.dirname(os.path.relpath(__file__)),
                    "./lib/Windows/x64/ten_vad.dll",
                )
            )
        else:
            if os.path.exists(
                os.path.join(
                    os.path.dirname(os.path.relpath(__file__)),
                    "./lib/Linux/x64/libten_vad.so",
                )
            ):
                self.vad_library = CDLL(
                    os.path.join(
                        os.path.dirname(os.path.relpath(__file__)), 
                        "./lib/Linux/x64/libten_vad.so",
                    )
                )
            else:
                self.vad_library = CDLL(
                    os.path.join(
                        os.path.dirname(
                            os.path.relpath(__file__)),
                            "./ten_vad_library/libten_vad.so",
                        )
                )
        
        self.vad_handler = c_void_p(0)
        self.out_probability = c_float()
        self.out_flags = c_int32()

        self.vad_library.ten_vad_create.argtypes = [
            POINTER(c_void_p),
            c_size_t,
            c_float,
        ]
        self.vad_library.ten_vad_create.restype = c_int

        self.vad_library.ten_vad_destroy.argtypes = [POINTER(c_void_p)]
        self.vad_library.ten_vad_destroy.restype = c_int

        self.vad_library.ten_vad_process.argtypes = [
            c_void_p,
            c_void_p,
            c_size_t,
            POINTER(c_float),
            POINTER(c_int32),
        ]
        self.vad_library.ten_vad_process.restype = c_int
        self.create_and_init_handler()

    def _process(self, audio: np.ndarray):
        input_pointer = self.get_input_data(audio)
        self.vad_library.ten_vad_process(
            self.vad_handler,
            input_pointer,
            c_size_t(self.hop_size),
            POINTER(c_float)(self.out_probability),
            POINTER(c_int32)(self.out_flags),
        )
        return self.out_probability.value, self.out_flags.value

    def create_and_init_handler(self):
        assert (
            self.vad_library.ten_vad_create(
                POINTER(c_void_p)(self.vad_handler),
                c_size_t(self.hop_size),
                c_float(self.threshold),
            ) 
            == 0
        ), "[TEN VAD]: create handler failure!"

    def __del__(self):
        assert (
            self.vad_library.ten_vad_destroy(
                POINTER(c_void_p)(self.vad_handler)
            )
            == 0
        ), "[TEN VAD]: destroy handler failure!"
    
    def get_input_data(self, audio_data: np.ndarray):
        audio_data = np.squeeze(audio_data)
        assert (
            len(audio_data.shape) == 1 
            and audio_data.shape[0] == self.hop_size
        ), "[TEN VAD]: audio data shape should be [%d], your shape is %s" % (
            self.hop_size,
            audio_data.shape[0]
        )
        assert (
            type(audio_data[0]) == np.int16
        ), "[TEN VAD]: audio data type error, must be int16"
        data_pointer = audio_data.__array_interface__["data"][0]
        return c_void_p(data_pointer)        

    def process(self, audio: np.ndarray, sr: int = 16000) -> List[SpeechSegment]:
        audio_data = (audio * 32767).astype(np.int16)
        speech_frame = self.get_speech_frame(audio_data)
        print(f"vad 分割后的音频个数: {len(speech_frame)}")

        speech_frame = self.merge_speech(speech_frame)
        print(f"合并处理后的音频个数: {len(speech_frame)}")

        segments = self.split_audio(audio, speech_frame, sr)
        
        return segments
    
    def get_speech_frame(self, audio: np.ndarray) -> List[SpeechSegment]:
        '''
        使用 ten_vad, 识别音频并划分说话片段。
        返回 List[SpeechFrame]
        '''
        num_frames = audio.shape[0] // self.hop_size
        pre_flag = 0
        start_index = 0
        speech_segment = []
        for i in range(num_frames):
            audio_data = audio[i * self.hop_size: (i + 1) * self.hop_size]
            _, out_flag = self._process(audio_data)
            if out_flag == 1:
                if pre_flag == 0:
                    start_index = i
            else:
                if pre_flag == 1:
                    speech_segment.append(SpeechSegment(start_frame=start_index, end_frame=i))
            pre_flag = out_flag
        if pre_flag == 1:
            speech_segment.append(SpeechSegment(start_frame=start_index, end_frame=num_frames))
        return speech_segment

    

    def merge_speech(self, speech_segments: List[SpeechSegment], sr = 16000) -> List[SpeechSegment]:
        '''
        根据 speech_frame 的内容，合并时间间隔小于 0.2s 的向相邻音频段
        '''
        if not speech_segments:
            return []
        merged = [speech_segments[0]]
        for i in range(1, len(speech_segments)):
            prev = merged[-1]
            curr = speech_segments[i]
            gap_frames = curr.start_frame - prev.end_frame
            gap_sec = gap_frames * self.hop_size / sr
            if gap_sec <= 0.2:
                prev.end_frame = curr.end_frame
            else:
                merged.append(curr)
        return merged
        



    def split_audio(self, audio: np.ndarray, speech_segments: List[SpeechSegment], sr: int = 16000) -> List[SpeechSegment]:
        '''
        根据 speech_segments, 对 audio 进行分割
        使用 self.overlap_sec 作为每段前后重叠时长（秒）
        修正：首段和末段如果重叠区不足，自动向另一端补偿，保证重叠区尽量完整且不越界。
        返回 [SpeechSegment{
                index: 0,       # 数组下标
                start_frame: 0, # 音频的开始帧
                end_frame: 1,   # 音频的结束帧
            }]
        '''
        audio_len = len(audio)
        index = 0
        frame_hop = self.hop_size
        overlap_frames = int(self.overlap_sec * sr / frame_hop)
        max_frame = audio_len // frame_hop
        merged_segments: List[SpeechSegment] = []
        for seg in speech_segments:
            # 理想的重叠区
            ideal_start = seg.start_frame - overlap_frames
            ideal_end = seg.end_frame + overlap_frames
            # 实际边界
            start_frame = max(0, ideal_start)
            end_frame = min(max_frame, ideal_end)
            # 左侧补偿：如果左边不够，右边补偿
            left_padding = 0 - ideal_start if ideal_start < 0 else 0
            # 右侧补偿：如果右边不够，左边补偿
            right_padding = ideal_end - max_frame if ideal_end > max_frame else 0
            # 补偿
            start_frame = max(0, start_frame - right_padding)
            end_frame = min(max_frame, end_frame + left_padding)
            # 再次校验
            start_frame = max(0, start_frame)
            end_frame = min(max_frame, end_frame)
            merged_segments.append(SpeechSegment(index=index, start_frame=start_frame, end_frame=end_frame))
            index += 1
        return merged_segments


    
    def merge_segments(self, segments: List[SpeechSegment]) -> List[SpeechSegment]:
        '''
        合并有重叠的相邻音频段
        '''
        if not segments:
            return []
        merged_segments: List[SpeechSegment] = [segments[0]]
        index = 1
        for i in range(1, len(segments)):
            prev = merged_segments[-1]
            curr = segments[i]
            if curr.start_frame <= prev.end_frame:
                # 合并
                new_start = prev.start_frame
                new_end = max(prev.start_frame, curr.end_frame)
                merged_segments[-1] = SpeechSegment(
                    index = index - 1,
                    start_frame = new_start,
                    end_frame = new_end,
                )
            else:
                curr.index = index
                merged_segments.append(curr)
                index += 1
        return merged_segments


    def del_noise(self, segments: List) -> List[Tuple[np.ndarray, int]]:
        '''
        剔除
        '''
        return
