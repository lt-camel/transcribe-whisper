import numpy as np
from typing import List, Tuple
from src.vad.vad_base import VADBase, Segment
from ctypes import c_int, c_int32, c_float, c_size_t, CDLL, c_void_p, POINTER
import os
import sys
from dataclasses import dataclass

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@dataclass
class SpeechFrame:
    start: int  # 起始帧索引
    end: int    # 结束帧索引

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

    def process(self, audio: np.ndarray, sr: int = 16000) -> List[Segment]:
        # 使用 ten vad 检测音频帧是否存在人声
        audio_data = (audio * 32767).astype(np.int16)
        speech_frame = self.get_speech_frame(audio_data)
        print(f"vad 分割后的音频个数: {len(speech_frame)}")

        speech_frame = self.merge_speech(speech_frame)
        print(f"合并处理后的音频个数: {len(speech_frame)}")

        segments = self.split_audio(audio, speech_frame, sr)
        
        return segments
    
    def get_speech_frame(self, audio: np.ndarray) -> List[SpeechFrame]:
        '''
        使用 ten_vad, 识别音频并划分说话片段。
        返回 List[SpeechFrame]
        '''
        num_frames = audio.shape[0] // self.hop_size
        pre_flag = 0
        start_index = 0
        speech_frame = []
        for i in range(num_frames):
            audio_data = audio[i * self.hop_size: (i + 1) * self.hop_size]
            _, out_flag = self._process(audio_data)
            if out_flag == 1:
                if pre_flag == 0:
                    start_index = i
            else:
                if pre_flag == 1:
                    speech_frame.append(SpeechFrame(start=start_index, end=i))
            pre_flag = out_flag
        if pre_flag == 1:
            speech_frame.append(SpeechFrame(start=start_index, end=num_frames))
        return speech_frame

    

    def merge_speech(self, speech_frame: List[SpeechFrame], sr = 16000) -> List[SpeechFrame]:
        '''
        根据 speech_frame 的内容，合并时间间隔小于 0.2s 的向相邻音频段
        '''
        if not speech_frame:
            return []
        merged = [speech_frame[0]]
        for i in range(1, len(speech_frame)):
            prev = merged[-1]
            curr = speech_frame[i]
            gap_frames = curr.start - prev.end
            gap_sec = gap_frames * self.hop_size / sr
            if gap_sec <= 0.2:
                prev.end = curr.end
            else:
                merged.append(curr)
        return merged
        



    def split_audio(self, audio: np.ndarray, speech_frame: List[SpeechFrame], sr: int = 16000) -> List[Segment]:
        '''
        根据 speech_frame, 对 audio 进行分割
        使用 self.overlap_sec 作为每段前后重叠时长（秒）
        返回 [{
                "start": 0.00,      # 音频的开始时间（秒）
                "end": 1.01,        # 音频的结束时间（秒）
                "duration": 1.01,   # 音频的持续时间（秒）
                "segment": []       # 音频段的音频数据数组
            }]
        '''
        segments: List[Segment] = []
        audio_len = len(audio)
        index = 0
        overlap_samples = int(self.overlap_sec * sr)
        for seg in speech_frame:
            start_sample = max(0, seg.start * self.hop_size - overlap_samples)
            end_sample = min(audio_len, seg.end * self.hop_size + overlap_samples)
            if end_sample <= start_sample:
                continue
            segment_audio = audio[start_sample:end_sample]
            start_sec = round(start_sample / sr, 3)
            end_sec = round(end_sample / sr, 3)
            duration = round((end_sample - start_sample) / sr, 3)

            segments.append(Segment(
                i = index,
                start = start_sec,
                end = end_sec,
                duration = duration,
                array = segment_audio
            ))
            index += 1

        return segments


    
    def merge_segments(segments: List[Segment], audio: np.ndarray, sr: int) -> List[Segment]:
        '''
        合并有重叠的相邻音频段
        '''
        if not segments:
            return []
        merged_segments: List[Segment] = [segments[0]]
        index = 1
        for i in range(1, len(segments)):
            prev = merged_segments[-1]
            curr = segments[i]
            if curr.start <= prev.end:
                # 合并
                new_start = prev.start
                new_end = max(prev.start, curr.end)
                new_start_sample = int(new_start * sr)
                new_end_sample = int(new_end * sr)
                new_segment_audio = audio[new_start_sample:new_end_sample]
                new_duration = round((new_end_sample - new_start_sample) / sr, 3)
                merged_segments[-1] = Segment(
                    i = index - 1,
                    start = new_start,
                    end = new_end,
                    duration = new_duration,
                    array = new_segment_audio
                )
            else:
                curr.i = index
                merged_segments.append(curr)
                index += 1
        return merged_segments



    def split_audio_back(self, audio: np.ndarray, speech_frame: List[SpeechFrame], sr: int = 16000) -> List[dict]:
        '''
        根据 speech_frame, 对 audio 进行分割
        使用 self.overlap_sec 作为每段前后重叠时长（秒）
        返回 [{
                "start": 0.00,      # 音频的开始时间（秒）
                "end": 1.01,        # 音频的结束时间（秒）
                "duration": 1.01,   # 音频的持续时间（秒）
                "segment": []       # 音频段的音频数据数组
            }]
        '''
        segments = []
        audio_len = len(audio)
        overlap_samples = int(self.overlap_sec * sr)
        for seg in speech_frame:
            start_sample = max(0, seg.start * self.hop_size - overlap_samples)
            end_sample = min(audio_len, seg.end * self.hop_size + overlap_samples)
            if end_sample <= start_sample:
                continue
            segment_audio = audio[start_sample:end_sample]
            start_sec = round(start_sample / sr, 3)
            end_sec = round(end_sample / sr, 3)
            duration = round((end_sample - start_sample) / sr, 3)
            segments.append({
                "start": start_sec,
                "end": end_sec,
                "duration": duration,
                "segment": segment_audio
            })
        return segments
    


    def del_noise(self, segments: List) -> List[Tuple[np.ndarray, int]]:
        '''
        剔除
        '''
        return
        

    def split(self, audio: np.ndarray, sr: int = 16000) -> List[Tuple[np.ndarray, int]]:
        frames = self._enframe(audio)
        energies = np.sum(frames ** 2, axis=1)
        speech_flags = energies > self.threshold
        segments = []
        start, in_speech = None, False
        for i, flag in enumerate(speech_flags):
            if flag and not in_speech:
                start = i
                in_speech = True
            elif not flag and in_speech:
                end = i
                if end - start >= self.min_speech_frames:
                    seg_audio = self._deframe(frames[start:end])
                    segments.append((seg_audio, sr))
                in_speech = False
        # 处理结尾
        if in_speech and start is not None and len(frames) - start >= self.min_speech_frames:
            seg_audio = self._deframe(frames[start:])
            segments.append((seg_audio, sr))
        return segments

    def _enframe(self, audio: np.ndarray) -> np.ndarray:
        num_frames = 1 + (len(audio) - self.frame_size) // self.frame_shift
        frames = np.stack([
            audio[i*self.frame_shift : i*self.frame_shift+self.frame_size]
            for i in range(num_frames)
        ])
        return frames

    def _deframe(self, frames: np.ndarray) -> np.ndarray:
        # 简单拼接帧
        return frames.flatten() 