from typing import List, Tuple

import numpy as np

from src.vad.vad_base import SpeechSegment


def duration_time(segment: SpeechSegment, sr: int, frame_hop: int) -> float:
    """
    计算单个音频段的持续时间（秒）。
    :param segment: SpeechSegment对象
    :param sr: 采样率
    :param frame_hop: 帧移
    :return: 持续时间（秒）
    """
    return round((segment.end_frame - segment.start_frame) * frame_hop / sr, 3)


def split_audio(audio: np.ndarray, segments: List[SpeechSegment], frame_hop: int = 160) -> List[np.ndarray]:
    """
    根据音频段的开始、结束帧，分割音频。
    :param audio: 原始音频
    :param segments: SpeechSegment列表
    :param frame_hop: 帧移
    :return: 切割后的音频片段列表
    """
    result = []
    for seg in segments:
        result.append(get_segment_audio(audio, seg, frame_hop))
    return result


def get_segment_audio(audio: np.ndarray, segment: SpeechSegment, frame_hop: int = 160) -> List[np.ndarray]:
    """
    根据音频段的开始、结束帧，分割音频。
    :param audio: 原始音频
    :param segment: SpeechSegment
    :param frame_hop: 帧移
    :return: 切割后的音频片段
    """
    start_sample = segment.start_frame * frame_hop
    end_sample = segment.end_frame * frame_hop
    return audio[start_sample:end_sample]


def frame_time(segment: SpeechSegment, sr: int, frame_hop: int) -> Tuple[float, float]:
    """
    计算音频段帧的开始和结束时间（秒）。
    :param segment: SpeechSegment对象
    :param sr: 采样率
    :param frame_hop: 帧移
    :return: (start_time, end_time) 单位：秒
    """
    start_time = round(segment.start_frame * frame_hop / sr, 3)
    end_time = round(segment.end_frame * frame_hop / sr, 3)
    return start_time, end_time


def merge_speaker(segments: List[SpeechSegment], speaker_labels: List[int]) -> Tuple[List[SpeechSegment], List[int]]:
    """
    根据VAD结果（SpeechSegment）和说话人标签，将相邻且同说话人的音频段合并。
    :param segments: SpeechSegment列表
    :param speaker_labels: 每个段的说话人标签
    :return: (合并后的SpeechSegment列表, 合并后每段的说话人标签)
    """
    if not segments or not speaker_labels or len(segments) != len(speaker_labels):
        return [], []
    merged_segments = []
    merged_labels = []
    cur_speaker = speaker_labels[0]
    cur_start = segments[0].start_frame
    cur_end = segments[0].end_frame
    idx = 0
    for i in range(1, len(segments)):
        if speaker_labels[i] == cur_speaker:
            cur_end = segments[i].end_frame
        else:
            merged_segments.append(SpeechSegment(index=idx, start_frame=cur_start, end_frame=cur_end))
            merged_labels.append(cur_speaker)
            idx += 1
            cur_speaker = speaker_labels[i]
            cur_start = segments[i].start_frame
            cur_end = segments[i].end_frame
    merged_segments.append(SpeechSegment(index=idx, start_frame=cur_start, end_frame=cur_end))
    merged_labels.append(cur_speaker)
    return merged_segments, merged_labels


def merge_and_split_audio_by_speaker(
        audio: np.ndarray,
        segments: List[SpeechSegment],
        speaker_labels: List[int],
        frame_hop: int = 160
) -> Tuple[List[SpeechSegment], List[np.ndarray], List[int]]:
    """
    根据VAD结果（SpeechSegment）和说话人标签，合并相邻同说话人段并切割音频。
    :param audio: 原始音频
    :param segments: SpeechSegment列表
    :param speaker_labels: 每个段的说话人标签
    :param frame_hop: 帧移
    :return: (合并后的SpeechSegment列表, 分割后的音频数组列表, 合并后每段的说话人标签)
    """
    merged_segments, merged_labels = merge_speaker(segments, speaker_labels)
    audio_segments = split_audio(audio, merged_segments, frame_hop)
    return merged_segments, audio_segments, merged_labels
