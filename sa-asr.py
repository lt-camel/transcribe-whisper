audio_file = "C:/Users/camel/Downloads/192.168.16.85/202507211039/309F-H02_20250718_184521.wav"


# from funasr import AutoModel
# # paraformer-zh is a multi-functional asr model
# # use vad, punc, spk or not as you need
# model = AutoModel(model="paraformer-zh", model_revision="v2.0.4",
#                   vad_model="fsmn-vad", vad_model_revision="v2.0.4",
#                   punc_model="ct-punc-c", punc_model_revision="v2.0.4",
#                   spk_model="cam++", spk_model_revision="v2.0.2",
#                   )

# res = model.generate(input=audio_file, 
#             batch_size_s=300, 
#             hotword="监护人|操作人|AI",    # 可选：角色关键词提示（结构化对话场景）
#             param_dict={
#                 "spk_num": 3       # 固定说话人数（2人或3人）
#                 }
#             )
# print(res)


# from src.vad.ten_vad.ten_vad import TenVAD
# from src.utils.speech_segment_utils import split_audio, frame_time, duration_time
# import librosa
# audio, sr = librosa.load(path=audio_file, sr=16000)
# print(f"sr = {sr}")
# hop_size = 128
# vad = TenVAD(frame_size=hop_size, energy_threshold=0.65, overlap_sec=0.16)
# segments = vad.process(audio, sr)

# audio_segments = split_audio(audio, segments, 128)

# import torch
# import torchaudio
# from speechbrain.inference.speaker import SpeakerRecognition  # 使用新接口
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.cluster import AgglomerativeClustering
# import numpy as np

# # 初始化 ECAPA-TDNN 声纹模型（SpeechBrain 1.0+）
# model = SpeakerRecognition.from_hparams(
#     source="D:/workspace/project/ai-model/transcribe-whisper/models/spkrec-ecapa-voxcele",
#     # savedir="tmp_ecapa_model",
#     run_opts={"device": "cuda"} if torch.cuda.is_available() else {"device": "cpu"}
# )

# # 假设 audio_segments 是 VAD 分割后的音频路径列表
# # audio_segments = ["segment1.wav", "segment2.wav", "segment3.wav"]

# # 提取所有片段的声纹嵌入
# long_segments = [seg for seg in segments if duration_time(seg, sr, hop_size) > 0.8]
# long_audio_segments = [audio_segments[seg.index] for seg in segments if duration_time(seg, sr, hop_size) > 0.8]
# embeddings = []
# for audio in long_audio_segments:
#     waveform = torch.from_numpy(audio).float()
#     embedding = model.encode_batch(waveform).squeeze(0).squeeze(0).cpu().numpy()
#     # print(embedding.shape)  # 打印: (1, 192)
#     embeddings.append(embedding)

# # 计算相似度矩阵（N x N）
# similarity_matrix = cosine_similarity(embeddings)
# # similarity_matrix = embeddings

# # 层次聚类（假设 2 个说话人）
# # cluster = AgglomerativeClustering(
# #     n_clusters=3,
# #     # affinity="precomputed",  # 使用预计算的相似度矩阵
# #     linkage="average"        # 平均链接法更鲁棒
# # )
# # labels = cluster.fit_predict(similarity_matrix)

# from sklearn.cluster import SpectralClustering
# cluster = SpectralClustering(n_clusters=3, affinity="precomputed")
# labels = cluster.fit_predict(similarity_matrix)

# # 输出结果
# for i, (seg, label) in enumerate(zip(long_segments, labels)):
#     start, end = frame_time(seg, sr, hop_size)
#     duration = duration_time(seg, sr, hop_size)
#     print(f"Segment {i:>3}_{seg.index:>3} [{start:>6.3f}, {end:>6.3f}] [{duration:>6.3f}]: speaker={label:>2}")

# 版本要求 modelscope version 升级至最新版本 funasr 升级至最新版本

# from modelscope.pipelines import pipeline
# sd_pipeline = pipeline(
#     task='speaker-diarization',
#     model='iic/speech_eres2net-large_speaker-diarization_common',
#     model_revision='v1.0.0'
# )
# # 如果有先验信息，输入实际的说话人数，会得到更准确的预测结果
# result = sd_pipeline(audio_file, oracle_num=3)
# print(result)

from pathlib import Path

import sherpa_onnx
import soundfile as sf
from src.vad.vad_base import SpeechSegment
from src.utils.speech_segment_utils import duration_time, frame_time


def init_speaker_diarization(num_speakers: int = -1, cluster_threshold: float = 0.5):
    """
    Args:
      num_speakers:
        If you know the actual number of speakers in the wave file, then please
        specify it. Otherwise, leave it to -1
      cluster_threshold:
        If num_speakers is -1, then this threshold is used for clustering.
        A smaller cluster_threshold leads to more clusters, i.e., more speakers.
        A larger cluster_threshold leads to fewer clusters, i.e., fewer speakers.
    """
    segmentation_model = "./models/onnx/sherpa-onnx-pyannote-segmentation-3-0/model.onnx"
    embedding_extractor_model = (
        "./models/onnx/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
    )

    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=segmentation_model
            ),
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=embedding_extractor_model
        ),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=num_speakers, threshold=cluster_threshold
        ),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    if not config.validate():
        raise RuntimeError(
            "Please check your config and make sure all required files exist"
        )

    return sherpa_onnx.OfflineSpeakerDiarization(config)


def progress_callback(num_processed_chunk: int, num_total_chunks: int) -> int:
    progress = num_processed_chunk / num_total_chunks * 100
    print(f"Progress: {progress:.3f}%")
    return 0

def convert_time_to_indices(time_objects, sample_rate):
    """
    将时间格式的对象转换为音频数组下标格式的对象
    
    参数:
        time_objects: 时间格式的对象列表，如 [{'start': 1.0, 'end': 2.0}]
        sample_rate: 音频的采样率（每秒采样点数）
        
    返回:
        转换后的对象列表，如 [{'start': 1, 'end': 2}]
    """
    converted = []
    for i, obj in enumerate(time_objects):
        converted.append(SpeechSegment(
            start_frame = int(round(obj.start * sample_rate)),
            end_frame = int(round(obj.end * sample_rate)),
            index = i,
        ))
    return converted


def main():
    wave_filename = audio_file
    if not Path(wave_filename).is_file():
        raise RuntimeError(f"{wave_filename} does not exist")

    audio, sample_rate = sf.read(wave_filename, dtype="float32", always_2d=True)
    print(f"sample_rate: {sample_rate}")
    print(f"len: {len(audio)}")
    audio = audio[:, 0]  # only use the first channel

    # Since we know there are 4 speakers in the above test wave file, we use
    # num_speakers 4 here
    sd = init_speaker_diarization(num_speakers=3)
    if sample_rate != sd.sample_rate:
        raise RuntimeError(
            f"Expected samples rate: {sd.sample_rate}, given: {sample_rate}"
        )

    show_progress = True

    if show_progress:
        result = sd.process(audio, callback=progress_callback).sort_by_start_time()
    else:
        result = sd.process(audio).sort_by_start_time()

    # for r in result:
    #     print(f"{r.start:.3f} -- {r.end:.3f} speaker_{r.speaker:02}")
        #  print(r) # this one is simpler

    print(f"sample_rate: {sample_rate}")
    print(f"len: {len(audio)}")
    segments = convert_time_to_indices(result, sample_rate)
    for seg in segments:
        start, end = frame_time(seg, sample_rate, 1)
        duration = duration_time(seg, sample_rate, 1)
        print(f"Segment {seg.index:>3} [{start:>6.3f}, {end:>6.3f}] [{duration:>6.3f}]")

    



if __name__ == "__main__":
    main()