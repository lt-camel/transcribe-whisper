# transcribe-whisper

本项目基于Whisper语音识别模型，结合ten_vad语音活动检测，实现高效的音频转录。

## 主要模块
- **audio_processing/**：音频加载与预处理（降噪、增强等）
- **vad/**：语音活动检测（VAD），默认ten_vad实现
- **asr/**：语音识别（ASR），默认Whisper实现
- **pipeline/**：主流程调度，串联音频处理、VAD、ASR
- **utils/**：工具类
- **tests/**：单元测试
- **docs/**：文档

## 使用方法
```bash
python main.py <音频文件路径>
```

## 依赖安装
```bash
pip install -r requirements.txt
```

