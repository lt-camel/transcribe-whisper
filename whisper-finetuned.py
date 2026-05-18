import torch
from datasets import load_dataset
from transformers import WhisperProcessor, WhisperTokenizer, WhisperFeatureExtractor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import soundfile as sf
import librosa

# 配置参数
MODEL_NAME = "./model/BELLE-2/Belle-whisper-large-v3-zh-punct"  # 可根据需要更换
DATASET_PATH = "datasets/datasets.jsonl"  # 数据集路径
OUTPUT_DIR = "./lora-whisper-finetuned2"
task="transcribe"
MAX_INPUT_LENGTH = 30  # 音频最大长度（秒）
BATCH_SIZE = 1
NUM_TRAIN_EPOCHS = 3
LEARNING_RATE = 1e-4
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 1. 加载 jsonl 数据集
def load_jsonl_dataset(feature_extractor, tokenizer):
    print("======= 加载 本地 数据集 =========")

    # 加载数据集
    tl_dataset = load_dataset("json", data_files={
        # "train": "datasets/login_datasets.jsonl",
        # "test": "datasets/login_datasets_test.jsonl",
        "train": "datasets/datasets.jsonl",
        "test": "datasets/datasets-eval.jsonl",
    }, cache_dir="./cache")
    # tl_dataset = DatasetDict()

    print(f"训练数据集个数{len(tl_dataset["train"])}")
    print(f"验证数据集个数{len(tl_dataset["test"])}")

    # 预处理数据集
    from datasets import Audio
    def prepare_dataset(data):
        audio = data["file_name"]
        data["path"] = audio["path"]
        data["audio"] = audio
        # 对音频数据进行预处理，将其转换为模型可以接受的输入格式
        data["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[
            0]
        # encode target text to label ids
        # 对文本进行分词，然后将分词后的结果转换为标签ID
        data["labels"] = tokenizer(data["sentence"]).input_ids
        return data

    tl_dataset = tl_dataset.cast_column("file_name", Audio(sampling_rate=16000))

    # print(tl_dataset["train"][0])

    tl_dataset = tl_dataset.map(prepare_dataset)
    return tl_dataset


if __name__ == "__main__":

    # 加载模型和分词器
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.enable_input_require_grads()  # 关键：确保输入张量可微分

    feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)
    tokenizer = WhisperTokenizer.from_pretrained(MODEL_NAME, task=task)
    processor = WhisperProcessor.from_pretrained(MODEL_NAME, task=task)
    model.config.forced_decoder_ids = None  # 允许多语言
    model.config.suppress_tokens = []
    
    # 加载数据集
    dataset = load_jsonl_dataset(feature_extractor, tokenizer)

    # LoRA 配置
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    # 设置格式
    dataset.set_format(type="torch", columns=["input_features", "labels"])

    # 训练参数
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,  # change to a repo name of your choice
        push_to_hub=False,
        # 训练相关
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=4,  # increase by 2x for every 2x decrease in batch size
        learning_rate=LEARNING_RATE,
        warmup_steps=500,
        max_steps=4000,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        gradient_checkpointing=True,  # 使用梯度检查点，减少显存占用（但会增加计算时间）。使用 lora 时，开启此参数，需要先执行 model.enable_input_require_grads()
        fp16=True,
        # 评估与保存
        eval_strategy="steps",
        per_device_eval_batch_size=BATCH_SIZE,
        save_steps=1000,
        eval_steps=500,
        load_best_model_at_end=False,    # 训练结束时加载最佳模型
        metric_for_best_model="eval_wer",
        greater_is_better=False,
        generation_config=model.generation_config,  # 显式传递生成配置
        # 生成任务专用
        predict_with_generate=True,
        generation_max_length=225,
        # 日志与监控
        logging_steps=100,
        report_to=["tensorboard"],
    )

    # 定义 Trainer
    class DataCollatorWhisper:
        def __init__(self, processor):
            self.processor = processor

        def __call__(self, features):
            # split inputs and labels since they have to be of different lengths and need different padding methods
            # first treat the audio inputs by simply returning torch tensors
            input_features = [{"input_features": feature["input_features"]} for feature in features]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

            # get the tokenized label sequences
            label_features = [{"input_ids": feature["labels"]} for feature in features]
            # pad the labels to max length
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

            # replace padding with -100 to ignore loss correctly
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

            # if bos token is appended in previous tokenization step,
            # cut bos token here as it's append later anyways
            if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
                labels = labels[:, 1:]

            batch["labels"] = labels

            return batch

    import evaluate
    metric = evaluate.load("./metrics/wer")
    def compute_metrics(pred):
        print("=========== compute_metrics ===========")
        # print(f"======== {len(pred.predictions)} =========")
        # print(f"======== {len(pred.predictions[0])} =========")
        # pred_ids = np.argmax(pred.predictions[0], axis=-1)    # whisper-small 的数据格式
        # pred_ids = pred.predictions                           # Belle-whisper-large-v3-zh-punct 全参微调
        pred_ids = pred.predictions[0]                          # Belle-whisper-large-v3-zh-punct + lora 部分参数微调
        label_ids = pred.label_ids


        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        # print("======== pred_str =========")
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        # print(pred_str)
        # print("======== label_str =========")
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        # print(label_str)

        wer = 100 * metric.compute(predictions=pred_str, references=label_str)

        return {"eval_wer": wer}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"].select(range(100)),
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWhisper(processor),
        tokenizer=processor.feature_extractor,
    )
    
    # 模拟评估
    dummy_eval = dataset["test"].select(range(2))
    eval_result = trainer.evaluate(dummy_eval)
    print("评估结果:", eval_result)

    # 开始训练
    trainer.train()

    # 保存模型（LoRA权重未合并）
    # model.save_pretrained(OUTPUT_DIR)
    # processor.save_pretrained(OUTPUT_DIR)
    # print(f"模型已保存到 {OUTPUT_DIR}")

    # 1. 合并 LoRA 权重并保存完整模型
    print("正在合并 LoRA 权重...")
    model = model.merge_and_unload()
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(f"合并权重后的模型已保存到 {OUTPUT_DIR}")

    # 2. 转换为 CTranslate2 格式
    try:
        print("正在转换为 CTranslate2 格式...")
        from ctranslate2.converters import TransformersConverter

        # 2. 使用 ctranslate2 直接转换
        converter = TransformersConverter(OUTPUT_DIR)
        converter.convert(
            output_dir=OUTPUT_DIR + "/ct2_model",
            quantization="float16",  # 可选: int8, int8_float16
            force=True,
        )
        print(f"ct2 模型已保存到 {(OUTPUT_DIR + "/ct2_model")}")
    except ImportError:
        print("未安装 ctranslate2，无法自动转换为 ct2 格式。请先运行 pip install ctranslate2")

