import random
from torch.utils.data import Dataset, Subset
from transformers import Trainer, TrainingArguments
import torch
from datasets import load_dataset
import wandb

# CoEncoderモデルとトークナイザーのインポート
from transformers import AutoTokenizer, AutoModelForCausalLM
from coencoder_src.modeling_co_encoder_old import CoEncoderForConditionalGeneration
from coencoder_src.tokenization_co_encoder import CoEncoderDualTokenizer

from accelerate import infer_auto_device_map, dispatch_model
import os

# WandBの初期化（メインプロセスでのみ）
if int(os.environ.get("LOCAL_RANK", 0)) == 0:
    wandb.init(project="coencoder_finetune", name="1e-3_coencoder_connector", entity="sudy_super")

torch.manual_seed(42)

model_name = "sudy-super/coencoder_test2"

# CoEncoderトークナイザーとモデルの読み込み
tokenizer = CoEncoderDualTokenizer.from_pretrained("co_model", trust_remote_code=True)
model = CoEncoderForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation="flash_attention_2"
)

model.model_parallel = True

tokenizer.text_tokenizer.pad_token = tokenizer.text_tokenizer.eos_token

model.gradient_checkpointing_enable()


# context_towerとlanguage_modelの重みを凍結
for param in model.context_tower.parameters():
    param.requires_grad = False

for param in model.language_model.parameters():
    param.requires_grad = False


# データセットの読み込み
dataset = load_dataset("sudy-super/coencoder_data")

# データセットの取得
train_data = dataset["train"]
val_data = dataset["validation"]
test_data = dataset["test"]

# `generate_inputs`関数をバッチ処理に対応
def generate_inputs(batch):
    contexts = []
    texts = []
    for context, conversations in zip(batch.get("context", []), batch["conversations"]):
        if not context:
            context = ""  # contextがNoneまたは空の場合、空文字列に設定
        text = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 9 Nov 2024

<|eot_id|>"""
        for c in conversations:
            if c["from"] == "user":
                text += f"""<|start_header_id|>user<|end_header_id|>

{c['value']}<|eot_id|>"""
            elif c["from"] == "assistant":
                text += f"""<|start_header_id|>assistant<|end_header_id|>

{c['value']}<|eot_id|>"""
        contexts.append(context)
        texts.append(text)
    return {'context': contexts, 'text': texts}

# `tokenize`関数をバッチ処理に対応
def tokenize(batch):
    # 最大トークン数の設定
    max_context_tokens = 65536

    # 各サンプルに対してcontextのトークン数を確認し、必要に応じてカット
    truncated_contexts = []
    for context in batch['context']:
        # contextを単独でトークン化してトークン数を確認
        context_tokens = tokenizer.context_tokenizer.tokenize(context)
        if len(context_tokens) > max_context_tokens:
            # トークン数が65536を超える場合、カット
            context = tokenizer.context_tokenizer.convert_tokens_to_string(context_tokens[:max_context_tokens])
        truncated_contexts.append(context)

    # contextをカットしたリストを用いて最終的にトークン化
    tokenized_outputs = tokenizer(
        context=truncated_contexts,
        text=batch['text'],
        truncation=True,
        max_length=131072,
        padding=False,
    )

    tokenized_outputs['length'] = [len(ids) for ids in tokenized_outputs['input_ids']]

    return tokenized_outputs

def data_collator(features):
    # context部分のトークンをパディング
    context_features = [{
        'input_ids': f['context_input_ids'],
        'attention_mask': f.get('context_attention_mask', [1] * len(f['context_input_ids']))
    } for f in features]
    context_batch = tokenizer.context_tokenizer.pad(
        context_features,
        padding=True,
        max_length=None,
        return_tensors="pt"
    )
    # text部分のトークンをパディング
    text_features = [{
        'input_ids': f['input_ids'],
        'attention_mask': f['attention_mask']
    } for f in features]
    text_batch = tokenizer.text_tokenizer.pad(
        text_features,
        padding=True,
        max_length=None,
        return_tensors="pt"
    )
    # ラベルのパディング（input_idsと同じ）
    label_features = [{'input_ids': f['input_ids']} for f in features]
    labels_batch = tokenizer.text_tokenizer.pad(
        label_features,
        padding=True,
        max_length=None,
        return_tensors="pt"
    )
    # パディングされたバッチを統合
    batch = {
        'context_input_ids': context_batch['input_ids'],
        'context_attention_mask': context_batch['attention_mask'],
        'input_ids': text_batch['input_ids'],
        'attention_mask': text_batch['attention_mask'],
        'labels': labels_batch['input_ids']
    }
    return batch

# データのシャッフルとフィルタリング、バッチ処理対応
train_data = train_data.shuffle(seed=42)
val_data = val_data.shuffle(seed=42)
test_data = test_data.shuffle(seed=42)

# データの前処理（キャッシュファイル名を削除）
train_data = train_data.map(
    generate_inputs,
    batched=True,
    num_proc=8,
    desc="Generating inputs for train data",
    load_from_cache_file=True
).filter(lambda x: x['text'] != '', num_proc=4).filter(lambda x: x['context'] != '', num_proc=4)

val_data = val_data.map(
    generate_inputs,
    batched=True,
    num_proc=8,
    desc="Generating inputs for validation data",
    load_from_cache_file=True
).filter(lambda x: x['text'] != '', num_proc=4).filter(lambda x: x['context'] != '', num_proc=4)

test_data = test_data.map(
    generate_inputs,
    batched=True,
    num_proc=8,
    desc="Generating inputs for test data",
    load_from_cache_file=True
).filter(lambda x: x['text'] != '', num_proc=4).filter(lambda x: x['context'] != '', num_proc=4)

# データのトークン化（キャッシュファイル名を削除）
train_data = train_data.map(
    tokenize,
    batched=True,
    num_proc=8,
    remove_columns=train_data.column_names,
    desc="Tokenizing train data",
    load_from_cache_file=True
)
val_data = val_data.map(
    tokenize,
    batched=True,
    num_proc=8,
    remove_columns=val_data.column_names,
    desc="Tokenizing validation data",
    load_from_cache_file=True
)
test_data = test_data.map(
    tokenize,
    batched=True,
    num_proc=8,
    remove_columns=test_data.column_names,
    desc="Tokenizing test data",
    load_from_cache_file=True
)

from datasets import concatenate_datasets, Dataset

def move_random_samples(eval_dataset, train_dataset, num_samples=4500):
    # 評価データセットのインデックスを取得
    eval_indices = list(range(len(eval_dataset)))
    # ランダムにインデックスをサンプリング
    random.seed(42)
    selected_indices = random.sample(eval_indices, num_samples)

    # サブセットを作成
    selected_subset = eval_dataset.select(selected_indices)
    remaining_eval_subset = eval_dataset.select([i for i in eval_indices if i not in selected_indices])

    # SubsetをDatasetオブジェクトに変換
    selected_subset = Dataset.from_dict(selected_subset.to_dict())

    # concatenate_datasetsでサブセットを結合
    train_dataset = concatenate_datasets([train_dataset, selected_subset])

    return train_dataset, remaining_eval_subset



# 評価データセットから4500件をトレーニングデータセットに移す
train_data, eval_data = move_random_samples(val_data, train_data, num_samples=4500)

# サンプルの長さに基づいてデータをソートし、バッチを形成するためのカスタムサンプラー
from torch.utils.data import Sampler
import numpy as np

class GroupedLengthSampler(Sampler):
    def __init__(self, lengths, batch_size, shuffle=True):
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle

        # インデックスと長さを取得
        self.indices_lengths = list(enumerate(self.lengths))

        # 長さに基づいてソート
        self.indices_lengths.sort(key=lambda x: x[1])

        # バッチを形成
        self.batches = [self.indices_lengths[i:i + self.batch_size] for i in range(0, len(self.indices_lengths), self.batch_size)]

        if self.shuffle:
            random.shuffle(self.batches)

        # フラットなインデックスリストを作成
        self.indices = [idx for batch in self.batches for idx, _ in batch]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)



"""
# 最初のバッチのトークン数を出力
first_batch = train_data[:1]
for i in range(len(first_batch)):
    context_tokens_count = len(first_batch['context_input_ids'][i])
    text_tokens_count = len(first_batch['input_ids'][i])
    print(f"Context tokens count: {context_tokens_count}")
    print(f"Text tokens count: {text_tokens_count}")
"""

from torch.utils.data import DataLoader

class CustomTrainer(Trainer):
    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        # サンプルの長さを取得
        lengths = self.train_dataset["length"]

        # カスタムサンプラーを作成
        sampler = GroupedLengthSampler(
            lengths=lengths,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True
        )

        # データローダーを作成
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

# トレーニング引数の設定
args = TrainingArguments(
    num_train_epochs=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=256,
    learning_rate=1e-3,
    adam_beta2=0.95,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=1,
    save_strategy="steps",
    eval_steps=135,
    save_steps=135,
    output_dir="output",
    report_to="wandb",
    save_total_limit=3,
    push_to_hub=False,
    seed=42,
    bf16=True,  # bf16を有効化
    bf16_full_eval=True,
    deepspeed="ds_config.json",  # DeepSpeed設定ファイルの指定
    gradient_checkpointing=True,
    optim="adamw_torch_fused",
    dataloader_pin_memory=True,
    dataloader_num_workers=4,
)

# Trainerの設定
trainer = CustomTrainer(
    model=model,
    args=args,
    train_dataset=train_data,
    eval_dataset=eval_data,
    data_collator=data_collator,
)

# トレーニング開始
trainer.train()

# 学習済みモデルの保存
model.save_pretrained("co_output_model")

# テストデータでの評価または予測
test_results = trainer.predict(test_dataset=test_data)
predictions = test_results.predictions

# 必要に応じて予測結果を保存
decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
with open("test_predictions.txt", "w", encoding="utf-8") as f:
    for pred in decoded_preds:
        f.write(pred + "\n")
