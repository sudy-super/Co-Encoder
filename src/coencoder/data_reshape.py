# booksum_en: chapter, instruction, output
# result_en: chapter, instruction, output
# result_jp: chapter, instruction, output
# baobabu: input, question, answer

"""
{
    "context": <context>,
    "conversations": [
        {"from": "user", "value": <value>},
        {"from": "assistant", "value": <value>}
    ]
}
"""

import json
import random

instruction_list = [
    "この文を短くまとめてください。",
    "この情報を手短に説明してください。",
    "上記の内容について簡潔に教えてください。",
    "これの要点だけ教えてもらえますか？",
    "このコンテキストを簡単に説明していただけると助かります。",
    "これを要約していただけますか？",
    "コンパクトにお願いします。",
    "この情報についてサクッと教えて",
    "上記の文章を短くしてください。",
]

def process_and_write(file_path, context_key, instruction_key=None, output_key=None, random_instruction=False, count=0, first_flags=None):
    global train_file, val_file, test_file

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line)
            if random_instruction:
                instruction = random.choice(instruction_list)
            elif instruction_key:
                instruction = data[instruction_key]
            else:
                instruction = random.choice(instruction_list)
            new_data = {
                "context": data[context_key],
                "instruction": instruction,
                "output": data[output_key]
            }
            final_data = {
                "context": new_data["context"],
                "conversations": [
                    {"from": "user", "value": new_data["instruction"]},
                    {"from": "assistant", "value": new_data["output"]}
                ]
            }

            # データの分割とファイルへの書き込み
            if count % 330 == 0:
                if not first_flags['test']:
                    test_file.write(",\n")
                else:
                    first_flags['test'] = False
                test_file.write(json.dumps(final_data, ensure_ascii=False, indent=4))
            elif count % 20 == 0:
                if not first_flags['val']:
                    val_file.write(",\n")
                else:
                    first_flags['val'] = False
                val_file.write(json.dumps(final_data, ensure_ascii=False, indent=4))
            else:
                if not first_flags['train']:
                    train_file.write(",\n")
                else:
                    first_flags['train'] = False
                train_file.write(json.dumps(final_data, ensure_ascii=False, indent=4))
            count += 1
    return count, first_flags

# 出力ファイルを開き、'['を書き込む
with open("data/train.json", "w", encoding="utf-8") as train_file, \
     open("data/val.json", "w", encoding="utf-8") as val_file, \
     open("data/test.json", "w", encoding="utf-8") as test_file:

    train_file.write("[\n")
    val_file.write("[\n")
    test_file.write("[\n")

    count = 0
    first_flags = {'train': True, 'val': True, 'test': True}

    # 各データファイルの処理
    count, first_flags = process_and_write("data/booksum_en.jsonl", "chapter", "instruction", "output", count=count, first_flags=first_flags)
    count, first_flags = process_and_write("data/result_en.jsonl", "chapter", "instruction", "output", count=count, first_flags=first_flags)

    # result_jp.jsonlの処理（エラーハンドリング付き）
    with open("data/result_jp.jsonl", "r", encoding="utf-8") as result_jp_file:
        for idx, line in enumerate(result_jp_file):
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print(f"JSONデコードエラーが発生しました。行番号: {idx + 1}")
                continue
            new_data = {
                "context": data["chapter"],
                "instruction": data["instruction"],
                "output": data["output"]
            }
            final_data = {
                "context": new_data["context"],
                "conversations": [
                    {"from": "user", "value": new_data["instruction"]},
                    {"from": "assistant", "value": new_data["output"]}
                ]
            }
            # データの分割とファイルへの書き込み
            if count % 330 == 0:
                if not first_flags['test']:
                    test_file.write(",\n")
                else:
                    first_flags['test'] = False
                test_file.write(json.dumps(final_data, ensure_ascii=False, indent=4))
            elif count % 20 == 0:
                if not first_flags['val']:
                    val_file.write(",\n")
                else:
                    first_flags['val'] = False
                val_file.write(json.dumps(final_data, ensure_ascii=False, indent=4))
            else:
                if not first_flags['train']:
                    train_file.write(",\n")
                else:
                    first_flags['train'] = False
                train_file.write(json.dumps(final_data, ensure_ascii=False, indent=4))
            count += 1

    # baobabu.jsonlの処理
    count, first_flags = process_and_write("data/baobabu.jsonl", "input", "question", "answer", count=count, first_flags=first_flags)
    count, first_flags = process_and_write("data/aozora_inst.jsonl", "input", "instruction", "response", count=count, first_flags=first_flags)
    count, first_flags = process_and_write("data/aozora_summary.jsonl", "input", output_key="generated", random_instruction=True, count=count, first_flags=first_flags)
    count, first_flags = process_and_write("data/aozora_summary1.jsonl", "input", output_key="generated", random_instruction=True, count=count, first_flags=first_flags)

    # 出力ファイルに']'を書き込む
    train_file.write("\n]")
    val_file.write("\n]")
    test_file.write("\n]")

    print(f"処理した合計アイテム数: {count}")