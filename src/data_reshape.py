import json


with open("C:\\Users\\rakut.LAPTOP-RNFJPRJD\\Downloads\\oasst2.jsonl", "r", encoding="utf-8") as in_file:
    train_list = []
    val_list = []
    count = 0
    for line in in_file:
        data = json.loads(line)
        if count % 200 == 0:
            val_list.append(data)
        else:
            train_list.append(data)
        count += 1

with open("C:\\Users\\rakut.LAPTOP-RNFJPRJD\\Downloads\\oasst2_train.json", "w", encoding="utf-8") as out_file:
    out_file.write(json.dumps(train_list, ensure_ascii=False, indent=4) + "\n")

with open("C:\\Users\\rakut.LAPTOP-RNFJPRJD\\Downloads\\oasst2_val.json", "w", encoding="utf-8") as out_file:
    out_file.write(json.dumps(val_list, ensure_ascii=False, indent=4) + "\n")



with open("C:\\Users\\rakut.LAPTOP-RNFJPRJD\\Downloads\\rose_rp_v2.jsonl", "r", encoding="utf-8") as in_file:
    train_list = []
    val_list = []
    count = 0
    for line in in_file:
        data = json.loads(line)
        if count % 100 == 0:
            val_list.append(data)
        else:
            train_list.append(data)
        count += 1

with open("C:\\Users\\rakut.LAPTOP-RNFJPRJD\\Downloads\\rp_train.json", "w", encoding="utf-8") as out_file:
    out_file.write(json.dumps(train_list, ensure_ascii=False, indent=4) + "\n")

with open("C:\\Users\\rakut.LAPTOP-RNFJPRJD\\Downloads\\rp_val.json", "w", encoding="utf-8") as out_file:
    out_file.write(json.dumps(val_list, ensure_ascii=False, indent=4) + "\n")


with open("C:\\Users\\rakut.LAPTOP-RNFJPRJD\\Downloads\\aozora_summary.jsonl", "r", encoding="utf-8") as in_file:
    train_list = []
    val_list = []
    count = 0
    for line in in_file:
        data = json.loads(line)
        if count % 100 == 0:
            val_list.append(data)
        else:
            train_list.append(data)
        count += 1

with open("C:\\Users\\rakut.LAPTOP-RNFJPRJD\\Downloads\\aozora_summary_train.json", "w", encoding="utf-8") as out_file:
    out_file.write(json.dumps(train_list, ensure_ascii=False, indent=4) + "\n")

with open("C:\\Users\\rakut.LAPTOP-RNFJPRJD\\Downloads\\aozora_summary_val.json", "w", encoding="utf-8") as out_file:
    out_file.write(json.dumps(val_list, ensure_ascii=False, indent=4) + "\n")


with open("C:\\Users\\rakut.LAPTOP-RNFJPRJD\\Downloads\\aozora_inst.jsonl", "r", encoding="utf-8") as in_file:
    train_list = []
    val_list = []
    count = 0
    for line in in_file:
        data = json.loads(line)
        if count % 100 == 0:
            val_list.append(data)
        else:
            train_list.append(data)
        count += 1

with open("C:\\Users\\rakut.LAPTOP-RNFJPRJD\\Downloads\\aozora_inst_train.json", "w", encoding="utf-8") as out_file:
    out_file.write(json.dumps(train_list, ensure_ascii=False, indent=4) + "\n")

with open("C:\\Users\\rakut.LAPTOP-RNFJPRJD\\Downloads\\raozora_inst_val.json", "w", encoding="utf-8") as out_file:
    out_file.write(json.dumps(val_list, ensure_ascii=False, indent=4) + "\n")


with open("C:\\Users\\rakut.LAPTOP-RNFJPRJD\\Downloads\\wikipedia-human-retrieval-ja.jsonl", "r", encoding="utf-8") as in_file:
    train_list = []
    val_list = []
    count = 0
    for line in in_file:
        data = json.loads(line)
        if count % 100 == 0:
            val_list.append(data)
        else:
            train_list.append(data)
        count += 1

with open("C:\\Users\\rakut.LAPTOP-RNFJPRJD\\Downloads\\baobabu_train.json", "w", encoding="utf-8") as out_file:
    out_file.write(json.dumps(train_list, ensure_ascii=False, indent=4) + "\n")

with open("C:\\Users\\rakut.LAPTOP-RNFJPRJD\\Downloads\\baobabu_val.json", "w", encoding="utf-8") as out_file:
    out_file.write(json.dumps(val_list, ensure_ascii=False, indent=4) + "\n")


with open("C:\\Users\\rakut.LAPTOP-RNFJPRJD\\Downloads\\amenokaku.json", "r", encoding="utf-8") as in_file:
    train_list = []
    val_list = []
    lines = json.load(in_file)
    count = 0
    c = 0
    for line in lines:
        if count % 1 == 0:
            data = line
            if c % 100 == 0:
                val_list.append(data)
            else:
                train_list.append(data)
            c += 1
        count += 1

with open("C:\\Users\\rakut.LAPTOP-RNFJPRJD\\Downloads\\amenokaku_train.json", "w", encoding="utf-8") as out_file:
    out_file.write(json.dumps(train_list, ensure_ascii=False, indent=4) + "\n")

with open("C:\\Users\\rakut.LAPTOP-RNFJPRJD\\Downloads\\amenokaku_val.json", "w", encoding="utf-8") as out_file:
    out_file.write(json.dumps(val_list, ensure_ascii=False, indent=4) + "\n")


with open("C:\\Users\\rakut.LAPTOP-RNFJPRJD\\Downloads\\null_instruct.json", "r", encoding="utf-8") as in_file:
    train_list = []
    val_list = []
    count = 0
    datas = json.load(in_file)
    for line in datas:
        data = line
        if count % 100 == 0:
            val_list.append(data)
        else:
            train_list.append(data)
        count += 1

with open("C:\\Users\\rakut.LAPTOP-RNFJPRJD\\Downloads\\null_train.json", "w", encoding="utf-8") as out_file:
    out_file.write(json.dumps(train_list, ensure_ascii=False, indent=4) + "\n")

with open("C:\\Users\\rakut.LAPTOP-RNFJPRJD\\Downloads\\null_val.json", "w", encoding="utf-8") as out_file:
    out_file.write(json.dumps(val_list, ensure_ascii=False, indent=4) + "\n")
