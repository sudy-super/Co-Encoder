import json

with open("C:\\Users\\rakut.LAPTOP-RNFJPRJD\\Downloads\\oasst2_llm-jp.jsonl", "r", encoding="utf-8") as oasst2_file:
    with open("C:\\Users\\rakut.LAPTOP-RNFJPRJD\\Downloads\\malum_train.json", "r", encoding="utf-8") as malum_file:
        with open("C:\\Users\\rakut.LAPTOP-RNFJPRJD\\Downloads\\cotangent_train.json", "r", encoding="utf-8") as cotangent_file:
            with open("C:\\Users\\rakut.LAPTOP-RNFJPRJD\\Downloads\\rp_train.json", "r", encoding="utf-8") as rp_file:
                with open("C:\\Users\\rakut.LAPTOP-RNFJPRJD\\Downloads\\baobabu_train.json", "r", encoding="utf-8") as baobabu_file:
                    with open("C:\\Users\\rakut.LAPTOP-RNFJPRJD\\Downloads\\aozora_summary_train.json", "r", encoding="utf-8") as as_file:
                        with open("C:\\Users\\rakut.LAPTOP-RNFJPRJD\\Downloads\\aozora_inst_train.json", "r", encoding="utf-8") as ai_file:
                            with open("C:\\Users\\rakut.LAPTOP-RNFJPRJD\\Downloads\\amenokaku_train.json", "r", encoding="utf-8") as amenokaku_file:
                                with open("C:\\Users\\rakut.LAPTOP-RNFJPRJD\\Downloads\\null_train.json", "r", encoding="utf-8") as null_file, open("C:\\Users\\rakut.LAPTOP-RNFJPRJD\\Downloads\\hachi_rlhf_train.json", "r", encoding="utf-8") as hachi_file, open("C:\\Users\\rakut.LAPTOP-RNFJPRJD\\Downloads\\minnade_train.json", "r", encoding="utf-8") as minnade_file, open("C:\\Users\\rakut.LAPTOP-RNFJPRJD\\Downloads\\oasst1_llm-jp.jsonl", "r", encoding="utf-8") as oasst1_file, open("C:\\Users\\rakut.LAPTOP-RNFJPRJD\\Downloads\\oasst2_en.jsonl", "r", encoding="utf-8") as oasst2_en_file:
                                    oasst_list = []
                                    malum_list = []
                                    cotangent_list = []
                                    rp_list = []
                                    baobabu_list = []
                                    as_list = []
                                    ai_list = []
                                    amenokaku_list = []
                                    null_list = []
                                    hachi_list = []
                                    minnade_list = []
                                    for line in oasst2_file:
                                        data = json.loads(line)
                                        conversations = []
                                        for d in data["conversations"]:
                                            if d["role"] == "user":
                                                conversations.append({"from": "human", "value": d["content"]})
                                            elif d["role"] == "assistant":
                                                conversations.append({"from": "gpt", "value": d["content"]})
                                        new_data = {"conversations": conversations}
                                        oasst_list.append(new_data)
                                    
                                    for line in oasst1_file:
                                        data = json.loads(line)
                                        oasst_list.append(data)
                                    
                                    for line in oasst2_en_file:
                                        data = json.loads(line)
                                        conversations = []
                                        for d in data["conversation"]:
                                            if d["role"] == "user":
                                                conversations.append({"from": "human", "value": d["content"]})
                                            elif d["role"] == "assistant":
                                                conversations.append({"from": "gpt", "value": d["content"]})
                                        new_data = {"conversations": conversations}
                                        oasst_list.append(new_data)

                                    data = json.load(malum_file)
                                    malum_list = data
                                    datas = json.load(cotangent_file)
                                    for data in datas:
                                        if data["input"] == "":
                                            new_data = {"conversations": [{"from": "human", "value": data["instruction"]}, {"from": "gpt", "value": data["output"]}]}
                                        else:
                                            new_data = {"conversations": [{"from": "human", "value": (data["input"] + "\n\n" + data["instruction"])}, {"from": "gpt", "value": data["output"]}]}
                                        cotangent_list.append(new_data)
                                    for data in json.load(rp_file):
                                        c_list = []
                                        for d in data["messages"]:
                                            if d["role"] == "system":
                                                c_list.append({"from": "human", "value": d["content"]})
                                            elif d["role"] == "user":
                                                if d["content"] == "":
                                                    continue
                                                c_list.append({"from": "human", "value": d["content"]})
                                            elif d["role"] == "assistant":
                                                c_list.append({"from": "gpt", "value": d["content"]})
                                        new_data = {"conversations": c_list}
                                        rp_list.append(new_data)
                                    for data in json.load(baobabu_file):
                                        if data["input"] == "":
                                            new_data = {"conversations": [{"from": "human", "value": data["question"]}, {"from": "gpt", "value": data["answer"]}]}
                                        else:
                                            new_data = {"conversations": [{"from": "human", "value": (data["input"] + "\n\n" + data["question"])}, {"from": "gpt", "value": data["answer"]}]}
                                        baobabu_list.append(new_data)
                                    c = 0
                                    summary_q = ["この文章を要約してください。", "これを短くまとめてください。", "この文章を簡潔にまとめて", "この文を短く説明してください。", "この文の概要を説明してください。", "この要旨を出力してください。"]
                                    for data in json.load(as_file):
                                        if c % 6 == 0:
                                            question = summary_q[0]
                                        elif c % 6 == 1:
                                            question = summary_q[1]
                                        elif c % 6 == 2:
                                            question = summary_q[2]
                                        elif c % 6 == 3:
                                            question = summary_q[3]
                                        elif c % 6 == 4:
                                            question = summary_q[4]
                                        elif c % 6 == 5:
                                            question = summary_q[5]
                                        new_data = {"conversations": [{"from": "human", "value": (data["input"] + "\n\n" + question)}, {"from": "gpt", "value": data["generated"]}]}
                                        baobabu_list.append(new_data)
                                        c += 1
                                    for data in json.load(ai_file):
                                        if data["input"] == "":
                                            new_data = {"conversations": [{"from": "human", "value": data["instruction"]}, {"from": "gpt", "value": data["response"]}]}
                                        else:
                                            new_data = {"conversations": [{"from": "human", "value": (data["input"] + "\n\n" + data["instruction"])}, {"from": "gpt", "value": data["response"]}]}
                                        ai_list.append(new_data)
                                    for data in json.load(amenokaku_file):
                                        if data["input"] == "":
                                            new_data = {"conversations": [{"from": "human", "value": data["instruction"]}, {"from": "gpt", "value": data["output"]}]}
                                        else:
                                            new_data = {"conversations": [{"from": "human", "value": (data["input"] + "\n\n" + data["instruction"])}, {"from": "gpt", "value": data["output"]}]}
                                        amenokaku_list.append(new_data)
                                    data = json.load(null_file)
                                    null_list = data
                                    data = json.load(hachi_file)
                                    hachi_list = data
                                    data = json.load(minnade_file)
                                    minnade_list = data


with open("C:\\Users\\rakut.LAPTOP-RNFJPRJD\\Downloads\\sft_train.json", "w", encoding="utf-8") as out_file:
    all_list = oasst_list + malum_list + cotangent_list + rp_list + baobabu_list + as_list + ai_list + amenokaku_list + null_list + hachi_list + minnade_list
    print(len(all_list))
    out_file.write(json.dumps(all_list, ensure_ascii=False, indent=4) + "\n")


with open("C:\\Users\\rakut.LAPTOP-RNFJPRJD\\Downloads\\oasst2_val.json", "r", encoding="utf-8") as oasst2_file:
    with open("C:\\Users\\rakut.LAPTOP-RNFJPRJD\\Downloads\\malum_val.json", "r", encoding="utf-8") as malum_file:
        with open("C:\\Users\\rakut.LAPTOP-RNFJPRJD\\Downloads\\cotangent_val.json", "r", encoding="utf-8") as cotangent_file:
            with open("C:\\Users\\rakut.LAPTOP-RNFJPRJD\\Downloads\\rp_val.json", "r", encoding="utf-8") as rp_file:
                with open("C:\\Users\\rakut.LAPTOP-RNFJPRJD\\Downloads\\baobabu_val.json", "r", encoding="utf-8") as baobabu_file:
                    with open("C:\\Users\\rakut.LAPTOP-RNFJPRJD\\Downloads\\aozora_summary_val.json", "r", encoding="utf-8") as as_file:
                        with open("C:\\Users\\rakut.LAPTOP-RNFJPRJD\\Downloads\\aozora_inst_val.json", "r", encoding="utf-8") as ai_file:
                            with open("C:\\Users\\rakut.LAPTOP-RNFJPRJD\\Downloads\\amenokaku_val.json", "r", encoding="utf-8") as amenokaku_file:
                                with open("C:\\Users\\rakut.LAPTOP-RNFJPRJD\\Downloads\\null_val.json", "r", encoding="utf-8") as null_file:
                                    oasst2_list = []
                                    malum_list = []
                                    cotangent_list = []
                                    rp_list = []
                                    baobabu_list = []
                                    as_list = []
                                    ai_list = []
                                    amenokaku_list = []
                                    null_list = []
                                    """
                                    data = json.load(oasst2_file)
                                    oasst2_list = data
                                    """
                                    data = json.load(malum_file)
                                    malum_list = data
                                    datas = json.load(cotangent_file)
                                    for data in datas:
                                        if data["input"] == "":
                                            new_data = {"conversations": [{"from": "human", "value": data["instruction"]}, {"from": "gpt", "value": data["output"]}]}
                                        else:
                                            new_data = {"conversations": [{"from": "human", "value": (data["input"] + "\n\n" + data["instruction"])}, {"from": "gpt", "value": data["output"]}]}
                                        cotangent_list.append(new_data)
                                    for data in json.load(rp_file):
                                        c_list = []
                                        for d in data["messages"]:
                                            if d["role"] == "system":
                                                c_list.append({"from": "human", "value": d["content"]})
                                            elif d["role"] == "user":
                                                if d["content"] == "":
                                                    continue
                                                c_list.append({"from": "human", "value": d["content"]})
                                            elif d["role"] == "assistant":
                                                c_list.append({"from": "gpt", "value": d["content"]})
                                        new_data = {"conversations": c_list}
                                        rp_list.append(new_data)
                                    for data in json.load(baobabu_file):
                                        if data["input"] == "":
                                            new_data = {"conversations": [{"from": "human", "value": data["question"]}, {"from": "gpt", "value": data["answer"]}]}
                                        else:
                                            new_data = {"conversations": [{"from": "human", "value": (data["input"] + "\n\n" + data["question"])}, {"from": "gpt", "value": data["answer"]}]}
                                        baobabu_list.append(new_data)
                                    c = 0
                                    summary_q = ["この文章を要約してください。", "これを短くまとめてください。", "この文章を簡潔にまとめて", "この文を短く説明してください。", "この文の概要を説明してください。", "この要旨を出力してください。"]
                                    for data in json.load(as_file):
                                        if c % 6 == 0:
                                            question = summary_q[0]
                                        elif c % 6 == 1:
                                            question = summary_q[1]
                                        elif c % 6 == 2:
                                            question = summary_q[2]
                                        elif c % 6 == 3:
                                            question = summary_q[3]
                                        elif c % 6 == 4:
                                            question = summary_q[4]
                                        elif c % 6 == 5:
                                            question = summary_q[5]
                                        new_data = {"conversations": [{"from": "human", "value": (data["input"] + "\n\n" + question)}, {"from": "gpt", "value": data["generated"]}]}
                                        baobabu_list.append(new_data)
                                        c += 1
                                    for data in json.load(ai_file):
                                        if data["input"] == "":
                                            new_data = {"conversations": [{"from": "human", "value": data["instruction"]}, {"from": "gpt", "value": data["response"]}]}
                                        else:
                                            new_data = {"conversations": [{"from": "human", "value": (data["input"] + "\n\n" + data["instruction"])}, {"from": "gpt", "value": data["response"]}]}
                                        ai_list.append(new_data)
                                    for data in json.load(amenokaku_file):
                                        if data["input"] == "":
                                            new_data = {"conversations": [{"from": "human", "value": data["instruction"]}, {"from": "gpt", "value": data["output"]}]}
                                        else:
                                            new_data = {"conversations": [{"from": "human", "value": (data["input"] + "\n\n" + data["instruction"])}, {"from": "gpt", "value": data["output"]}]}
                                        amenokaku_list.append(new_data)
                                    data = json.load(null_file)
                                    null_list = data


with open("C:\\Users\\rakut.LAPTOP-RNFJPRJD\\Downloads\\sft_val.json", "w", encoding="utf-8") as out_file:
    all_list = malum_list + cotangent_list + rp_list + baobabu_list + as_list + ai_list + amenokaku_list + null_list
    print(len(all_list))
    out_file.write(json.dumps(all_list, ensure_ascii=False) + "\n")

"""
all_list2 = malum_list + cotangent_list

with open("C:\\Users\\rakut.LAPTOP-RNFJPRJD\\Downloads\\sft_high.json", "w", encoding="utf-8") as high_file:
    all_list = all_list1 + all_list2
    print(len(all_list))
    high_file.write(json.dumps(all_list, ensure_ascii=False, indent=4) + "\n")
"""