import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

tokenizer = AutoTokenizer.from_pretrained("/malum/model_1", use_fast=False, legacy=False)

model = AutoModelForCausalLM.from_pretrained("/malum/model_1", torch_dtype=torch.bfloat16,) # device_map="auto")
model.eval()

streamer = TextStreamer(
    tokenizer,
    skip_prompt=False,
    skip_special_tokens=False,
)

initial = 0
while True:
    prompt = input("入力:")
    print("--------")
    if initial == 0:
        prompt = f"<s>[INST] <<SYS>>\nあなたは誠実で優秀な日本人のアシスタントです。\n<</SYS>>\n\n{prompt} [/INST] "
        initial = 1
    else:
        prompt = prompt + "<s>[INST] " + prompt + " [/INST] "

    token_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")

    with torch.no_grad():
        model.to("cuda")
        output_ids = model.generate(
            token_ids.to("cuda"),
            max_new_tokens=500,
            # min_new_tokens=500,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            top_k=40,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            streamer=streamer,
        )