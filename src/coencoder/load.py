from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import torch

tokenizer = AutoTokenizer.from_pretrained("./co_model", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("./co_model", torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto").to("cuda")

"""
streamer = TextStreamer(
        tokenizer,
        skip_prompt=False,
        skip_special_tokens=False,
)
"""

# プロンプトの準備
prompt = "Q:まどか☆マギカでは誰が一番かわいい？\nA:"

# 推論の実行
token_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
with torch.no_grad():
    output_ids = model.generate(
        token_ids.to(model.device),
        max_new_tokens=64,
        min_new_tokens=64,
        do_sample=True,
        temperature=0.8,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

output = tokenizer.decode(output_ids.tolist()[0])
print(output)

