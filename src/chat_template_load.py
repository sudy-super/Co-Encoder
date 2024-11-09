from  transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", token="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")

messages = [
    # {"role": "system", "content": "You are an honest and talented Japanese assistant."},
    {"role": "user", "content": "Who are you?"},
    {"role": "assistant", "content": "I am Japanese assistant."},
    {"role": "user", "content": "What can you do?"}
]

print(tokenizer.apply_chat_template(messages, tokenize=False))
