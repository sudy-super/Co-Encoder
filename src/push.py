from model.modeling_co_encoder import CoEncoderForConditionalGeneration
from model.configuration_co_encoder import CoEncoderConfig
from model.tokenization_co_encoder import CoEncoderDualTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

CoEncoderConfig.register_for_auto_class("AutoConfig")
CoEncoderForConditionalGeneration.register_for_auto_class("AutoModelForCausalLM")
# CoEncoderDualTokenizer.register_for_auto_class("AutoTokenizer")


# tokenizer = AutoTokenizer.from_pretrained("./co_model", trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained("./co_model", torch_dtype=torch.bfloat16, trust_remote_code=True)


model.push_to_hub("coencoder_test2")

# tokenizer.push_to_hub("coencoder_test2")

