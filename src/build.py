from coencoder_src.building_co_encoder import CoEncoderModelBuilder, CoEncoderTokenizerBuilder

import torch

builder = CoEncoderModelBuilder(
            context_model_name="Qwen/Qwen2-0.5B",
            text_model_name="meta-llama/Llama-3.1-8B-Instruct",
            output_path="./co_model",
            auth_token="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"
)

"""
tokenizer_builder = CoEncoderTokenizerBuilder(
            context_model_name="Qwen/Qwen2-0.5B",
            text_model_name="meta-llama/Llama-3.1-8B-Instruct",
            output_path="./co_model",
            auth_token="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"
)
"""

builder.build_and_save_model(
    begin_of_context_token_id=128002,
    end_of_context_token_id=128003
)
"""
tokenizer_builder.build_and_save_tokenizer()
"""