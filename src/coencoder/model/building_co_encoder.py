# coding=utf-8
"""CoEncoder model builder"""

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers.utils import is_flash_attn_2_available

from .modeling_co_encoder import (
    CoEncoderForConditionalGeneration, 
    CoEncoderConfig, 
    CoEncoderContextLanguageConnector,
    CoEncoderContextTower
)


import torch
import os

class CoEncoderTokenizerBuilder:
    """
    A class to build and save a CoEncoder tokenizer from separate LLM modules.
    """

    def __init__(self, context_model_name, text_model_name, output_path, auth_token=None):
        """
        Initialize the CoEncoderTokenizerBuilder.

        Args:
            context_model_name (str): The name or path of the context LLM.
            text_model_name (str): The name or path of the text LLM.
            output_path (str): The path to save the CoEncoder tokenizer.
        """
        self.context_model_name = context_model_name
        self.text_model_name = text_model_name
        self.output_path = output_path
        self.auth_token = auth_token
    
    def build_and_save_tokenizer(self):
        """
        Build the CoEncoder tokenizer from separate LLMs and save it.
        """
        # Load the separate models
        context_tokenizer = AutoTokenizer.from_pretrained(
            self.context_model_name, 
            use_fast=True,
            use_auth_token=self.auth_token if self.auth_token is not None else None
        )
        text_tokenizer = AutoTokenizer.from_pretrained(
            self.text_model_name, 
            use_fast=True,
            use_auth_token=self.auth_token if self.auth_token is not None else None
        )

        context_tokenizer.save_pretrained(self.output_path + "/context_tokenizer")
        text_tokenizer.save_pretrained(self.output_path + "/text_tokenizer")

        print(f"CoEncoder tokenizer saved to {self.output_path}")

class CoEncoderModelBuilder:
    """
    A class to build and save a CoEncoder model from separate LLM modules.
    """

    def __init__(self, context_model_name, text_model_name, output_path, auth_token=None):
        """
        Initialize the CoEncoderModelBuilder.

        Args:
            context_model_name (str): The name or path of the context LLM.
            text_model_name (str): The name or path of the text LLM.
            output_path (str): The path to save the combined CoEncoder model.
        """
        self.context_model_name = context_model_name
        self.text_model_name = text_model_name
        self.output_path = output_path
        self.auth_token = auth_token

    def build_and_save_model(
        self,
        ignore_index=-100,
        projector_hidden_act="gelu",
        context_feature_layer=-2,
        context_feature_select_strategy="default",
        begin_of_context_token_id=None,
        end_of_context_token_id=None
    ):
        """
        Build the CoEncoder model from separate LLMs and save it.
        """
        # Load the separate models
        context_model = AutoModelForCausalLM.from_pretrained(
            self.context_model_name, 
            attn_implementation=(
                "flash_attention_2" if is_flash_attn_2_available() else "eager"
            ),
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
            use_auth_token=self.auth_token if self.auth_token is not None else None
        )
        text_model = AutoModelForCausalLM.from_pretrained(
            self.text_model_name, 
            attn_implementation=(
                "flash_attention_2" if is_flash_attn_2_available() else "eager"
            ),
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
            use_auth_token=self.auth_token if self.auth_token is not None else None
        )

        # Create CoEncoder config
        config = CoEncoderConfig(
            context_config=context_model.config,
            text_config=text_model.config,
            ignore_index=ignore_index,
            projector_hidden_act=projector_hidden_act,
            context_feature_layer=context_feature_layer,
            context_feature_select_strategy=context_feature_select_strategy,
            begin_of_context_token_id=begin_of_context_token_id,
            end_of_context_token_id=end_of_context_token_id,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16
        )

        # Initialize CoEncoder model
        co_encoder_model = CoEncoderForConditionalGeneration(config)

        # Load state dict for context tower
        co_encoder_model.context_tower.tower.load_state_dict(context_model.state_dict())

        # Load state dict for language model
        co_encoder_model.language_model.load_state_dict(text_model.state_dict())

        # The connector is already initialized in the CoEncoderForConditionalGeneration constructor

        # Save the combined model
        co_encoder_model.save_pretrained(self.output_path, max_shard_size="10GB")
        # config.save_pretrained(self.output_path)

        print(f"CoEncoder model saved to {self.output_path}")

    @classmethod
    def from_pretrained(cls, model_path):
        """
        Load a pre-built CoEncoder model.

        Args:
            model_path (str): Path to the saved CoEncoder model.

        Returns:
            CoEncoderForConditionalGeneration: The loaded CoEncoder model.
        """
        config = CoEncoderConfig.from_pretrained(model_path)
        model = CoEncoderForConditionalGeneration.from_pretrained(model_path, config=config)
        return model

# Usage example:
# builder = CoEncoderModelBuilder("bert-base-uncased", "gpt2", "./co_encoder_model")
# builder.build_and_save_model()

# To load the saved model:
# loaded_model = CoEncoderModelBuilder.from_pretrained("./co_encoder_model")

# tokenizer_builder = CoEncoderTokenizerBuilder("bert-base-uncased", "gpt2", "./co_encoder_model")
# tokenizer_builder.build_and_save_tokenizer()