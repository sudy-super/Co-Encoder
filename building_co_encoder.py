# coding=utf-8
"""CoEncoder model builder"""

from transformers import AutoModel, AutoModelForCausalLM, AutoConfig
from .modeling_co_encoder import (
    CoEncoderForConditionalGeneration, 
    CoEncoderConfig, 
    CoEncoderMultiModalProjector,
    CoEncoderContextTower
)
import torch
import os

class CoEncoderModelBuilder:
    """
    A class to build and save a CoEncoder model from separate LLM modules.
    """

    def __init__(self, context_model_name, text_model_name, output_path):
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

    def build_and_save_model(self):
        """
        Build the CoEncoder model from separate LLMs and save it.
        """
        # Load the separate models
        context_model = AutoModel.from_pretrained(self.context_model_name)
        text_model = AutoModelForCausalLM.from_pretrained(self.text_model_name)

        # Create CoEncoder config
        config = CoEncoderConfig(
            context_config=context_model.config,
            text_config=text_model.config
        )

        # Initialize CoEncoder model
        co_encoder_model = CoEncoderForConditionalGeneration(config)

        # Load state dict for context tower
        co_encoder_model.context_tower.llm.load_state_dict(context_model.state_dict())

        # Load state dict for language model
        co_encoder_model.language_model.load_state_dict(text_model.state_dict())

        # The multi_modal_projector is already initialized in the CoEncoderForConditionalGeneration constructor

        # Save the combined model
        co_encoder_model.save_pretrained(self.output_path)
        config.save_pretrained(self.output_path)

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