# coding=utf-8
"""CoEncoder model configuration"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers import CONFIG_MAPPING

logger = logging.get_logger(__name__)


class CoEncoderConfig(PretrainedConfig):
    r"""
    """

    model_type = "co_encoder"

    def __init__(
        self,
        context_config=None,
        text_config=None,
        ignore_index=-100,
        projector_hidden_act="gelu",
        context_feature_layer=-2,
        context_feature_select_strategy="default",
        tie_word_embeddings=False,
        **kwargs,
    ):
        self.ignore_index = ignore_index
        self.projector_hidden_act = projector_hidden_act
        self.context_feature_select_strategy = context_feature_select_strategy

        if context_feature_select_strategy not in ["default"]:
            raise ValueError(
                "context_feature_select_strategy should be one of 'default'."
                f"Got: {context_feature_select_strategy}"
            )
        
        if isinstance(context_config, dict):
            context_config["model_type"] = (
                context_config["model_type"] if "model_type" in context_config else "qwen2"
            )
            context_config = CONFIG_MAPPING[context_config["model_type"]](**context_config)
        
        self.context_config = context_config

        if isinstance(text_config, dict):
            text_config["model_type"] = text_config["model_type"] if "model_type" in text_config else "llama"
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            text_config = CONFIG_MAPPING["llama"]()

        self.text_config = text_config

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)