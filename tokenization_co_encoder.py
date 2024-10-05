# coding=utf-8
"""Tokenization classes for CoEncoder"""

from typing import List, Union, Optional
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.utils import logging
from transformers.feature_extraction_utils import BatchFeature

logger = logging.get_logger(__name__)

class CoEncoderDualTokenizer(ProcessorMixin):
    r"""
    CoEncoderDualTokenizer is tokenizer for the CoEncoder model. It processes context and main text.

    Args:
        context_tokenizer ([`PreTrainedTokenizer`]):
            The tokenizer for context.
        text_tokenizer ([`PreTrainedTokenizer`]):
            The tokenizer for main text.
    """

    attributes = ["context_tokenizer", "text_tokenizer"]
    context_tokenizer_class = "AutoTokenizer"
    text_tokenizer_class = "AutoTokenizer"

    def __init__(self, context_tokenizer=None, text_tokenizer=None):
        super().__init__(context_tokenizer, text_tokenizer)

    def __call__(
        self,
        context: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        return_tensors: Optional[str] = None,
        **kwargs
    ) -> BatchFeature:
        """
        Main method to prepare inputs for the CoEncoder model.

        Args:
            context: Context text input.
            text: Main text input.
            return_tensors: Type of tensors to return.

        Returns:
            BatchFeature: A BatchFeature object containing model inputs.
        """
        if context is None and text is None:
            raise ValueError("You must provide either context or text.")

        features = {}

        if context is not None:
            context_features = self.context_tokenizer(
                context,
                return_tensors=return_tensors,
                **kwargs.get("context_kwargs", {})
            )
            features.update({f"context_{k}": v for k, v in context_features.items()})

        if text is not None:
            text_features = self.text_tokenizer(
                text,
                return_tensors=return_tensors,
                **kwargs.get("text_kwargs", {})
            )
            features.update({f"text_{k}": v for k, v in text_features.items()})

        return BatchFeature(data=features, tensor_type=return_tensors)

    def batch_decode(self, *args, **kwargs):
        """
        Calls the batch_decode method of the text_tokenizer.
        """
        return self.text_tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        Calls the decode method of the text_tokenizer.
        """
        return self.text_tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        """
        Returns the model input names.
        """
        return list(dict.fromkeys(self.context_tokenizer.model_input_names + self.text_tokenizer.model_input_names))