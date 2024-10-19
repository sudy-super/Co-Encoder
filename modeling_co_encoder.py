# coding=utf-8
"""PyTorch CoEncoder model."""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn

from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from transformers.image_processing_utils import select_best_resolution
from transformers.modeling_outputs import ModelOutput
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from .configuration_co_encoder import CoEncoderConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "CoEncoderConfig"


@dataclass
class CoEncoderCausalLMOutputWithPast(ModelOutput):
    """
    Base class for CoEncoder causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.context_config.num_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        context_hidden_states (`torch.FloatTensor`, *optional*):
            A `torch.FloatTensor` of size (batch_size, sequence_length, hidden_size)`.
            context_hidden_states of the model produced by the context encoder and after projecting the last hidden state.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    context_hidden_states: Optional[torch.FloatTensor] = None


class CoEncoderMultiModalProjector(nn.Module):
    def __init__(self, config: CoEncoderConfig):
        super().__init__()

        self.linear_1 = nn.Linear(config.context_config.hidden_size, config.text_config.hidden_size, bias=True)
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size, bias=True)

    def forward(self, context_features):
        hidden_states = self.linear_1(context_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class CoEncoderContextTower(nn.Module):
    def __init__(self, config: CoEncoderConfig):
        super().__init__()

        self.llm = AutoModelForCausalLM.from_config(
            config.context_config
        )
        self.select_layer = config.context_feature_layer
    
    def feature_select(self, llm_outputs):
        hidden_states = llm_outputs.hidden_states
        return hidden_states[self.select_layer]

    def forward(self, inputs):
        outputs = self.llm(inputs, output_hidden_states=True)
        features = self.feature_select(outputs)

        return features
    

class CoEncoderPreTrainedModel(PreTrainedModel):
    config_class = CoEncoderConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["CoEncoderMultiModalProjector", "CoEncoderContextTower"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True

    def _init_weights(self, module):
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        )
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class CoEncoderForConditionalGeneration(CoEncoderPreTrainedModel):
    def __init__(self, config: CoEncoderConfig):
        super().__init__(config)
        self.context_tower = CoEncoderContextTower(config)
        self.multi_modal_projector = CoEncoderMultiModalProjector(config)

        self.language_model = AutoModelForCausalLM.from_config(
            config.text_config
        )

        self.vocab_size = config.text_config.vocab_size
        self.ignore_index = config.ignore_index if hasattr(config, 'ignore_index') else -100
        self.begin_of_context_token_id = config.begin_of_context_token_id
        self.end_of_context_token_id = config.end_of_context_token_id
        
        self.post_init()
    
    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()
    
    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()
    
    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)
    
    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)
    
    def get_decoder(self):
        return self.language_model.get_decoder()
    
    def tie_weights(self):
        return self.language_model.tie_weights()
    
    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None, pad_to_multiple_of=None) -> nn.Embedding:
        model_embeds = self.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        # update vocab size
        self.config.text_config.vocab_size = model_embeds.num_embeddings
        self.vocab_size = model_embeds.num_embeddings
        return model_embeds
    
    def _merge_context_features(
        self,
        context_features,
        inputs_embeds,
        input_ids,
        attention_mask,
        position_ids=None,
        labels=None,
    ):
        """
        Merge input_ids with context features into final embeddings

        Args:
            context_features (`torch.Tensor` of shape `(num_contexts, embed_dim)`):
                All context vectors
            inputs_embeds (`torch.Tensor` of shape `(batch_size, sequence_length, embed_dim)`):
                Token embeddings before merging with context embeddings
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Input_ids of tokens
            attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Mask to avoid performing attention on padding token indices.
            position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of positions of each input sequence tokens.
            labels (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*)
                Labels for computing the loss
        """
        batch_size, seq_length, embed_dim = inputs_embeds.shape
        num_contexts = context_features.size(0)

        # Create embeddings for begin and end of context tokens
        begin_context_embed = self.get_input_embeddings()(torch.tensor(self.begin_of_context_token_id, device=context_features.device))
        end_context_embed = self.get_input_embeddings()(torch.tensor(self.end_of_context_token_id, device=context_features.device))

        # Add the embeddings to the context features
        context_features = torch.cat([begin_context_embed.unsqueeze(0), context_features, end_context_embed.unsqueeze(0)], dim=0)

        # Extend sequence length to accommodate context features
        new_seq_length = seq_length + num_contexts + 2
        
        # Create new tensors with extended sequence length
        new_inputs_embeds = torch.cat([context_features.unsqueeze(0).expand(batch_size, -1, -1), inputs_embeds], dim=1)
        new_attention_mask = torch.cat([
            torch.ones(batch_size, num_contexts + 2, device=attention_mask.device, dtype=attention_mask.dtype),
            attention_mask
        ], dim=1)
        new_position_ids = torch.arange(new_seq_length, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        if labels is not None:
            new_labels = torch.cat([
                torch.full((batch_size, num_contexts + 2), self.ignore_index, device=labels.device, dtype=labels.dtype),
                labels
            ], dim=1)
        else:
            new_labels = None

        return new_inputs_embeds, new_attention_mask, new_position_ids, new_labels

    @replace_return_docstrings(output_type=CoEncoderCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        context_input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CoEncoderCausalLMOutputWithPast]:
        """
        Perform a forward pass through the CoEncoder model, optionally conditioning on context input.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Token IDs of the input sequence.
            context_input_ids (`torch.LongTensor` of shape `(batch_size, context_sequence_length)`, *optional*):
                Token IDs of the context input sequence.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices.
            position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indices of positions of each input sequence token.
            past_key_values (`List[torch.FloatTensor]`, *optional*):
                Pre-computed hidden-states (key and value tensors) that can be used to speed up sequential decoding.
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids`, you can pass an embedded representation directly.
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the language modeling loss.
            use_cache (`bool`, *optional*):
                If `True`, past key values will be used to speed up decoding.
            output_attentions (`bool`, *optional*):
                If `True`, return the attention tensors for each layer.
            output_hidden_states (`bool`, *optional*):
                If `True`, return the hidden states of all layers.
            return_dict (`bool`, *optional*):
                If `True`, return a `CoEncoderCausalLMOutputWithPast` instead of a plain tuple.

        Returns:
            `Union[Tuple, CoEncoderCausalLMOutputWithPast]`: A tuple containing various model outputs or a `CoEncoderCausalLMOutputWithPast` instance.
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Process context input through ContextTower
        if context_input_ids is not None:
            context_features = self.context_tower(context_input_ids)
            context_features = self.multi_modal_projector(context_features)
        else:
            context_features = None

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if context_features is not None:
            inputs_embeds, attention_mask, position_ids, labels = self._merge_context_features(
                context_features,
                inputs_embeds,
                input_ids,
                attention_mask,
                position_ids,
                labels,
            )

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = outputs[0]

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CoEncoderCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            context_hidden_states=context_features,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        context_features=None,
        **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "context_features": context_features,
            }
        )
        return model_inputs