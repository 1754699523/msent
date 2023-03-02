from __future__ import barry_as_FLUFL

import copy
import warnings
from typing import Optional, Tuple
import math
import torch
import random
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BertLayer

from transformers.activations import ACT2FN
from transformers.configuration_bart import BartConfig
from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings_to_callable,
)
from transformers.modeling_outputs import (
    BaseModelOutput,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from transformers.modeling_bart import (
    BART_INPUTS_DOCSTRING,
    shift_tokens_right,
    _prepare_bart_decoder_inputs,
    _make_linear_from_emb,
    _reorder_buffer, SinusoidalPositionalEmbedding, LearnedPositionalEmbedding, EncoderLayer, LayerNorm, invert_mask,
)

class BartEncoder(nn.Module):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer
    is a :class:`EncoderLayer`.

    Args:
        config: BartConfig
    """

    def __init__(self, config: BartConfig, embed_tokens):
        super().__init__()

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = embed_tokens.embedding_dim
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = config.max_position_embeddings

        self.embed_tokens = embed_tokens

        if config.static_position_embeddings:
            self.embed_positions = SinusoidalPositionalEmbedding(
                config.max_position_embeddings, embed_dim, self.padding_idx
            )
        else:
            self.embed_positions = LearnedPositionalEmbedding(
                config.max_position_embeddings,
                embed_dim,
                self.padding_idx,
                config.extra_pos_embeddings,
            )

        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layernorm_embedding = LayerNorm(embed_dim) if config.normalize_embedding else nn.Identity()
        # mbart has one extra layer_norm
        self.layer_norm = LayerNorm(config.d_model) if config.add_final_layer_norm else None

    def forward(
            self,
            input_ids,
            mixture_ids=None,
            attention_mask=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False,
    ):
        # check attention mask and invert
        if attention_mask is not None:
            attention_mask = invert_mask(attention_mask)

        inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        embed_pos = self.embed_positions(input_ids)
        x = inputs_embeds + embed_pos

        if mixture_ids is not None:
            embed_mix = self.mixture_embeddings(mixture_ids)
            x = x + 0.2 * embed_mix

        x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states = [] if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states.append(x)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                attn = None
            else:
                x, attn = encoder_layer(x, attention_mask, output_attentions=output_attentions)

            if output_attentions:
                all_attentions = all_attentions + (attn,)

        if self.layer_norm:
            x = self.layer_norm(x)
        if output_hidden_states:
            encoder_states.append(x)
            # T x B x C -> B x T x C
            encoder_states = tuple(hidden_state.transpose(0, 1) for hidden_state in encoder_states)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if not return_dict:
            return tuple(v for v in [x, encoder_states, all_attentions] if v is not None)

        return BaseModelOutput(last_hidden_state=x, hidden_states=encoder_states, attentions=all_attentions)