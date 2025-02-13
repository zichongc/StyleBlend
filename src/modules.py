from typing import Literal, Callable, Optional, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.utils import deprecate, scale_lora_layers, unscale_lora_layers
from .utils import Handler


sd_indexes = {
    'down_blocks.0.attentions.0.transformer_blocks.0.attn1': 0,
    'down_blocks.0.attentions.1.transformer_blocks.0.attn1': 1,
    'down_blocks.1.attentions.0.transformer_blocks.0.attn1': 2,
    'down_blocks.1.attentions.1.transformer_blocks.0.attn1': 3,
    'down_blocks.2.attentions.0.transformer_blocks.0.attn1': 4,
    'down_blocks.2.attentions.1.transformer_blocks.0.attn1': 5,
    'mid_block.attentions.0.transformer_blocks.0.attn1': 6,
    'up_blocks.1.attentions.0.transformer_blocks.0.attn1': 7 ,
    'up_blocks.1.attentions.1.transformer_blocks.0.attn1': 8,
    'up_blocks.1.attentions.2.transformer_blocks.0.attn1': 9,
    'up_blocks.2.attentions.0.transformer_blocks.0.attn1': 10,
    'up_blocks.2.attentions.1.transformer_blocks.0.attn1': 11,
    'up_blocks.2.attentions.2.transformer_blocks.0.attn1': 12,
    'up_blocks.3.attentions.0.transformer_blocks.0.attn1': 13,
    'up_blocks.3.attentions.1.transformer_blocks.0.attn1': 14,
    'up_blocks.3.attentions.2.transformer_blocks.0.attn1': 15,
    'down_blocks.0.attentions.0.transformer_blocks.0.attn2': 0,
    'down_blocks.0.attentions.1.transformer_blocks.0.attn2': 1,
    'down_blocks.1.attentions.0.transformer_blocks.0.attn2': 2,
    'down_blocks.1.attentions.1.transformer_blocks.0.attn2': 3,
    'down_blocks.2.attentions.0.transformer_blocks.0.attn2': 4,
    'down_blocks.2.attentions.1.transformer_blocks.0.attn2': 5,
    'mid_block.attentions.0.transformer_blocks.0.attn2': 6,
    'up_blocks.1.attentions.0.transformer_blocks.0.attn2': 7 ,
    'up_blocks.1.attentions.1.transformer_blocks.0.attn2': 8,
    'up_blocks.1.attentions.2.transformer_blocks.0.attn2': 9,
    'up_blocks.2.attentions.0.transformer_blocks.0.attn2': 10,
    'up_blocks.2.attentions.1.transformer_blocks.0.attn2': 11,
    'up_blocks.2.attentions.2.transformer_blocks.0.attn2': 12,
    'up_blocks.3.attentions.0.transformer_blocks.0.attn2': 13,
    'up_blocks.3.attentions.1.transformer_blocks.0.attn2': 14,
    'up_blocks.3.attentions.2.transformer_blocks.0.attn2': 15,
}


def feature_blender():
    def inject(attn, query, key, value):
        """
        query is from composition style branch to texture style branch
        key and value are from texture style branch to composition style branch
        """
        query[[1, 3], ...] = query[[2, 2], ...]
        key[[0, 2], ...] = key[[3, 3], ...]
        value[[0, 2], ...] = value[[3, 3], ...]
        return query, key, value

    return inject


def structure_injector(direction: Literal['c2t', 't2c'] = 'c2t'):
    """
    direction:
        c2t: features injection from composition style branch to texture style branch
        t2c: features injection from texture style branch to composition style branch
    """

    def inject(attn, query, key, value):
        if direction == 'c2t':
            query[[1, 3], ...] = query[[2, 2], ...]
        elif direction == 't2c':
            query[[0, 2], ...] = query[[3, 3], ...]

        return query, key, value

    return inject


def appearance_injector(direction: Literal['c2t', 't2c'] = 'c2t'):
    """
    direction:
        c2t: features injection from composition style branch to texture style branch
        t2c: features injection from texture style branch to composition style branch
    """

    def inject(attn, query, key, value):
        if direction == 'c2t':
            key[[1, 3], ...] = key[[2, 2], ...]
            value[[1, 3], ...] = value[[2, 2], ...]
        elif direction == 't2c':
            key[[0, 2], ...] = key[[3, 3], ...]
            value[[0, 2], ...] = value[[3, 3], ...]

        return query, key, value

    return inject


def styleblend_self_attention(block_class, csb_lora_scale=1.0, feature_injector=None, handler: Handler = None,
                                steps_to_blend=None):
    class Attention(block_class):
        _parent_class = block_class
        _scale = csb_lora_scale
        _feature_injector = feature_injector
        _steps_to_blend = steps_to_blend if steps_to_blend is not None else [i for i in range(handler.total_steps)]

        def forward(
            attn,
            hidden_states: torch.Tensor,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            temb: Optional[torch.Tensor] = None,
            *args,
            **kwargs,
        ) -> torch.Tensor:
            if len(args) > 0 or kwargs.get("scale", None) is not None:
                deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
                deprecate("scale", "1.0.0", deprecation_message)

            residual = hidden_states
            if attn.spatial_norm is not None:
                hidden_states = attn.spatial_norm(hidden_states, temb)

            input_ndim = hidden_states.ndim

            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

            batch_size, sequence_length, _ = (
                hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
            )

            if attention_mask is not None:
                attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
                # scaled_dot_product_attention expects attention_mask shape to be
                # (batch, heads, source_length, target_length)
                attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

            if attn.group_norm is not None:
                hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

            # Adjust the lora scale for composition style branch (csb).
            # Keep the lora scale to 1.0 for texture style branch (tsb).
            # For each batch: {neg_composition, neg_texture, composition, texture}, hidden states in shape [4, S, D]
            fsb_hidden_states = hidden_states[[1, 3], ...]
            fsb_encoder_hidden_states = encoder_hidden_states[[1, 3], ...]

            fsb_query = attn.to_q(fsb_hidden_states)
            fsb_key = attn.to_k(fsb_encoder_hidden_states)
            fsb_value = attn.to_v(fsb_encoder_hidden_states)

            scale_lora_layers(attn, attn._scale)

            csb_hidden_states = hidden_states[[0, 2], ...]
            csb_encoder_hidden_states = encoder_hidden_states[[0, 2], ...]

            csb_query = attn.to_q(csb_hidden_states)
            csb_key = attn.to_k(csb_encoder_hidden_states)
            csb_value = attn.to_v(csb_encoder_hidden_states)
            
            unscale_lora_layers(attn, attn._scale)

            query = torch.stack([csb_query[0], fsb_query[0], csb_query[1], fsb_query[1]])
            key = torch.stack([csb_key[0], fsb_key[0], csb_key[1], fsb_key[1]])
            value = torch.stack([csb_value[0], fsb_value[0], csb_value[1], fsb_value[1]])

            # Custom QKV feature injection
            if (attn._feature_injector is not None 
                and isinstance(attn._feature_injector, Callable) and handler.cur_step in attn._steps_to_blend
            ):
                query, key, value = attn._feature_injector(query, key, value)

            inner_dim = key.shape[-1]
            head_dim = inner_dim // attn.heads

            query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            if attn.norm_q is not None:
                query = attn.norm_q(query)
            if attn.norm_k is not None:
                key = attn.norm_k(key)

            # the output of sdp = (batch, num_heads, seq_len, head_dim)
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )

            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            hidden_states = hidden_states.to(query.dtype)

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

            if attn.residual_connection:
                hidden_states = hidden_states + residual

            hidden_states = hidden_states / attn.rescale_output_factor 

            return hidden_states

    return Attention


def styleblend_cross_attention(block_class, csb_lora_scale=1.0, feature_injector=None, handler: Handler = None,
                                 steps_to_blend=None):
    class Attention(block_class):
        _parent_class = block_class
        _scale = csb_lora_scale
        _feature_injector = feature_injector
        _steps_to_blend = steps_to_blend if steps_to_blend is not None else [i for i in range(handler.total_steps)]

        def forward(
            attn,
            hidden_states: torch.Tensor,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            temb: Optional[torch.Tensor] = None,
            *args,
            **kwargs,
        ) -> torch.Tensor:
            if len(args) > 0 or kwargs.get("scale", None) is not None:
                deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
                deprecate("scale", "1.0.0", deprecation_message)

            residual = hidden_states
            if attn.spatial_norm is not None:
                hidden_states = attn.spatial_norm(hidden_states, temb)

            input_ndim = hidden_states.ndim

            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

            batch_size, sequence_length, _ = (
                hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
            )

            if attention_mask is not None:
                attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
                # scaled_dot_product_attention expects attention_mask shape to be
                # (batch, heads, source_length, target_length)
                attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

            if attn.group_norm is not None:
                hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

            # Adjust the lora scale for composition style branch (csb).
            # Keep the lora scale to 1.0 for texture style branch (tsb).
            # For each batch: {neg_composition, neg_texture, composition, texture}, hidden states in shape [4, S, D]
            fsb_hidden_states = hidden_states[[1, 3], ...]
            fsb_encoder_hidden_states = encoder_hidden_states[[1, 3], ...]

            fsb_query = attn.to_q(fsb_hidden_states)
            fsb_key = attn.to_k(fsb_encoder_hidden_states)
            fsb_value = attn.to_v(fsb_encoder_hidden_states)
            
            scale_lora_layers(attn, attn._scale)
            csb_hidden_states = hidden_states[[0, 2], ...]
            csb_encoder_hidden_states = encoder_hidden_states[[0, 2], ...]

            csb_query = attn.to_q(csb_hidden_states)
            csb_key = attn.to_k(csb_encoder_hidden_states)
            csb_value = attn.to_v(csb_encoder_hidden_states)
            
            unscale_lora_layers(attn, attn._scale)

            query = torch.stack([csb_query[0], fsb_query[0], csb_query[1], fsb_query[1]])
            key = torch.stack([csb_key[0], fsb_key[0], csb_key[1], fsb_key[1]])
            value = torch.stack([csb_value[0], fsb_value[0], csb_value[1], fsb_value[1]])

            # Custom QKV feature injection
            if attn._feature_injector is not None and isinstance(attn._feature_injector,
                                                                 Callable) and handler.cur_step in attn._steps_to_blend:
                query, key, value = attn._feature_injector(query, key, value)

            inner_dim = key.shape[-1]
            head_dim = inner_dim // attn.heads

            query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            if attn.norm_q is not None:
                query = attn.norm_q(query)
            if attn.norm_k is not None:
                key = attn.norm_k(key)

            # the output of sdp = (batch, num_heads, seq_len, head_dim)
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )

            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            hidden_states = hidden_states.to(query.dtype)

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

            if attn.residual_connection:
                hidden_states = hidden_states + residual

            hidden_states = hidden_states / attn.rescale_output_factor

            return hidden_states

    return Attention
