from dataclasses import dataclass
from typing import Optional, Union
import torch
from PIL import Image


@dataclass
class CustomizedStableDiffusionOutput:
    images: Optional[Union[Image.Image, torch.Tensor]]


class Handler:
    def __init__(self, total_steps=30):
        self.cur_step = 0
        self.total_steps = total_steps

    def update_step(self, n=1):
        self.cur_step += n

    def reset_step(self):
        self.cur_step = 0


def customize_token_embeddings(tokenizer, text_encoder, token_info: dict, mean_init=False, ):
    """token_info in format {'name': (placeholder, init.)}"""
    # 1. add placeholder tokens in tokenizer
    placeholder_tokens = [token_info[key][0] for key in token_info.keys()]
    num_added_tokens = tokenizer.add_tokens(placeholder_tokens)
    if num_added_tokens != len(placeholder_tokens):
        raise ValueError('The tokenizer already contains some of the tokens. Please pass different placeholder tokens.')

    # 2. convert the init tokens and placeholder tokens to ids
    # init tokens
    init_token_ids = []
    for key in token_info.keys():
        init_token = token_info[key][1]
        init_token_id = tokenizer.encode(init_token, add_special_tokens=False)
        if len(init_token_id) > 1:
            raise ValueError("The initializer token must be a single token.")
        init_token_ids.append(init_token_id[0])

    # placeholder tokens
    placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)
    # update the token_info: {'name': (placeholder, init_token, placeholder tid, init_token tid)}
    for i, key in enumerate(token_info.keys()):
        ph_tid = placeholder_token_ids[i]
        init_tid = init_token_ids[i]
        token_info[key] += (ph_tid, init_tid,)

    # 3. resize the token embeddings
    text_encoder.resize_token_embeddings(len(tokenizer))

    # 4. initialize the newly added placeholder tokens using the embeddings of the init tokens
    token_embeds = text_encoder.get_input_embeddings().weight.data
    with torch.no_grad():
        for key in token_info:
            new_tid = token_info[key][2]
            init_tid = token_info[key][3]
            if mean_init:
                token_embeds[new_tid] = token_embeds.mean(dim=0, keepdims=True).clone()
            else:
                token_embeds[new_tid] = token_embeds[init_tid].clone()

    return placeholder_token_ids, token_info


def save_embeddings(text_encoder, save_path, token_info: dict):
    """token_info in format {'name': (placeholder, init_token, placeholder tid, init_token tid)}"""
    placeholder_tids = [token_info[key][2] for key in token_info.keys()]
    placeholder_tokens = [token_info[key][0] for key in token_info.keys()]

    token_embeds = text_encoder.get_input_embeddings().weight.data
    custom_embeds_dict = {
        added_token: token_embeds[tid].detach().cpu().clone() for tid, added_token in
        zip(placeholder_tids, placeholder_tokens)
    }

    torch.save(custom_embeds_dict, save_path)
    print('Saved embeddings to', save_path)


def register_embeddings(tokenizer, text_encoder, custom_embeds):
    placeholder_tokens = list(custom_embeds.keys())

    num_added_tokens = tokenizer.add_tokens(placeholder_tokens)
    if num_added_tokens != len(placeholder_tokens):
        raise ValueError('The tokenizer already contains some of the tokens. Please pass different placeholder tokens.')

    placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)
    text_encoder.resize_token_embeddings(len(tokenizer))
    token_embeds = text_encoder.get_input_embeddings().weight.data

    with torch.no_grad():
        for new_token_id, token in zip(placeholder_token_ids, placeholder_tokens):
            token_embeds[new_token_id] = custom_embeds[token].clone()
