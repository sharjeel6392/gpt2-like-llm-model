import numpy as np
from src.utility.gpt_model import GPTMODEL
from src.utility.get_gpt2_weights import download_gpt2
import torch

def assign(param, array):
    tensor = torch.tensor(array)
    if param is None:
        return torch.nn.Parameter(tensor)
    else:
        with torch.no_grad():
            param.copy_(tensor)
        return param

def get_params(gpt:GPTMODEL, model_size:str = '124M'):
    settings, params = download_gpt2(
        model_size=model_size, out_dir='gpt2'
    )
    load_weights_into_gpt(gpt, params)

def load_weights_into_gpt(gpt: GPTMODEL, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])
    for b in range(len(params['blocks'])):
        q_w, k_w, v_w = np.split((params['blocks'][b]['attn']['c_attn'])['w'], 3, axis=-1)
        gpt.transformer_blocks[b].attn.W_query.weight = assign(gpt.transformer_blocks[b].attn.W_query.weight, q_w.T)
        gpt.transformer_blocks[b].attn.W_key.weight = assign(gpt.transformer_blocks[b].attn.W_key.weight, k_w.T)
        gpt.transformer_blocks[b].attn.W_value.weight = assign(gpt.transformer_blocks[b].attn.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split((params['blocks'][b]['attn']['c_attn'])['b'], 3, axis=-1)
        gpt.transformer_blocks[b].attn.W_query.bias = assign(gpt.transformer_blocks[b].attn.W_query.bias, q_b)
        gpt.transformer_blocks[b].attn.W_key.bias = assign(gpt.transformer_blocks[b].attn.W_key.bias, k_b)
        gpt.transformer_blocks[b].attn.W_value.bias = assign(gpt.transformer_blocks[b].attn.W_value.bias, v_b)

        gpt.transformer_blocks[b].attn.out_proj.weight = assign(gpt.transformer_blocks[b].attn.out_proj.weight, params['blocks'][b]['attn']['c_proj']['w'].T)
        gpt.transformer_blocks[b].attn.out_proj.bias = assign(gpt.transformer_blocks[b].attn.out_proj.bias, params['blocks'][b]['attn']['c_proj']['b'])

        gpt.transformer_blocks[b].ffn.layers[0].weight = assign(gpt.transformer_blocks[b].ffn.layers[0].weight, params['blocks'][b]['mlp']["c_fc"]['w'].T)
        gpt.transformer_blocks[b].ffn.layers[0].bias = assign(gpt.transformer_blocks[b].ffn.layers[0].bias, params['blocks'][b]['mlp']["c_fc"]['b'])

        gpt.transformer_blocks[b].ffn.layers[2].weight = assign(gpt.transformer_blocks[b].ffn.layers[2].weight, params['blocks'][b]['mlp']["c_proj"]['w'].T)
        gpt.transformer_blocks[b].ffn.layers[2].bias = assign(gpt.transformer_blocks[b].ffn.layers[2].bias, params['blocks'][b]['mlp']["c_proj"]['b'])

        gpt.transformer_blocks[b].norm1.scale = assign(gpt.transformer_blocks[b].norm1.scale, params['blocks'][b]['ln_1']['g'])
        gpt.transformer_blocks[b].norm1.shift = assign(gpt.transformer_blocks[b].norm1.shift, params['blocks'][b]['ln_1']['b'])

        gpt.transformer_blocks[b].norm2.scale = assign(gpt.transformer_blocks[b].norm2.scale, params['blocks'][b]['ln_2']['g'])
        gpt.transformer_blocks[b].norm2.shift = assign(gpt.transformer_blocks[b].norm2.shift, params['blocks'][b]['ln_2']['b'])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params['g'])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params['b'])
    gpt.out_head.weight = assign(gpt.out_head.weight, params['wte'])