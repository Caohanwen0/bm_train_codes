import os
from json import dumps
from transformers import RobertaConfig

def convert_roberta_config(save_path, version = None, json_path = None):
    assert version is not None or json_path is not None
    if version:
        old_config = RobertaConfig.from_pretrained(version)
    else:
        old_config = RobertaConfig.from_json_file(json_path)
    new_config = {}
    new_config['vocab_size'] = old_config.vocab_size
    new_config['pad_token_id'] = old_config.pad_token_id
    new_config['dim_model'] = old_config.hidden_size
    new_config['position_size'] = old_config.max_position_embeddings
    new_config['dim_ff'] = old_config.intermediate_size
    new_config['norm_eps'] = old_config.layer_norm_eps
    new_config['num_heads'] = old_config.num_attention_heads
    new_config['num_layers'] = old_config.num_hidden_layers
    new_config['pad_token_id'] = old_config.pad_token_id
    new_config['dropout_p'] = old_config.attention_probs_dropout_prob
    with open(os.path.join(save_path, 'config.json') , 'w')as f:
        f.write(dumps(new_config, indent = 1))

convert_roberta_config('.', 'roberta-base')
        