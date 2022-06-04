import os
import torch
import json

from collections import OrderedDict
from transformers import RobertaModel, RobertaConfig, RobertaForMaskedLM
from model_center.model.config import RobertaConfig as myConfig

def convert_model():
    config: RobertaConfig = RobertaConfig.from_pretrained('roberta-base')

    num_layers = config.num_hidden_layers
    lmhead_bert = RobertaForMaskedLM(config)
    lmhead_bert.load_state_dict(torch.load('roberta-base-twitter/roberta-base-twitter.pt', map_location = "cpu"))
    dict = lmhead_bert.state_dict()
    new_dict = OrderedDict()

    new_dict['input_embedding.weight'] = dict['roberta.embeddings.word_embeddings.weight']
    new_dict['position_embedding.weight'] = dict['roberta.embeddings.position_embeddings.weight']
    new_dict['token_type_embedding.weight'] = dict['roberta.embeddings.token_type_embeddings.weight']

    for i in range(num_layers):
        new_dict['encoder.layers.' + str(i) + '.self_att.layernorm_before_attention.weight'] = (
            dict['roberta.embeddings.LayerNorm.weight'] if i == 0
            else dict['roberta.encoder.layer.' + str(i - 1) + '.output.LayerNorm.weight'])
        new_dict['encoder.layers.' + str(i) + '.self_att.layernorm_before_attention.bias'] = (
            dict['roberta.embeddings.LayerNorm.bias'] if i == 0
            else dict['roberta.encoder.layer.' + str(i - 1) + '.output.LayerNorm.bias'])
        new_dict['encoder.layers.' + str(i) + '.self_att.self_attention.project_q.weight'] = dict[
            'roberta.encoder.layer.' + str(i) + '.attention.self.query.weight']
        new_dict['encoder.layers.' + str(i) + '.self_att.self_attention.project_q.bias'] = dict[
            'roberta.encoder.layer.' + str(i) + '.attention.self.query.bias']
        new_dict['encoder.layers.' + str(i) + '.self_att.self_attention.project_k.weight'] = dict[
            'roberta.encoder.layer.' + str(i) + '.attention.self.key.weight']
        new_dict['encoder.layers.' + str(i) + '.self_att.self_attention.project_k.bias'] = dict[
            'roberta.encoder.layer.' + str(i) + '.attention.self.key.bias']
        new_dict['encoder.layers.' + str(i) + '.self_att.self_attention.project_v.weight'] = dict[
            'roberta.encoder.layer.' + str(i) + '.attention.self.value.weight']
        new_dict['encoder.layers.' + str(i) + '.self_att.self_attention.project_v.bias'] = dict[
            'roberta.encoder.layer.' + str(i) + '.attention.self.value.bias']
        new_dict['encoder.layers.' + str(i) + '.self_att.self_attention.attention_out.weight'] = dict[
            'roberta.encoder.layer.' + str(i) + '.attention.output.dense.weight']
        new_dict['encoder.layers.' + str(i) + '.self_att.self_attention.attention_out.bias'] = dict[
            'roberta.encoder.layer.' + str(i) + '.attention.output.dense.bias']
        new_dict['encoder.layers.' + str(i) + '.ffn.layernorm_before_ffn.weight'] = dict[
            'roberta.encoder.layer.' + str(i) + '.attention.output.LayerNorm.weight']
        new_dict['encoder.layers.' + str(i) + '.ffn.layernorm_before_ffn.bias'] = dict[
            'roberta.encoder.layer.' + str(i) + '.attention.output.LayerNorm.bias']
        new_dict['encoder.layers.' + str(i) + '.ffn.ffn.w_in.w.weight'] = dict[
            'roberta.encoder.layer.' + str(i) + '.intermediate.dense.weight']
        new_dict['encoder.layers.' + str(i) + '.ffn.ffn.w_in.w.bias'] = dict[
            'roberta.encoder.layer.' + str(i) + '.intermediate.dense.bias']
        new_dict['encoder.layers.' + str(i) + '.ffn.ffn.w_out.weight'] = dict[
            'roberta.encoder.layer.' + str(i) + '.output.dense.weight']
        new_dict['encoder.layers.' + str(i) + '.ffn.ffn.w_out.bias'] = dict[
            'roberta.encoder.layer.' + str(i) + '.output.dense.bias']

    new_dict['encoder.output_layernorm.weight'] = dict[
        'roberta.encoder.layer.' + str(num_layers - 1) + '.output.LayerNorm.weight']
    new_dict['encoder.output_layernorm.bias'] = dict[
        'roberta.encoder.layer.' + str(num_layers - 1) + '.output.LayerNorm.bias']

    new_dict['lm_head.dense.weight'] = dict['lm_head.dense.weight']
    new_dict['lm_head.dense.bias'] = dict['lm_head.dense.bias']
    new_dict['lm_head.layer_norm.weight'] = dict['lm_head.layer_norm.weight']
    new_dict['lm_head.layer_norm.bias'] = dict['lm_head.layer_norm.bias']
    new_dict['lm_head.decoder.weight'] = dict['lm_head.decoder.weight']
    new_dict['lm_head.decoder.bias'] = dict['lm_head.decoder.bias']

    roberta = RobertaModel(config)
    dict = roberta.state_dict()
    new_dict['pooler.dense.weight'] = dict['pooler.dense.weight']
    new_dict['pooler.dense.bias'] = dict['pooler.dense.bias']

    torch.save(new_dict, os.path.join('roberta-base-twitter','transformed_pytorch_model.pt'))


if __name__ == "__main__":
    convert_model()