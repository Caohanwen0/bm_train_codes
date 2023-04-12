# coding=utf-8
# Copyright 2020 The OpenBMB team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import torch
import json

from collections import OrderedDict
from transformers import RobertaModel, RobertaConfig, RobertaForMaskedLM
from model_center.model.config import RobertaConfig as myConfig
from model_center.model import RobertaModel

base_path = '/home/zhaoweilin/ModelCenter'

def scale_roberta_model(scale, model = None, version = None):
    if model is None:
        assert version is not None
        model = RobertaModel.from_pretrained(version)
    config: RobertaConfig = RobertaConfig.from_pretrained(version)

    model_dict = model.state_dict()

    model_dict['input_embedding.weight'] /= scale
    new_dict['position_embedding.weight'] /= scale
    new_dict['token_type_embedding.weight'] /= scale

    for i in range(num_layers):
        new_dict['encoder.layers.' + str(i) + '.self_att.layernorm_before_attention.weight'] /= scale
        new_dict['encoder.layers.' + str(i) + '.self_att.layernorm_before_attention.bias'] /= scale
        new_dict['encoder.layers.' + str(i) + '.self_att.self_attention.attention_out.weight'] /= scale
        new_dict['encoder.layers.' + str(i) + '.self_att.self_attention.attention_out.bias'] /= scale
        new_dict['encoder.layers.' + str(i) + '.ffn.ffn.w_in.w.weight'] /= scale
        new_dict['encoder.layers.' + str(i) + '.ffn.ffn.w_in.w.bias'] /= scale
        new_dict['encoder.layers.' + str(i) + '.ffn.ffn.w_out.weight'] /= scale
        new_dict['encoder.layers.' + str(i) + '.ffn.ffn.w_out.bias'] /= scale

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

    roberta = RobertaModel.from_pretrained(version)
    dict = roberta.state_dict()
    new_dict['pooler.dense.weight'] = dict['pooler.dense.weight']
    new_dict['pooler.dense.bias'] = dict['pooler.dense.bias']
    
    //  
    # torch.save(new_dict, os.path.join(base_path, 'configs', 'roberta', version, 'pytorch_model.pt'))


if __name__ == "__main__":
