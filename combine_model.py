import os
import string
import copy

import torch
import torch.nn as nn
from transformers import RobertaTokenizer,RobertaForMaskedLM, RobertaModel,BertForMaskedLM,BertTokenizer
from transformers import RobertaConfig
from tokenizers import Tokenizer

def get_tokenizer_and_save_path():
    save_prefix = '/data/private/caohanwen/OpenSoCo/huggingface/XLM'
    os.makedirs(save_prefix, exist_ok= True)
    tokenizer = Tokenizer.from_file('/data1/private/caohanwen/XLM.json')
    return tokenizer, save_prefix

def combine_bert():
    pass

def combine_roberta(version):
    old_tokenizer = RobertaTokenizer.from_pretrained(version)
    new_tokenizer, save_prefix = get_tokenizer_and_save_path()
    model = RobertaForMaskedLM.from_pretrained(version)

    saved_embedding = {}
    def add_token(token, new_idx):
        embedding_dict = {}
        embedding_dict['word_embedding'] = model.roberta.embeddings.word_embeddings.weight.data[old_tokenizer.convert_tokens_to_ids(token)]
        embedding_dict['decoder'] = model.lm_head.decoder.weight.data[old_tokenizer.convert_tokens_to_ids(token)]
        embedding_dict['decoder_bias'] = model.lm_head.bias[old_tokenizer.convert_tokens_to_ids(token)]
        saved_embedding[new_idx] = embedding_dict

    add_token(old_tokenizer.pad_token, 0)
    add_token(old_tokenizer.unk_token, 1)
    add_token(old_tokenizer.cls_token, 2)
    add_token(old_tokenizer.sep_token, 3)
    add_token(old_tokenizer.mask_token, 4)
    unk = 0
    for word,idx in new_tokenizer.get_vocab().items():
        if idx < 5: #跳过special tokens
            continue
        if old_tokenizer.convert_tokens_to_ids(word) != \
            old_tokenizer.convert_tokens_to_ids(old_tokenizer.unk_token):
            add_token(word, idx) # 只有旧词表里也包含这个词的时候才把embedding保存下来
        else:
            unk+=1

    print(f"{len(saved_embedding)} words are in old vocab,{unk} words are not")
    # 保存模型
    config = RobertaConfig.from_pretrained('nyu-mll/roberta-med-small-1M-1')
    new_size = len(new_tokenizer.get_vocab()) # new vocab size
    new_word_embedding = nn.Embedding(new_size, config.hidden_size, padding_idx = model.roberta.embeddings.word_embeddings.padding_idx)
    new_decoder = nn.Linear(config.hidden_size, new_size)
    new_decoder_bias = nn.Parameter(torch.zeros(new_size), requires_grad = False)
    unk=0
    for word,idx in new_tokenizer.get_vocab().items():
        try:
            new_word_embedding.weight.data[idx]=saved_embedding[idx]['word_embedding']
            new_decoder.weight.data[idx] = saved_embedding[idx]['decoder']
            new_decoder_bias[idx] = saved_embedding[idx]['decoder_bias']
        except KeyError:
            unk+=1
            new_word_embedding.weight.data[idx] = torch.normal(mean=0, std=1.0, size=(config.hidden_size,))
            new_decoder.weight.data[idx] = torch.normal(mean=0.0, std=0.02, size=(config.hidden_size,))
            new_decoder_bias[idx] = torch.tensor(0.) # zero bias initiation
        
    print(f"Get new embedding, {unk} words are initialized via torch.normal...")

    # assert model.roberta.embeddings.word_embeddings.weight.data.shape == new_word_embedding.weight.data.shape 
    # assert model.lm_head.decoder.weight.data.shape == new_decoder.weight.data.shape 
    model.roberta.embeddings.word_embeddings = new_word_embedding
    model.lm_head.decoder = new_decoder
    model.decoder.bias = new_decoder_bias
    model.bias = model.decoder.bias 

    model.config.__dict__['vocab_size'] = new_size
    torch.save(model.state_dict(), os.path.join(save_prefix, version + '.pt'))
    print(f"Saving model to path {os.path.join(save_prefix, version + '.pt')}")

combine_roberta('nyu-mll/roberta-med-small-1M-1')
