import os
import string
import copy

import torch
import torch.nn as nn
from transformers import RobertaTokenizer,RobertaForMaskedLM, RobertaModel
from transformers import RobertaConfig
from tokenizers import Tokenizer

def get_tokenizer():
    tokenizer = Tokenizer.from_file('tokenizer/tokenizer_.json')
    return tokenizer

old_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
new_tokenizer = get_tokenizer()
model = RobertaForMaskedLM.from_pretrained('roberta-base')

saved_embedding = {}
def add_token(token, new_idx):
    embedding_dict = {}
    embedding_dict['word_embedding'] = model.roberta.embeddings.word_embeddings.weight.data[old_tokenizer.convert_tokens_to_ids(token)]
    embedding_dict['decoder'] = model.lm_head.decoder.weight.data[old_tokenizer.convert_tokens_to_ids(token)]
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
config = RobertaConfig()
new_size = len(new_tokenizer.get_vocab())
new_word_embedding = nn.Embedding(new_size, config.hidden_size, padding_idx = model.roberta.embeddings.word_embeddings.padding_idx)
new_decoder = nn.Linear(config.hidden_size, new_size)
unk=0
for word,idx in new_tokenizer.get_vocab().items():
    try:
        new_word_embedding.weight.data[idx]=saved_embedding[idx]['word_embedding']
        new_decoder.weight.data[idx] = saved_embedding[idx]['decoder']
    except KeyError:
        unk+=1
        # new_word_embedding.weight.data[idx] = torch.normal(mean=0.0, std=1.0, size=(config.hidden_size,))
        # new_decoder.weight.data[idx] = torch.normal(mean=0.0, std=1.0, size=(config.hidden_size,))
    
print(f"Get new embedding, {unk} words are initialized via torch.normal...")

assert model.roberta.embeddings.word_embeddings.weight.data.shape == new_word_embedding.weight.data.shape 
assert model.lm_head.decoder.weight.data.shape == new_decoder.weight.data.shape 
model.roberta.embeddings.word_embeddings = new_word_embedding
model.lm_head.decoder = new_decoder

model.config.__dict__['vocab_size'] = new_size
# model.save_pretrained('roberta-base-twitter')
torch.save(model.state_dict(), 'roberta-base-twitter/roberta-base-twitter.pt')