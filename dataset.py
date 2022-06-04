import torch
import torch.utils.data as data
from model_center.dataset import MMapIndexedDataset
import random
import numpy as np
import scipy.linalg
from tokenizers import Tokenizer

def get_tokenizer():
    tokenizer = Tokenizer.from_file('/home/user/bm_train_codes/tokenizer/tokenizer.json')
    return tokenizer

class BertDataset(data.Dataset):
    def __init__(self, input_ids:     MMapIndexedDataset,# original sentence
                       lm_pos:        MMapIndexedDataset,# mask position
                       masked_labels: MMapIndexedDataset,# mask label
                       length_list:   MMapIndexedDataset,
                       max_seq_length = 512,
                       pad_id = 0):

        self.input_ids = input_ids
        self.lm_pos = lm_pos
        self.masked_labels = masked_labels
        self.length_list = length_list
        self.max_seq_length = max_seq_length
        self.pad_id = pad_id
   
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        input_ids = np.array(self.input_ids[index], dtype='int32')
        lm_pos = self.lm_pos[index]
        masked_labels = self.masked_labels[index]
        length_list = self.length_list[index]

        input_length = len(input_ids)
        if input_length < self.max_seq_length:
            input_ids = np.pad(input_ids, (0, self.max_seq_length - input_length))
            # padding zero after not long enough instance

        if len(length_list) == 0: # only one instance(not merged short instance)
            ones = np.ones([input_length, input_length])
            zeros = np.zeros([self.max_seq_length - input_length, self.max_seq_length - input_length])
            attention_mask = scipy.linalg.block_diag(ones, zeros)
        else:
            block_matrix_list = []
            for length in length_list:
                block_matrix_list.append(np.ones([length, length]))
            block_matrix_list.append(np.zeros([self.max_seq_length - sum(length_list), self.max_seq_length - sum(length_list)]))
            attention_mask = scipy.linalg.block_diag(*block_matrix_list)
        
        labels = np.full([self.max_seq_length, ], -100, dtype="int32")
        labels[lm_pos] = masked_labels
        return torch.LongTensor(input_ids), torch.LongTensor(attention_mask).byte(), torch.LongTensor(labels)

if __name__ == "__main__":
    tokenizer = get_tokenizer()
    input_ids = MMapIndexedDataset('/home/caohanwen/bm_train_codes/masked_output/original_wwm/input_ids_0_38')
    length_list = MMapIndexedDataset('/home/caohanwen/bm_train_codes/masked_output/original_wwm/length_list_0_38')
    masked_labels = MMapIndexedDataset('/home/caohanwen/bm_train_codes/masked_output/original_wwm/masked_labels_0_38')
    lm_pos = MMapIndexedDataset('/home/caohanwen/bm_train_codes/masked_output/original_wwm/lm_pos_0_38')
    bert_dataset = BertDataset(input_ids, lm_pos, masked_labels, length_list)
    
    for i in range(len(bert_dataset)):
        if len(length_list[i]) != 0:
            input_ids, attention_mask, labels = bert_dataset[i]
            
            break