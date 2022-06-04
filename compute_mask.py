import collections
import logging
import os
import random
from typing import List, Dict, Any, Optional
import multiprocessing
import json
import numpy as np
import argparse
from transformers import BertTokenizer
import time
import tqdm

import indexed_dataset
import torch
import logging

from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from transformers import RobertaTokenizerFast

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

file_num_per_json = 10

logger = logging.getLogger(__name__) 
def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input_dataset', default="/root/dedup", type=str, help='Path to input TXT') 
    group = parser.add_argument_group(title='tokenizer')
    group.add_argument('--vocab_path', default="bert-base-chinese", type=str, help='Path of tokenizer')
    #group.add_arguemnt('--tokenizer_folder', default = "/home/caohanwen/tokenizer", type=str)
    group = parser.add_argument_group(title='output data')
    group.add_argument("--output_prefix", default="/root/bm_train_codes/masked_output/", type=str)

    group.add_argument('--json-keys', nargs='+', default=['input_ids', 'lm_pos', 'masked_labels', 'length_list'],
                       help='space separate listed of keys to extract from json')
    group.add_argument("--mask-method", type=str, default='wwm', 
                        choices=['char', 'wwm','char_noreplace', 'char_interval', 'char_subsampling'], help="mask method")
    group.add_argument('--dataset-impl', type=str, default='mmap',
                       choices=['lazy', 'cached', 'mmap'])
    group = parser.add_argument_group(title='training setting')
    group.add_argument('--max_seq_length', type=int, default=512,
                       help='Number of worker processes to launch')
    group.add_argument('--max_predictions_per_seq', type=int, default=512,
                       help='Number of worker processes to launch')
    group.add_argument('--short_seq_prob', type=float, default=0.1,
                       help='Number of worker processes to launch')
    group.add_argument('--masked_lm_prob', type=float, default=0.15,
                       help='Number of worker processes to launch')
    group.add_argument('--workers', type=int, default=64,
                       help='Number of worker processes to launch')
    group.add_argument('--dupe_factor', type=int, default=5,
                       help='Number of worker processes to launch')
    group.add_argument('--log_interval', type=int, default=10000,
                       help='Interval between progress updates')
    group.add_argument('--random_seed', type=int, default=129)
    group.add_argument('--dataset_number', type=int, default=201923)
    args = parser.parse_args()
    args.keep_empty = False

    args.rank = 0
    args.make_vocab_size_divisible_by = 128

    return args

class Encoder(object):

    def __init__(self, tokenizer, mask_method: str, rng: random.Random, rns: np.random.RandomState,
                 max_seq_length: int, short_seq_prob: float, masked_lm_prob: float, max_predictions_per_seq: int,
                 dupe_factor: int, noreplace=False, text_column: str = "text"):
        self.tokenizer = tokenizer
        self.cls_token_id = self.tokenizer.encode('<s>').ids[0] #debug
        self.sep_token_id = self.tokenizer.encode('</s>').ids[0]
        self.mask_token_id = self.tokenizer.encode('<mask>').ids[0]
        self.mask_method = mask_method
        # for some reason @dataclass instances are not pickable
        # would use args as a constructor parameter
        # self.args = args
        self.rng = rng
        self.rns = rns
        self.max_seq_length = max_seq_length
        self.noreplace = 'noreplace' in mask_method
        self.add_interval = 'interval' in mask_method
        self.subsampling = 'subsampling' in mask_method
        if self.subsampling:
            with open('/home/caohanwen/subsample/subsample.json', encoding='utf-8') as f:
                self.subsampling_dict = json.load(f)
        self.short_seq_prob = short_seq_prob
        self.masked_lm_prob = masked_lm_prob
        self.max_predictions_per_seq = max_predictions_per_seq
        self.dupe_factor = dupe_factor
        self.text_column: str = text_column

    def encode(self, document: List[str]) -> Dict[str, Any]:
        """Create `TrainingInstance`s from documents."""
        vocab_words = list(self.tokenizer.get_vocab().keys())
        all_tokens, all_masked_lm_positions, all_masked_lm_labels = [], [], []
        document = json.loads(document.strip())
        document = document['text']
        # document.append('</s>')
        # length = len(document) * 2 - 1
        # for i in range(1, length, 2):
        #     document.insert(i, '</s>')
        # if document[-1] == "[EOS]":
        #     document = document[:-1]
        for _ in range(self.dupe_factor): # for every masking time
            tokens, masked_lm_positions, masked_lm_labels = self.create_instances(
                    document, self.max_seq_length, self.short_seq_prob,
                    self.masked_lm_prob, self.max_predictions_per_seq, vocab_words, self.rng)
            all_tokens.extend(tokens)
            all_masked_lm_positions.extend(masked_lm_positions)
            all_masked_lm_labels.extend(masked_lm_labels)
        full_length = []
        not_full_length = []
        p = self.rns.permutation(len(all_tokens))
        for index in p:
            ans = {"input_ids": all_tokens[index], 'lm_pos': all_masked_lm_positions[index], 'masked_labels': all_masked_lm_labels[index]}
            if len(all_tokens[index]) == self.max_seq_length:
                full_length.append(ans)
            elif len(all_tokens[index]) > 0:
                not_full_length.append(ans)
        # return full_length
        return full_length, not_full_length

    def create_instances(self, 
                        document, max_seq_length, short_seq_prob, masked_lm_prob,
                        max_predictions_per_seq, vocab_words, rng):
        if 'char' in self.mask_method:
            return self.create_instances_by_char(document, max_seq_length, short_seq_prob, masked_lm_prob,
                                            max_predictions_per_seq, vocab_words, rng)
        if self.mask_method == 'wwm':
            return self.create_instances_by_wwm(document, max_seq_length, short_seq_prob, masked_lm_prob,
                                            max_predictions_per_seq, vocab_words, rng)

    def create_instances_by_wwm(self,
                                 document, max_seq_length, short_seq_prob, masked_lm_prob,
                                 max_predictions_per_seq, vocab_words, rng):
        """Creates `TrainingInstance`s for a single document."""
        doc_tokens = []
        for sen in document:
            word_list = sen.split()
            sentence_ids = [] # ids of a single sentence
            for word in word_list:
                word_id = self.tokenizer.encode(word, add_special_tokens=False).ids
                # because add_specil_tokens = False, it doesn't append [cls] [sep] at beginning and end
                # now word_id looks like tensor([7592, 1010, 2026, 2365, 2003, 3013, 2075, 1012])
                # this is a sentence!
                # each id is a word in vocabulary!!!
                if len(word_id) > 1:
                    for i in range(1, len(word_id)):
                        word_id[i] = -word_id[i]
                sentence_ids.extend(word_id)
            if len(sentence_ids) > 0:
                doc_tokens.append(sentence_ids)
        '''
        doc_tokens is:[
            [id1, id2, id3...], this is a sentence
            [id1, id2, id3...], this is a sentence
            ...
        ]
        '''
        # Account for [CLS], [SEP]
        # 
        max_num_tokens = max_seq_length - 2

        # We *usually* want to fill up the entire sequence since we are padding
        # to `max_seq_length` anyways, so short sequences are generally wasted
        # computation. However, we *sometimes*
        # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
        # sequences to minimize the mismatch between pre-training and fine-tuning.
        # The `target_seq_length` is just a rough target however, whereas
        # `max_seq_length` is a hard limit.
        target_seq_length = max_num_tokens
        if rng.random() < short_seq_prob:# 保留short instance
            target_seq_length = rng.randint(2, max_num_tokens) # 从2~max_length中任选一个

        # We DON'T just concatenate all of the tokens from a document into a long
        # sequence and choose an arbitrary split point because this would make the
        # next sentence prediction task too easy. Instead, we split the input into
        # segments "A" and "B" based on the actual "sentences" provided by the user
        # input.
        current_chunk = []
        current_length = 0 # length of current dataset
        i = 0
        all_tokens, all_masked_lm_positions, all_masked_lm_labels = [], [], []
        while i < len(doc_tokens): # for every sentence
            segment = doc_tokens[i] # segment looks like [id1, id2, id3...]
            current_chunk.append(segment) # add this sentence to current dataset
            current_length += len(segment)
            # 要么到最后一句了，要么长度超过了target
            if i == len(doc_tokens) - 1 or current_length >= target_seq_length:
                if current_chunk:
                    # `a_end` is how many segments from `current_chunk` go into the `A`
                    # (first) sentence.
                    tokens = []
                    for j in range(len(current_chunk)):
                        tokens.extend(current_chunk[j])
                    assert len(tokens) >= 1

                    self.truncate_seq_pair(tokens, max_num_tokens, rng) #!!!!!
                    tokens.insert(0, self.cls_token_id) # [cls] at begining
                    tokens.append(self.sep_token_id) # [sep] at end
                    (tokens, masked_lm_positions,
                     masked_lm_labels) = self.create_masked_lm_predictions(
                        tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng)
                    all_tokens.append(tokens)
                    all_masked_lm_positions.append(masked_lm_positions)
                    all_masked_lm_labels.append(masked_lm_labels)
                current_chunk = []
                current_length = 0
            i += 1
        return all_tokens, all_masked_lm_positions, all_masked_lm_labels

    def create_instances_by_char(self,
                            document, max_seq_length, short_seq_prob, masked_lm_prob,
                            max_predictions_per_seq, vocab_words, rng):
        """Creates `TrainingInstance`s for a single document."""
        doc_tokens = []
        for sen in document:
            doc_tokens.append(self.tokenizer.encode(sen, add_special_tokens=False))
        # Account for [CLS], [SEP]
        max_num_tokens = max_seq_length - 2

        # We *usually* want to fill up the entire sequence since we are padding
        # to `max_seq_length` anyways, so short sequences are generally wasted
        # computation. However, we *sometimes*
        # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
        # sequences to minimize the mismatch between pre-training and fine-tuning.
        # The `target_seq_length` is just a rough target however, whereas
        # `max_seq_length` is a hard limit.
        target_seq_length = max_num_tokens
        if rng.random() < short_seq_prob:# 如果保留short sequence   
            target_seq_length = rng.randint(2, max_num_tokens) # 
        # pad enough short sequence together to get a data 
        # which length <= target_seq_length

        # We DON'T just concatenate all of the tokens from a document into a long
        # sequence and choose an arbitrary split point because this would make the
        # next sentence prediction task too easy. Instead, we split the input into
        # segments "A" and "B" based on the actual "sentences" provided by the user
        # input.
        current_chunk = []
        current_length = 0
        i = 0
        all_tokens, all_masked_lm_positions, all_masked_lm_labels = [], [], []
        # doc tokens is a list of list, each element in doc_token is a dataset
        # i.e., a long list of ids, 
        # for every chinese word, first id is positive, the following ids are negative
        while i < len(doc_tokens):
            segment = doc_tokens[i] # a piece of passage
            current_chunk.append(segment)
            current_length += len(segment)
            if i == len(doc_tokens) - 1 or current_length >= target_seq_length:
                if current_chunk:
                    # `a_end` is how many segments from `current_chunk` go into the `A`
                    # (first) sentence.
                    tokens = []
                    for j in range(len(current_chunk)):
                        tokens.extend(current_chunk[j])
                    assert len(tokens) >= 1

                    self.truncate_seq_pair(tokens, max_num_tokens, rng)
                    tokens.insert(0, self.cls_token_id) #[cls]
                    tokens.append(self.sep_token_id) #[sep]
                    
                    (tokens, masked_lm_positions,
                     masked_lm_labels) = self.create_masked_lm_predictions(
                        tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng)
                    all_tokens.append(tokens)
                    all_masked_lm_positions.append(masked_lm_positions)
                    all_masked_lm_labels.append(masked_lm_labels)
                current_chunk = []
                current_length = 0
            i += 1

        return all_tokens, all_masked_lm_positions, all_masked_lm_labels

    def create_masked_lm_predictions(self, tokens, masked_lm_prob,
                                     max_predictions_per_seq, vocab_words, rng):
        """Creates the predictions for the masked LM objective.
        Note that tokens is a list of ids.vocab_words are word from vocab.txt"""

        num_to_predict = min(max_predictions_per_seq,
                             max(1, int(round(len(tokens) * masked_lm_prob))))
        cand_indexes = []

        if self.add_interval and len(tokens) > 4 * num_to_predict:
            for (i, token) in enumerate(tokens):
                if token == self.cls_token_id or token == self.sep_token_id:
                    continue
                if i < len(tokens) - 3 * num_to_predict:
                    cand_indexes.append([i])
                else:
                    break
            cand_indexes = rng.sample(cand_indexes, num_to_predict)
            cand_indexes = sorted(cand_indexes, key=lambda d:d[0])
            cand_indexes = [[item[0] + 3 * index] for index, item in enumerate(cand_indexes)]
            output_tokens = tokens.copy()
        else:
            for (i, token) in enumerate(tokens):
                if token == self.cls_token_id or token == self.sep_token_id:
                    continue
                if len(cand_indexes) >= 1 and token < 0:
                    cand_indexes[-1].append(i)
                else:
                    cand_indexes.append([i])

            rng.shuffle(cand_indexes)
            tokens = [abs(token) for token in tokens] 
            # convert the negative tokens back to positive
            output_tokens = tokens.copy()

        masked_lms = []
        covered_indexes = set()
        for index_set in cand_indexes:
            if len(masked_lms) >= num_to_predict:
                break
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            if len(masked_lms) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                if tokens[index] == self.cls_token_id or tokens[index] == self.sep_token_id:
                    continue
                covered_indexes.add(index)
                masked_token = None
                if self.noreplace:
                    # 85% of the time, replace with [MASK]
                    if rng.random() < 0.85:
                        masked_token = self.mask_token_id
                    else:
                    # 15% of the time, keep original
                        masked_token = tokens[index]
                else:
                    if rng.random() < 0.8:
                        masked_token = self.mask_token_id
                    else:
                        # 10% of the time, keep original
                        if rng.random() < 0.5:
                            masked_token = tokens[index]
                        # 10% of the time, replace with random word
                        else:
                            random_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]
                            masked_token = self.tokenizer.encode(random_token).ids[0]
                            # if tokens[index] > 7992:
                            #     random_token = vocab_words[rng.randint(7992, len(vocab_words) - 1)]
                            #     masked_token = self.tokenizer.convert_tokens_to_ids(random_token)
                            # else:
                            #     random_token = vocab_words[rng.randint(0, 7992)]
                            #     masked_token = self.tokenizer.convert_tokens_to_ids(random_token)
                output_tokens[index] = masked_token
                masked_lms.append((index, tokens[index]))

        masked_lms = sorted(masked_lms, key=lambda x: x[0])

        masked_lm_positions = []
        masked_lm_labels = []
        for p in masked_lms:
            masked_lm_positions.append(p[0])
            masked_lm_labels.append(p[1])
        return output_tokens, masked_lm_positions, masked_lm_labels

    def truncate_seq_pair(self, tokens, max_num_tokens, rng):
        """Truncates a pair of sequences to a maximum sequence length."""
        while True:
            total_length = len(tokens)
            if total_length <= max_num_tokens:
                break
                # if (8038 <= tokens[-1] <= 8048) or (21049 <= tokens[-1] < 21094):
                #     if rng.random() < 0.5:
                #         tokens.pop()
                # break
            # We want to sometimes truncate from the front and sometimes from the
            # back to add more randomness and avoid biases.
            if rng.random() < 0.5:
                del tokens[0]
            else:
                tokens.pop()

def merge_short(not_full_output, max_length):
    def merge_instance(dict1, dict2, length1, length2):
        dict1['input_ids'] = dict1['input_ids'] + dict2['input_ids']
        dict2_pos = [item + length1 for item in dict2['lm_pos']]
        dict1['lm_pos'] = dict1['lm_pos'] + dict2_pos
        dict1['masked_labels'] = dict1['masked_labels'] + dict2['masked_labels']
        if 'length_list' in dict1:
            dict1['length_list'].append(length2)
        else:
            dict1['length_list'] = [length1, length2]
        return length1+length2
    logger.info('merge short sentence...')
    initital_length = len(not_full_output) # 共有几个短例子
    not_full_length_list = [len(item['input_ids']) for item in not_full_output] # 所有短例子的长度
    length_index = {}
    logger.info('assign not full instance by length...')
    for i, length in enumerate(tqdm.tqdm(not_full_length_list)):
        try:
            length_index[length].append(i)
        except:
            length_index[length] = [i]
    logger.info('sort the length_list')
    length_list = sorted(list(length_index.keys()), reverse=True)
    delete_index = []

    for i in tqdm.trange(len(length_list)):
        if i >= len(length_list):
            break
        
        for index in length_index[length_list[i]]:
            total_length = length_list[i]

            while total_length + length_list[-1] <= max_length:
                total_length = merge_instance(not_full_output[index], not_full_output[length_index[length_list[-1]][-1]], total_length, length_list[-1])
                delete_index.append(length_index[length_list[-1]][-1])
                length_index[length_list[-1]].pop()
                if len(length_index[length_list[-1]]) == 0:
                    length_list.pop()
                if len(length_list) <= 1: #无instance可merge
                    break

    new_output = []
    hold_index = list(set(range(initital_length)) - set(delete_index))
    for index in tqdm.tqdm(hold_index, desc='delete index'):
        new_output.append(not_full_output[index])

    logger.info('reduce numbers of instance from {} to {}'.format(initital_length, len(new_output)))
    return new_output

def train_tokenizer():
    special_tokens = ["<pad>","<unk>","<s>","</s>","<mask>"]
    tokenizer = Tokenizer(BPE(unk_token="<unk>", sep_token="</s>", mask_token="<mask>",cls_token="<s>", pad_token="<pad>"))
    file_dir = '/home/caohanwen/bm_train_codes/twitter/original_raw' 
    files = [os.path.join(file_dir,filename) for filename in os.listdir(file_dir)]
    trainer = BpeTrainer(special_tokens=special_tokens, min_frequency = 0, vocab_size = 50265)
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train(files, trainer)
    tokenizer.save('/home/caohanwen/bm_train_codes/tokenizer/tokenizer_.json')

def get_tokenizer():
    tokenizer = Tokenizer.from_file('/root/bm_train_codes/tokenizer/tokenizer_.json')
    return tokenizer

def main():
    args = get_args()

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    rng = random.Random(args.random_seed)
    rns = np.random.RandomState(seed=args.random_seed)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -  %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logger.info,
    )
    tokenizer = get_tokenizer()
    # tokenizer = BertTokenizer.from_pretrained(args.vocab_path) # use transfomers
    encoder = Encoder(tokenizer, mask_method=args.mask_method, rng=rng, rns=rns,
                      max_seq_length=args.max_seq_length,
                      short_seq_prob=args.short_seq_prob,
                      masked_lm_prob=args.masked_lm_prob,
                      dupe_factor=args.dupe_factor,
                      max_predictions_per_seq=args.max_predictions_per_seq)
    file_name = args.input_dataset.split('/')[-1].split('.')[0]
    output_prefix = os.path.join(args.output_prefix + '{}_{}'.format(file_name, args.mask_method))
    # try:
    #     output_names = os.listdir(output_prefix)
    #     index_output = [int(item.split('.')[0].split('_')[-1]) for item in output_names]
    #     max_index = max(index_output) 
    # except:
    #     os.makedirs(output_prefix, exist_ok=True)
    #     max_index = -1
    # start_index = max_index + 1
    filename_list = os.listdir(args.input_dataset)
    filename_list = sorted(filename_list)
    logger.info(f"Use {len(filename_list)} file for pretraining...") 
    k = 10 #start index of masked_output file
    file_cnt = 0 # construct an output with 10 input file
    total_full_output = []
    total_filename = []
    skip_cnt = 0
    for filename in filename_list: # for every imput file
        os.makedirs(output_prefix, exist_ok=True)
        if skip_cnt > 0:
            logger.info(f"file {filename} has already been masked... skip to next file.")
            skip_cnt -= 1
            continue  
        test_name = "{}/input_ids_{}.bin".format(output_prefix,k)
        if os.path.isfile(test_name):
            logger.info(f"file {filename} has already been masked... skip to next file.")
            skip_cnt = file_num_per_json - 1
            k += 1
            continue
        fin = open(os.path.join(args.input_dataset, filename), encoding='utf-8')
        pool = multiprocessing.Pool(args.workers)
        encoded_docs = pool.imap(encoder.encode, fin, 25)
        full_output = []
        not_full_output = []
        for i, output in enumerate(encoded_docs, start=1): # i start from 1
            full, not_full = output
            full_output.extend(full)
            not_full_output.extend(not_full)
            if i % 10000 == 0:
                logger.info('process {} docs'.format(i))
        fin.close()
        pool.close()
        # size of full_output = dup_num * dataset_num
        logger.info('Finish encoding the docs...')
        not_full_output = merge_short(not_full_output, args.max_seq_length)
        full_output.extend(not_full_output)
        # logger.info('shuffle output')
        # random.shuffle(full_output)
        output_bin_files = {}
        output_idx_files = {}
        builders = {}
        total_full_output.extend(full_output)
        total_filename.append(filename)
        if file_cnt >= file_num_per_json or filename == filename_list[-1]:
            logger.info('build index dataset with {} instances'.format(len(total_full_output)))
            logger.info(f'Using json file from {total_filename[0]} to {total_filename[-1]} to construct output file input_ids_{k}.bin etc...')
            random.shuffle(total_full_output)
            for key in args.json_keys:
                output_bin_files[key] = "{}/{}_{}.bin".format(output_prefix, key, k)
                output_idx_files[key] = "{}/{}_{}.idx".format(output_prefix, key, k)
                builders[key] = indexed_dataset.make_builder(output_bin_files[key],
                                                    impl=args.dataset_impl,
                                                    vocab_size=len(tokenizer.get_vocab()))
            for output in tqdm.tqdm(total_full_output): 
                for key in args.json_keys:
                    builders[key].add_item(torch.IntTensor(output.get(key, [])))

            for key in args.json_keys:
                builders[key].finalize(output_idx_files[key])
            total_full_output, total_filename = [], []
            file_cnt = 0
            k += 1
        else:
            file_cnt += 1
        
    logger.info("Finishing data preprocessing...")

if __name__ == '__main__':
    main()
