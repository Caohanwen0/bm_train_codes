import collections
import logging
import os
import random
# from typing import List, Dict, Any, Optional
import multiprocessing
from multiprocessing import set_start_method
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
from mp import Encoder

file_num_per_json = 5

logger = logging.getLogger(__name__) 
def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input_dataset', default="/data/private/caohanwen/OpenSoCo/preprocessed_twitter_dedup", type=str, help='Path to input TXT') 
    group = parser.add_argument_group(title='tokenizer')
    #group.add_argument('--vocab_path', default="bert-base-chinese", type=str, help='Path of tokenizer')
    group.add_argument('--tokenizer_folder', default = "/home/caohanwen/tokenizer/", type=str)
    group = parser.add_argument_group(title='output data')
    group.add_argument("--output_prefix", default="/data_new/private/caohanwen/masked_output/XLM_", type=str)

    group = parser.add_argument_group(title='start index')
    group.add_argument("--start_index", default=0, type=int)
     
    group.add_argument('--json-keys', nargs='+', default=['input_ids', 'lm_pos', 'masked_labels', 'length_list'],
                       help='space separate listed of keys to extract from json')
    group.add_argument("--mask-method", type=str, default='char', 
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
    group.add_argument('--dupe_factor', type=int, default=1,
                       help='Number of worker processes to launch')
    group.add_argument('--log_interval', type=int, default=10000,
                       help='Interval between progress updates')
    group.add_argument('--random_seed', type=int, default=12)
    group.add_argument('--dataset_number', type=int, default=201923)
    args = parser.parse_args()
    args.keep_empty = False

    args.rank = 0
    args.make_vocab_size_divisible_by = 128

    return args

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

def get_tokenizer(args):
    tokenizer = Tokenizer.from_file(args.tokenizer_folder)
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
    tokenizer = get_tokenizer(args)
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
    logger.info(f"Use {len(filename_list)} file for pretraining...")
    random.seed(args.random_seed) 
    random.shuffle(filename_list)
    k = args.start_index #start index of masked_output file
    file_cnt = 1 # construct an output with 10 input file
    total_full_output = []
    total_filename = []
    skip_cnt = 0
    for filename in filename_list: # for every imput file
        logger.info(f"Start encoding docs {filename}...")
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
        with multiprocessing.get_context('spawn').Pool(args.workers) as pool:
            encoded_docs = pool.imap(encoder.encode, fin, chunksize=30)
            full_output = []
            not_full_output = []
            for i, output in enumerate(encoded_docs, start=1): # i start from 1
                full, not_full = output
                full_output.extend(full)
                not_full_output.extend(not_full)
                if i % 10000 == 0:
                    logger.info('process {} docs'.format(i))
            pool.close()
        fin.close()
        # pool.close()
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
            logger.info(f'Using json file from {total_filename[0]} to {total_filename[-1]} to construct output file input_ids_{k}.bin etc...')
            total_full_output, total_filename = [], []
            file_cnt = 1
            k += 1
        else:
            file_cnt += 1
        
    logger.info("Finishing data preprocessing...")

if __name__ == '__main__':
    main()
