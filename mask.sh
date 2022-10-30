#!/bin/bash
 
#SBATCH -J mask


python compute_mask.py --workers 64 --input_dataset /data_new/private/caohanwen/preprocessed_en_1 --output_prefix /data_new/private/caohanwen/masked_output/ --tokenizer_folder /data_new/private/caohanwen/tokenizer/tokenizer_twitter_reddit_ccnews.json --start_index 0 --random_seed 1312
