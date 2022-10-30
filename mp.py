import random,json
import numpy as np
from typing import List, Dict, Any, Optional
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
        if len(document)<1:
            return [], []
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
                    try:
                        assert len(tokens) >= 1
                    except:
                        continue

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
            doc_tokens.append(self.tokenizer.encode(sen, add_special_tokens=False).ids)
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
                    try:
                        assert len(tokens) >= 1
                    except:
                        continue

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
