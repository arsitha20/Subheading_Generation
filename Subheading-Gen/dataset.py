import torch
import os
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Union, List, Dict, Tuple

import pandas as pd
import numpy as np

class Subheading_Dataset(Dataset):
    def __init__(self, args, data_path, seq2seq_tokenizer, electra_tokenizer):
        self.data = pd.read_csv(data_path)
        self.len = self.data.shape[0]
        
        self.headline_col = args.headline_col
        self.subheading_col = args.subheading_col
        self.body_col = args.body_col

        self.headline_max_len = args.headline_max_len
        self.subheading_max_len = args.subheading_max_len
        self.body_max_len = args.body_max_len

        self.mlm_probability = args.mlm_probability

        self.seq2seq_tokenizer = seq2seq_tokenizer
        self.electra_tokenizer = electra_tokenizer
        
        # BART Part
        self.bos_index = self.seq2seq_tokenizer.bos_token_id
        self.eos_index = self.seq2seq_tokenizer.eos_token_id
        self.pad_index = self.seq2seq_tokenizer.pad_token_id
        self.eos_token = '</s>'
        ##elf.eos_token = ''
        self.ignore_index = -100

        # Electra Part
        self.cls = self.electra_tokenizer.cls_token
        self.sep = self.electra_tokenizer.sep_token
        self.mask = "[MASK]"

    def mask_tokens(self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
            """
            Prepare masked tokens inputs/inputs for masked language modeling: 80% MASK, 10% random, 10% original.
            """
            inputs = inputs.clone()

            # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
            probability_matrix = torch.full(inputs.shape, self.mlm_probability)
            if special_tokens_mask is None:
                special_tokens_mask = [1 if val == 2 or val ==0 else 0 for val in inputs.tolist()]
                special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
            else:
                special_tokens_mask = special_tokens_mask.bool()

            probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
            masked_indices = torch.bernoulli(probability_matrix).bool()

            # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
            indices_replaced = torch.bernoulli(torch.full(inputs.shape, 0.8)).bool() & masked_indices
            inputs[indices_replaced] = self.electra_tokenizer.convert_tokens_to_ids(self.electra_tokenizer.mask_token)

            # 10% of the time, we replace masked input tokens with random word
            indices_random = torch.bernoulli(torch.full(inputs.shape, 0.5)).bool() & masked_indices & ~indices_replaced
            random_words = torch.randint(len(self.electra_tokenizer), inputs.shape, dtype=torch.long)
            inputs[indices_random] = random_words[indices_random]

            # The rest of the time (10% of the time) we keep the masked input tokens unchanged
            return inputs

    def add_ignored_data(self, inputs:List, max_len:int) -> torch.Tensor:
        if len(inputs) < max_len:
            pad = np.array([self.ignore_index] * (max_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
        else:
            inputs = np.array(inputs[:max_len])

        return  torch.from_numpy(inputs)

    def __len__(self):
        return self.len

    def __getitem__(self, idx:int) -> Dict:

        instance = self.data.iloc[idx]

        seq2seq_enc_inputs = self.seq2seq_tokenizer.encode_plus(text=instance[self.body_col]+self.eos_token, padding='max_length', truncation=True, max_length=self.body_max_len, 
        add_special_tokens=True, return_token_type_ids=False, return_tensors='pt')

        seq2seq_dec_inputs = self.seq2seq_tokenizer.encode_plus(text= self.eos_token + instance[self.subheading_col], padding='max_length', truncation=True, max_length=self.subheading_max_len, 
        add_special_tokens=True, return_token_type_ids=False, return_tensors='pt')

        electra_labels = self.electra_tokenizer.encode_plus(text=self.cls + instance[self.headline_col], padding='max_length', truncation=True, max_length=self.headline_max_len, 
        add_special_tokens=False, return_token_type_ids=False, return_tensors='pt')

        electra_mask_input_ids = self.mask_tokens(electra_labels['input_ids'][0])


        seq2_seq_label_ids = self.seq2seq_tokenizer.encode(instance[self.subheading_col] + self.eos_token)
        seq2_seq_label_ids = self.add_ignored_data(seq2_seq_label_ids, max_len=self.subheading_max_len)

        seq2seq_dict = dict()
        seq2seq_dict['encoder_input_ids'] = seq2seq_enc_inputs['input_ids'][0]
        seq2seq_dict['encoder_attention_mask'] = seq2seq_enc_inputs['attention_mask'][0]
        seq2seq_dict['decoder_input_ids'] = seq2seq_dec_inputs['input_ids'][0]
        seq2seq_dict['decoder_attention_mask'] = seq2seq_dec_inputs['attention_mask'][0]
        seq2seq_dict['decoder_labels'] = seq2_seq_label_ids
        
        electra_dict = dict()
        electra_dict['labels'] = electra_labels['input_ids'][0]
        electra_dict['generator_input_ids'] = electra_mask_input_ids
        electra_dict['attention_mask'] = electra_labels['attention_mask'][0]

        return dict(bart=seq2seq_dict, electra=electra_dict)