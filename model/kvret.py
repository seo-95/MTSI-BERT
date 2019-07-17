import os
import torch
import json
from torch.utils.data import Dataset
import pdb



class KvretDataset(Dataset):
    """Kvret dataset by Stanford NLP group"""

    _INTENTS_TO_IDX = {'schedule': 0,
                         'weather': 1,
                          'navigate': 2}
    _IDX_TO_LABEL = ['schedule', 'weather', 'navigate']


    def __init__(self, json_path):
        """
        Args:
            json_path (string): Path to the json file of the dataset
        """
        self._dataset = []        

        with open(json_path) as json_file:
            json_data = json.load(json_file)

        for t_sample in json_data:
            for utterance in t_sample['dialogue']:
                self._dataset.append({'id': t_sample['scenario']['uuid'],\
                                    'utterance': utterance['data']['utterance'],\
                                    'intent': t_sample['scenario']['task']['intent']})        

    def __len__(self):
        return len(self._dataset)


    def __getitem__(self, idx):
        
        X = self._dataset[idx]['utterance']
        y = self._dataset[idx]['intent']

        return X, KvretDataset._INTENTS_TO_IDX[y]


    def get_max_tokens_per_dialogue(self, tokenizer):
        """
        This utilities is provided in order to find the maximum length of a dialogue inside the dataset based on the given tokenizer

        Input:
            tokenizer : The tokenizer you want to use for the count of tokens. The tokenizer must implement the tokenize(str) method
        
        Output:
            max_num_tokens
            max_num_sentences
            id : ID of the dialogue having the max tokens
        """
        curr_id = ''
        curr_tokens = 0
        curr_sentences = 0
        curr_max = {'num_tokens': 0,\
                    'id': '',\
                    'num_sentences': 0}

        for sample in self._dataset:
            if curr_id != '' and sample['id'] == curr_id:
                curr_tokens += len(tokenizer.tokenize(sample['utterance']))
                curr_sentences += 1

            else: 
                # curr_tokens + num_of_[SEP] = curr_tokens + curr_sentences - 1
                if curr_id != '' and curr_tokens + curr_sentences - 1 >= curr_max['num_tokens']:
                    curr_max['num_tokens'] = curr_tokens
                    curr_max['id'] = curr_id
                    curr_max['num_sentences'] = curr_sentences
                
                curr_id = sample['id']
                curr_tokens = len(tokenizer.tokenize(sample['utterance']))
                curr_sentences = 1

        return curr_max['num_tokens'], curr_max['num_sentences'], curr_max['id']


    def get_max_tokens_per_sentence(self, tokenizer):
        """
        This utilities returns the longest sentence in dataset based on the given tokenizer

        Input:
            tokenizer : The tokenizer you want to use for the count of tokens. The tokenizer must implement the tokenize(str) method
        
        Output:
            max_num_tokens
            sentence
            id : ID of the dialogue having the max tokens
        """
        curr_max = {'num_tokens': 0,\
                    'sentence': '',\
                    'id': ''}

        for t_sample in self._dataset:
            curr_len = len(tokenizer.tokenize(t_sample['utterance']))
            if curr_len > curr_max['num_tokens']:
                curr_max['num_tokens'] = curr_len
                curr_max['sentence'] = t_sample['utterance']
                curr_max['id'] = t_sample['id']
        
        return curr_max['num_tokens'],  curr_max['sentence'], curr_max['id']
            



    