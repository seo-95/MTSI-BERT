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
                self._dataset.append({'utterance': utterance['data']['utterance'],\
                                    'intent': t_sample['scenario']['task']['intent']})        

    def __len__(self):
        return len(self._dataset)


    def __getitem__(self, idx):
        
        X = self._dataset[idx]['utterance']
        y = self._dataset[idx]['intent']
        


        return X, KvretDataset._INTENTS_TO_IDX[y]

    