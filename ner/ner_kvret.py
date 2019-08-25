import os
import torch
import json
from torch.utils.data import Dataset
import pdb


class NERKvretDataset(Dataset):
    """
    This class is the kvret dataset version specific for NER
    """

    def __init__(self, json_path):
        """
        Args:
            json_path (string): Path to the json file of the dataset
        """
        self._dataset = []

        with open(json_path) as json_file:
            json_data = json.load(json_file)

        self._slots_t = set()
        for idx, t_sample in enumerate(json_data):
            for turn in t_sample['dialogue']:
                curr_dialogue = []
                if turn['turn'] == 'driver':
                    curr_utt = turn['data']['utterance']
                    
                elif turn['turn'] == 'assistant':
                    # for slot type count purpose
                    t = list(turn['data']['slots'].keys())
                    [self._slots_t.add(e) for e in t]
                    
                    # here save utterance + entities contained
                    curr_entities = turn['data']['slots']
                    self._dataset.append((curr_utt, curr_entities))


    def __len__(self):
        return len(self._dataset)


    def __getitem__(self, idx):
        return self._dataset[idx]


    def get_slots_type_num(self):
        return len(self._slots_t)


    def build_spacy_dataset(self):
        """
        Return a dataset compatible with spacy NER
        """
        
        spacy_data = []

        for sample in self._dataset:
            utt = sample[0].lower()
            slots = sample[1]
            
            entities_l = []
            for slot_type in slots:
                slot_val = slots[slot_type].lower()
                start_idx = utt.find(slot_val)
                # if not found then continue
                if start_idx == -1:
                    continue
                end_idx = start_idx + len(slot_val)
                entity_tuple = (start_idx, end_idx, str(slot_type))
                entities_l.append(entity_tuple)

            # do not add if no entities were found
            if len(entities_l) > 0:
                curr_res = (utt, {'entities': entities_l})
                spacy_data.append(curr_res)

        return spacy_data