import os
import torch
import json
from torch.utils.data import Dataset
import pdb



class KvretDataset(Dataset):
    """
    Kvret dataset class
    
    This class handles the Kvret dataset by Stanford NLP group.

    Input:
        json_path : path of the kvret.json

    Output: (of the __getitem__(idx))
        utterance : text utterance
        intent : integer label for the intent
        dialogue_id : id of the dialogue


    DATASET: the internal dataset is stored as a dictionary having the following structure
        {
            'id': < id of the dialogue >
            'utterances': < list of utterances > where turn can be 'user' or 'agent'
            'turns' : < list of turns > where turn can be 'user' and 'agent'
            'intent': < name of intent for the dialogue: schedule, weather, navigate >
            'kb_action': < name of the action for the dialogue: fetch, insert >
        }
    """

    _INTENT_TO_IDX = {'schedule': 0,
                        'weather': 1,
                        'navigate': 2}
    _IDX_TO_INTENT = ['schedule', 'weather', 'navigate']

    # KB actions
    _ACTION_TO_IDX = {'fetch': 0,
                        'insert': 1}
    _IDX_TO_ACTION = ['fetch', 'insert']

    # actors (avoid 0 cause it could be used for padding)
    _ACTOR_TO_IDX = {'driver': 1,
                        'assistant': 2}


    def __init__(self, json_path):
        """
        Args:
            json_path (string): Path to the json file of the dataset
        """
        self._dataset = []
        self._dialogueID_to_idx = {}

        with open(json_path) as json_file:
            json_data = json.load(json_file)

        get_action = lambda items : 'insert' if items is None else 'fetch'

        for idx, t_sample in enumerate(json_data):
            if len(t_sample['dialogue']) < 2:
                continue
            curr_dialog = {'id': t_sample['scenario']['uuid'],\
                            'utterances': [],\
                            'turns': [],\
                            'intent': t_sample['scenario']['task']['intent'],\
                            'kb_action': get_action(t_sample['scenario']['kb']['items'])}
            void_utterance = False
            for utterance in t_sample['dialogue']:
                if len(utterance['data']['utterance']) == 0:
                    void_utterance = True
                    break
                curr_dialog['utterances'].append(utterance['data']['utterance']) 
                curr_dialog['turns'].append(utterance['turn'])
            if not void_utterance:
                self._dataset.append(curr_dialog)
                self._dialogueID_to_idx[curr_dialog['id']] = len(self._dataset) - 1


    def __len__(self):
        return len(self._dataset)


    def __getitem__(self, idx):
        
        utterances = self._dataset[idx]['utterances']
        turns_id = []
        dialogue_turns = self._dataset[idx]['turns']
        for t in dialogue_turns:
            turns_id.append(KvretDataset._ACTOR_TO_IDX[t])
        intent = self._dataset[idx]['intent']
        action = self._dataset[idx]['kb_action']
        dialogue_id = self._dataset[idx]['id']
        
        return utterances, turns_id,\
                            KvretDataset._INTENT_TO_IDX[intent],\
                            KvretDataset._ACTION_TO_IDX[action],\
                            dialogue_id


    def get_max_tokens_per_dialogue(self, tokenizer):
        """
        Returns the maximum length of a dialogue inside the dataset based on the given tokenizer

        Input:
            tokenizer : The tokenizer you want to use for the count of tokens. The tokenizer must implement the tokenize(str) method
        
        Output:
            max_num_tokens
            max_num_sentences
            id : ID of the dialogue having the max tokens
        """
        
        curr_max = {'num_tokens': 0,\
                    'id': '',\
                    'num_sentences': 0}

        for dialogue in self._dataset:
            curr_tokens = 0
            for utt in dialogue['utterances']:
                curr_tokens += len(tokenizer.tokenize(utt))
            if curr_tokens > curr_max['num_tokens']:
                curr_max['num_tokens'] = curr_tokens
                curr_max['id'] = dialogue['id']
                curr_max['num_sentences'] = len(dialogue['utterances'])

        return curr_max['num_tokens'], curr_max['num_sentences'], curr_max['id']


    def get_max_tokens_per_sentence(self, tokenizer):
        """
        Returns the longest sentence in dataset based on the given tokenizer

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

        for dialogue in self._dataset:
            for utt in dialogue['utterances']:
                curr_len = len(tokenizer.tokenize(utt))
                if curr_len > curr_max['num_tokens']:
                    curr_max['num_tokens'] = curr_len
                    curr_max['sentence'] = utt
                    curr_max['id'] = dialogue['id']
        
        return curr_max['num_tokens'],  curr_max['sentence'], curr_max['id']


    def get_max_utterances_per_dialogue(self):
        
        curr_max = 0
        max_dialogue_id = ''

        for dialogue in self._dataset:
            curr_len = len(dialogue['utterances'])
            if curr_len > curr_max:
                curr_max = curr_len
                max_dialogue_id = dialogue['id']

        return curr_max, max_dialogue_id


    def get_action_frequency(self):
        
        fetch_count = 0
        insert_count = 0

        for dialogue in self._dataset:
            if dialogue['kb_action'] == 'fetch':
                fetch_count += 1
            else:
                if dialogue['kb_action'] == 'insert':
                    insert_count += 1
                else:
                    raise '[EXCEPTION] -- Neither fetch nor insert'

        assert fetch_count + insert_count == len(self._dataset), '[ASSERT FAILED] -- Misleading count!'

        return fetch_count, insert_count


    def get_max_num_user_utterances(self):

        curr_max = 0
        max_dialogue_id = ''

        for dialogue in self._dataset:
            curr_user_u_count = 0
            for t in dialogue['turns']:
                if t == 'driver':
                    curr_user_u_count += 1

            if curr_user_u_count > curr_max:
                curr_max = curr_user_u_count
                max_dialogue_id = dialogue['id']

        return curr_max, max_dialogue_id


    def get_dialogues_with_subsequent_same_actor_utterances(self):

        id_list = []
        for dialogue in self._dataset:
            previous_turn = ''
            for t in dialogue['turns']:
                if t == previous_turn:
                    id_list.append(dialogue['id'])
                    break
                previous_turn = t

        return id_list

    
    def remove_subsequent_actor_utterances(self):

        """
        To remove all the subsequent utterances of the same actor and keep only the last one
        """

        for dialogue in self._dataset:
            previous_turn = ''
            delimiters_l = []
            sx_idx = -1
            dx_idx = -1
            for curr_idx, t in enumerate(dialogue['turns']):
                if t == previous_turn:
                    # set the sx index if first time for this occurence
                    if sx_idx == -1:
                        sx_idx = curr_idx - 1
                    dx_idx = curr_idx
                    # if last turn then save delimiters
                    if curr_idx == len(dialogue['turns']) - 1:
                        delimiters_l.append({'sx': sx_idx, 'dx': dx_idx})

                else:
                    # save delimiters and reset the sx and dx index if a range was found
                    if sx_idx != -1 and dx_idx != -1:
                        delimiters_l.append({'sx': sx_idx, 'dx': dx_idx})
                        sx_idx = -1
                        dx_idx = -1

                previous_turn = t

            for delim in delimiters_l:
                del dialogue['turns'][delim['sx']:delim['dx']]
                del dialogue['utterances'][delim['sx']:delim['dx']]

            assert len(dialogue['utterances']) == len(dialogue['turns']), '[ASSERT FAILED] -- len(utt) != len(turns)'

    
    def get_total_user_utterances(self):

        curr_count = 0
        for dialogue in self._dataset:
            curr_count += dialogue['turns'].count('driver')

        return curr_count
