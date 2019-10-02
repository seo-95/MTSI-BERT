import pdb
import random

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import spacy


class BLAdapterDataset(Dataset):
    """
    MTSIAdapterDataset is a module implementing the adapter pattern used as intermediary between you program and the dataset when using Bert.
    It performs all the operations to adapt your dataset input to the one that Bert expects. It does the tokenization, indices transformation, padding etc.
    Another important operation is the __getitem__(idx) that returns the utterances of the dataset corresponding to that dialogye concatenated with
    the first utterance of another random dialogue.

    Input:
        dataset : A class extending Dataset containing you dataset
        tokenizer : The tokenizer to use
        max_sequence_len : The max length of the sequence (for padding)

    Output:
        tok_ids : Indices of each token detected byt the Bert tokenizer
        intent : the intent index for that utterance

    Note:
        Takes care about the feature order returned by the dataset __getitem__()
    """

    def __init__(self, dataset, max_sequence_len, max_dialogue_len):
        self._dataset = dataset
        self._max_sequence_len = max_sequence_len
        self._max_dialogue_len = max_dialogue_len
        self._nlp = spacy.load('en_core_web_md')


    def __len__(self):
        return self._dataset.__len__()


    def __getitem__(self, idx):
        utterances, dialogue_turns, intent, action, dialogue_id = self._dataset.__getitem__(idx)
        # copy the list to avoid modifications happen also in the internal dataset
        utterances = list(utterances)
        dialogue_turns = list(dialogue_turns)
        
        # the random dialogue can also be this one
        random_dialogue_idx = random.randint(0, self._dataset.__len__()-1)
        random_utterances , other_turns, _, _, _ = self._dataset.__getitem__(random_dialogue_idx)
        utterances.append(random_utterances[0])
        dialogue_turns.append(other_turns[0])


        utts_l = []
        utts_lengths = []
        for utt in utterances:
            tokens_l = []
            doc = self._nlp(utt)
            for token in doc:
                if token.is_punct==False:
                    tokens_l.append(token.vector)
            # apply dialogue padding both to utterances and turn vector
            curr_seq_len = len(tokens_l)
            if curr_seq_len <= 0:
                pdb.set_trace()
            assert curr_seq_len > 0, '[ASSERT FAILED] -- seq len < 0' 
            utts_lengths.append(curr_seq_len)

            if curr_seq_len < self._max_sequence_len:
                residual = self._max_sequence_len - curr_seq_len
                tokens_l = tokens_l + [[0]*300]*residual
            utts_l.append(tokens_l)
        

        return torch.tensor(utts_l), torch.tensor(dialogue_turns), torch.tensor(utts_lengths), intent, action, dialogue_id






        
        















        # this vector will contain list of utterances ids
        utt_ids = []
        for utt in utterances:
            tok_utt = self._tokenizer.tokenize(utt)
            tok_utt_len = len(tok_utt)
            # apply padding if needed
            assert tok_utt_len <= self._max_sequence_len
            if tok_utt_len < self._max_sequence_len:
                tok_utt = self.do_padding(tok_utt, self._max_sequence_len)
            tok_idx = self._tokenizer.convert_tokens_to_ids(tok_utt)
            utt_ids.append(tok_idx)

        # apply dialogue padding both to utterances and turn vector
        dialogue_len = len(utt_ids)
        if dialogue_len < self._max_dialogue_len:
            residual = self._max_dialogue_len - dialogue_len
            utt_ids = utt_ids + [[0]*self._max_sequence_len]*residual
            dialogue_turns = dialogue_turns + [0]*residual

        assert len(utt_ids) == self._max_dialogue_len, '[ASSERT FAILED] -- wrong dialogue len of ' + str(len(utt_ids))
        assert len(utt_ids[0]) == self._max_sequence_len, '[ASSERT FAILED] -- wrong sentence len of ' + str(len(utt_ids[0]))
        
        return torch.tensor(utt_ids), torch.tensor(dialogue_turns), intent, action, dialogue_id


    def do_padding(self, tok_text, max_len, pad_token = '[PAD]'):
        """
        Method for applying padding to the tokenized sentence until reaching max_len
        
        Input:
            tok_text : list containing the tokenized text
            max_len : the max len to pad
        """

        diff = max_len - len(tok_text)
        assert diff >= 0

        res = tok_text

        for count in range(diff):
            res.append(pad_token)

        return res
