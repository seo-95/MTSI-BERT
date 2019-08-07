from pytorch_transformers import BertTokenizer
from torch.utils.data import Dataset
import torch
import pdb
from torch.nn.utils.rnn import pad_sequence





class FriendlyBert(Dataset):
    """
    FriendlyBert is a module implementing the adapter pattern used as intermediary between you program and the dataset when using Bert.
    It performs all the operations to adapt your dataset input to the one that Bert expects. It does the tokenization, indices transformation, padding etc.
    
    Input:
        dataset : A class extending Dataset containing you dataset
        tokenizer : The tokenizer to use
        max_sequence_len : Tha max length of the sequence (for padding)

    Output:
        tok_ids : Indices of each token detected byt the Bert tokenizer
        intent : the intent index for that utterance

    Note:
        Takes care about the feature order returned by the dataset __getitem__()
    """

    def __init__(self, dataset, tokenizer, max_sequence_len, max_dialogue_len):
        self._dataset = dataset
        self._tokenizer = tokenizer
        self._max_sequence_len = max_sequence_len
        self._max_dialogue_len = max_dialogue_len


    def __len__(self):
        return self._dataset.__len__()


    def __getitem__(self, idx):
        utterances, dialogue_turns, intent, action, dialogue_id = self._dataset.__getitem__(idx)

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

        assert len(utt_ids) == self._max_dialogue_len, '[ASSERT FAILED] -- wrong dialogue len of ' + len(utt_ids)
        assert len(utt_ids[0]) == self._max_sequence_len, '[ASSERT FAILED] -- wrong sentence len of ' + len(utt_ids[0])

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




        
        