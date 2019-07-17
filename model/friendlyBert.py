from pytorch_transformers import BertTokenizer
from torch.utils.data import Dataset
import torch
import pdb





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
        The intent should be the last returned elements in the tuple provided by the __getitem__() of the dataset you provide. The text the first one.
    """

    def __init__(self, dataset, tokenizer, max_sequence_len):
        self._dataset = dataset
        self._tokenizer = tokenizer
        self._max_sequence_len = max_sequence_len


    def __len__(self):
        return self._dataset.__len__()


    def __getitem__(self, idx):
        data = self._dataset.__getitem__(idx)

        # prepare the utterance
        text = data[0]
        tok_text = self._tokenizer.tokenize(text)
        tok_text_len = len(tok_text)

        # apply padding if needed
        assert tok_text_len <= self._max_sequence_len

        if tok_text_len < self._max_sequence_len:
            tok_text = self.do_padding(tok_text, self._max_sequence_len)

        tok_idx = self._tokenizer.convert_tokens_to_ids(tok_text)

        # extract the intent
        intent = data[-1]
        pdb.set_trace()
        #TODO debug
        if tok_text_len == self._max_sequence_len - 1:
            None#print(torch.tensor(tok_idx))

        return torch.tensor(tok_idx), intent


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




        
        