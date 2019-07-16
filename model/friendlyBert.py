from pytorch_transformers import BertTokenizer
from torch.utils.data import Dataset
import torch
import pdb





class FriendlyBert(Dataset):
    """
    FriendlyBert is a module implementing the adapter pattern used as intermediary between you program and the dataset when using Bert.
    It performs all the operations to adapt your dataset input to the one that Bert expects. It does the tokenization, indices transformation etc.
    
    Input:
        dataset : A class extending Dataset containing you dataset
        pretrained : The string of one of the pretrained modes. For more info look at tokenization.py
        do_lower_case : Set to False if pretrained is cased. (Default:False)

    Output:
        tok_ids : Indices of each token detected byt the Bert tokenizer
        intent : the intent index for that utterance

    Note:
        The intent should be the last returned elements in the tuple provided by the __getitem__() of the dataset you provide
    """

    def __init__(self, dataset, pretrained, do_lower_case = True):
        self._dataset = dataset
        self._tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case = do_lower_case)



    def __len__(self):
        return self._dataset.__len__()


    def __getitem__(self, idx):
        data = self._dataset.__getitem__(idx)

        # prepare the utterance
        tok_text = self._tokenizer.tokenize(data[0])
        tok_idx = self._tokenizer.convert_tokens_to_ids(tok_text)

        #TODO padding

        # extract the intent
        intent = data[-1]


        pdb.set_trace()
        return torch.tensor(tok_idx[0:6]), intent
        
        