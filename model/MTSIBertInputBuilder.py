
import pdb
from abc import ABC

import torch


"""
This module contains some example of tensor builder. A tensor builder is a class that implements 
the abstract class 'TensorBuilder' for build the input for Bert in several different ways. 
"""

class MTSITensorBuilder(ABC):

    _BERT_CLS_IDX = 101
    _BERT_SEP_IDX = 102
    _BERT_MASK_IDX = 103
    
    def build_tensor(self, dialogue_windows_list: list, device) -> list:
        """
        This method receives a list of tensors and return a list of tensors of the same size of the given one by 
        concatenating them in the way you choose for the Bert input.

        A dialogue window list is a list with as many entries as dialogues. Each entry is composed by a list of windows,
        each window is a list of tensor.

        dialogue_window_list([dialogue[window[tensor]]])
        """

        raise NotImplementedError('Abstract class method not implemented')

    def get_attention_and_toktypeids(self, bert_input: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        This method receives a tensor (probably the one produced by build_tensor) and returns the attention mask and
        token_type_ids tensor for BERT
        """

        raise NotImplementedError('Abstract class method not implemented')



class TwoSepTensorBuilder(MTSITensorBuilder):
    """
    Result shape:
        [CLS] T(1) T(2) ... T(n-1) [SEP] T(n) [SEP]
    """
    def build_tensor(self, dialogue_windows_list, device):

        # the result will be a list of lists with a number of entries equal to the number of different dialogues 
        res_l = []
        sep_tensor = torch.tensor(MTSITensorBuilder._BERT_SEP_IDX).reshape(1).to(device)
        for dialogue_idx, win_list in enumerate(dialogue_windows_list):
            curr_dial_l = []
            for win in win_list:
                # at first append [CLS]
                curr_builded_t = torch.tensor(MTSITensorBuilder._BERT_CLS_IDX).reshape(1).to(device)
                for idx, t in enumerate(win):
                    if idx == len(win) - 1:
                        # if only one tensor then append <t[SEP]>
                        if idx == 0:
                            curr_builded_t = torch.cat((curr_builded_t, t, sep_tensor))
                        # else if last one append <[SEP]t[SEP]>
                        else:
                            curr_builded_t = torch.cat((curr_builded_t, sep_tensor,\
                                            t, sep_tensor))
                    else:
                        curr_builded_t = torch.cat((curr_builded_t, t))
                curr_dial_l.append(curr_builded_t)
            
            res_l.append(curr_dial_l)
        return res_l


    def build_attention_and_toktypeids(self, bert_input):

        """
        returns the attention and token_type_ids tensor for the bert input
        """
 
        tok_type_l = []
        attention_l = []
        for batch in bert_input:
            curr_dialogue_tokt = []
            curr_dialogue_attention = []
            for win_idx, win in enumerate(batch):
                # the first windows is composed by only one sentence
                # in addition avoid computations for padding dialogue windows
                curr_win_attention = self.__build_attention(win)
                if win_idx == 0 or win[0] == 0:
                    curr_tok_type_ids = torch.zeros(len(win))
                    curr_dialogue_tokt.append(curr_tok_type_ids)
                    curr_dialogue_attention.append(curr_win_attention)
                    continue
                # search the second and first [SEP]
                sep_idx = None
                for idx, _ in enumerate(win):
                    if win[idx] == self._BERT_SEP_IDX:
                        sep_idx = idx
                        break
                assert sep_idx != None, '[ASSERT FAILED] -- sep missing'
                curr_tok_type_ids = torch.zeros(len(win))
                
                curr_tok_type_ids[sep_idx+1:] = 1
                curr_dialogue_tokt.append(curr_tok_type_ids)
                curr_dialogue_attention.append(curr_win_attention)
            tok_type_l.append(curr_dialogue_tokt)
            attention_l.append(curr_dialogue_attention)

            # build tensor from list of lists
            tok_type_tensor = None
            attention_tensor = None
            for t, a in zip(tok_type_l, attention_l):
                tmp_tok = torch.stack(t).unsqueeze(0)
                tmp_att = torch.stack(a).unsqueeze(0)
                if tok_type_tensor is None and attention_tensor is None:
                    tok_type_tensor = tmp_tok
                    attention_tensor = tmp_att
                else:
                    
                    tok_type_tensor = torch.cat((tok_type_tensor, tmp_tok), dim=0)
                    attention_tensor = torch.cat((attention_tensor, tmp_att), dim=0)

        return tok_type_tensor, attention_tensor


    def __build_attention(self, win):
        
        attention_t = torch.zeros(len(win))
        for idx, _ in enumerate(win):
            if win[idx] != 0:
                attention_t[idx] = 1

        return attention_t


            
