
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
                            #TODO pad until 345 (114*3 +3tokens=342+3=345)
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
