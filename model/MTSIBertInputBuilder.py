
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

    def get_attention_and_segment(self, bert_input: torch.Tensor) -> (torch.Tensor, torch.Tensor):
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
    def build_tensor(self, windows_list, device):

        # the result will be a list of lists with a number of entries equal to the number of different dialogues 
        res_l = []
        sep_tensor = torch.tensor(MTSITensorBuilder._BERT_SEP_IDX).reshape(1).to(device)

        for win in windows_list:
            # at first append [CLS]
            curr_builded_t = torch.tensor(MTSITensorBuilder._BERT_CLS_IDX).reshape(1).to(device)
            for idx, t in enumerate(win):
                # remove padding for each utterance
                no_padded = self.__remove_padding(t)
                if idx == len(win) - 1:
                    # if only one tensor then append <t[SEP]>
                    if idx == 0:
                        curr_builded_t = torch.cat((curr_builded_t, no_padded, sep_tensor))
                    # else if last one append <[SEP]t[SEP]>
                    else:
                        curr_builded_t = torch.cat((curr_builded_t, sep_tensor,\
                                        no_padded, sep_tensor))
                else:
                    curr_builded_t = torch.cat((curr_builded_t, no_padded))
            
            # for count windows length
            #with open('booh.txt', 'a') as f:
            #    f.write(str(len(curr_builded_t))+'\n')
            res_l.append(curr_builded_t)

        return res_l


    def __remove_padding(self, t):
        return t[t!=0]


    def build_attention_and_segment(self, bert_input):

        attention_mask = []
        segment_mask = []

        for idx, t in enumerate(bert_input):

            # build attention
            tmp_attention = torch.zeros(len(t))
            non_zero = len(t[t!=0])
            tmp_attention[:non_zero] = 1

            # build segment
            tmp_segment = torch.zeros(len(t))
            # if is the first window then we have only 1 sentence. Avoid also dialogue padding tensors
            if idx != 0 and t[0] != 0:
                first_segment_end = self.__find_first_occurrence(t, MTSITensorBuilder._BERT_SEP_IDX)
                tmp_segment[first_segment_end+1:] = 1

            attention_mask.append(tmp_attention)
            segment_mask.append(tmp_segment)
        pdb.set_trace()
        return torch.stack(attention_mask), torch.stack(segment_mask)


    def __find_first_occurrence(self, t, token):
        for idx, v in enumerate(t):
            if v == token:
                return idx



            
