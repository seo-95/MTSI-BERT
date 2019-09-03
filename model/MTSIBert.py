
import pdb
from collections import deque

import torch
import torch.nn.functional as F
from pytorch_transformers import BertModel, BertTokenizer
from torch import nn

from .MTSIBertInputBuilder import MTSITensorBuilder


class MTSIBert(nn.Module):
    """Implementation of MTSI-Bert"""
    _BERT_H_DIM = 768 
    _BERT_CLS_IDX = 101
    _BERT_SEP_IDX = 102
    _BERT_MASK_IDX = 103


    def __init__(self, num_layers_encoder, num_layers_eod,
                n_intents, batch_size, pretrained, seed, window_size):

        super(MTSIBert, self).__init__()

        torch.manual_seed(seed)
        
        self._n_intents = n_intents
        self._batch_size = batch_size

        self.__build_nn(pretrained)

        # Input dimensions
        self._window_size = window_size
        self._sliding_win = SlidingWindow(window_size)


    def __build_nn(self, pretrained):

        # architecture stack
        self._bert = BertModel.from_pretrained(pretrained)
        
        # classifiers
        self._eod_classifier = nn.Linear(in_features = MTSIBert._BERT_H_DIM,
                                        out_features = 2)
        self._intent_classifier = nn.Linear(MTSIBert._BERT_H_DIM, self._n_intents)
        self._action_classifier = nn.Linear(MTSIBert._BERT_H_DIM, 2)
        self._softmax = F.softmax
    

    def init_hiddens(self, num_windows, device):

        encoder_hidden = torch.zeros(self._encoder_num_layers, num_windows, self._encoder_hidden_dim).to(device)
        eod_hidden = torch.zeros(self._eod_num_layers, num_windows, self._eod_hidden_dim).to(device)
        return (encoder_hidden, eod_hidden)


    def forward(self, input, turns, dialogue_ids, tensor_builder: MTSITensorBuilder,\
                device='cpu'):

        """
        It works only with batch size B=1
        Input:
            input : padded tensor input of dim `1 x D_LEN x S_LEN`
            turns : the turns for the input dialogue having shape `1 x 14`
            dialogue_ids : list of string ids of len `B` (B=1)
            tensor_builder: the class for building the input tensor for BERT
            device : the device on which to work

        Output:
            prediction : tensor having shape `B x NUM_CLASSES`
            hidden : tensor having shape `NUM_LAYERS x 1 X 768`
        """
        # remove the batch dimension
        input = input.squeeze(0)
        turns = turns.squeeze(0)

        # PRE PROCESSING for obtaining windows
        # from single input sentence to window packed sentences
        windows_l = self._sliding_win.pack_windows(input, turns)
        # returns a padded tensor of dim NUM_WINDOWS x SEQ_LEN
        bert_input = tensor_builder.build_tensor(windows_l, device)
        
        attention_mask, segment_mask = tensor_builder.build_attention_and_segment(bert_input)
        attention_mask = attention_mask.to(device)
        segment_mask = segment_mask.to(device)

        ### BERT
        # only for PC debug: BERT too computationally expensive on cpu and out of mem on GPU    
        if str(device) == 'cpu':
            num_win = len(bert_input)
            seq_len = len(bert_input[0])
            bert_cls_out = torch.randn((num_win, 768))
            bert_hiddens = torch.randn((num_win, seq_len, 768))
        else:
            # bert_input.shape == [WIN_PER_DIALOGUE x WIN_LENGTH]
            bert_hiddens, bert_cls_out = self._bert(input_ids = bert_input,
                                                token_type_ids = segment_mask,
                                                attention_mask = attention_mask)
        
        ### LOGITS and predictions
        logits_eod = self._eod_classifier(bert_cls_out)
        logits_intent = self._intent_classifier(bert_cls_out[0])
        logits_action = self._action_classifier(bert_cls_out[0])

        prediction_eod = self._softmax(logits_eod, dim=0)
        prediction_intent = self._softmax(logits_intent, dim=0)
        prediction_action = self._softmax(logits_action, dim=0)
        
        return {'logit': logits_eod, 'prediction': prediction_eod},\
                {'logit': logits_intent, 'prediction': prediction_intent},\
                {'logit': logits_action, 'prediction': prediction_action}
                



class SlidingWindow():
    """
    This class is used to provide the input tensor for Bert. It implements the sliding window taking in consideration
    both the new user utterance and the previous user utterance concatenated with the agent reply to that utterance.

        [CLS] <Q(t-1)> <R(t-1)> [SEP] <Q(t)> [SEP]

    Input:
            device : the device must be the same on which both input and the MTSIBert model are.

    """

    def __init__(self, window_max_size):
        
        # init dialog status
        self._curr_window = deque()
        self._curr_dialog_id = ''
        self._window_max_size = window_max_size
        # init the turn before the firts one
        self._curr_turn = 0


    def pack_windows(self, input, turns, device='cpu'):
        """
        build the windows list of the given size
        """

        window_list = []
        for utt, turn in zip(input, turns):
            # user
            if turn == 1:
                window_list.append(self.add_to_dialog_window(utt))
            # agent
            elif turn == 2:
                # insert in the window but not use it as input (agent utterances has not to be classified)
                self.add_to_dialog_window(utt)
            elif turn == 0:
                # is dialogue padding and we discard it
                pass

        return window_list


    def add_to_dialog_window(self, utterance):

        if len(self._curr_window) < self._window_max_size:
            # append new utterance
            self._curr_window.append(utterance)
        else:
            if len(self._curr_window) == self._window_max_size:
                #remove first element
                self._curr_window.popleft()
                self._curr_window.append(utterance)

        return list(self._curr_window)


    def sliding_window_flush(self):
        """
        Flush the current dialogue window
        """
        self._curr_window.clear()
        return self._curr_window
