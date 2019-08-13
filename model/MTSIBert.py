
import pdb
from collections import deque

import torch
import torch.nn.functional as F

from pytorch_transformers import BertModel, BertTokenizer
from torch import nn

from .MTSIBertInputBuilder import MTSITensorBuilder


class MTSIBert(nn.Module):
    """Implementation of MTSI-Bert"""
    _BERT_H = 768 
    _BERT_CLS_IDX = 101
    _BERT_SEP_IDX = 102
    _BERT_MASK_IDX = 103


    def __init__(self, num_layers, n_intents, batch_size, window_length, windows_per_batch,\
                pretrained, seed, window_size):
        super(MTSIBert, self).__init__()

        torch.manual_seed(seed)

        self.num_layers = num_layers
        self.n_intents = n_intents
        self.batch_size = batch_size
        self.bert_hidden_dim = MTSIBert._BERT_H
        self.gru_hidden_dim = MTSIBert._BERT_H

        self._window_size = window_size
        self._sliding_win = SlidingWindow(window_size)

        self._window_length = window_length
        self._windows_per_batch = windows_per_batch

        # architecture stack
        self._bert = BertModel.from_pretrained(pretrained)
        # gru for eod classification
        self._gru = nn.GRU(input_size = self.bert_hidden_dim,\
                            hidden_size = self.gru_hidden_dim,\
                            num_layers = self.num_layers,\
                            batch_first = True)
        self._eod_classifier = nn.Linear(in_features = self.gru_hidden_dim,\
                                out_features = 2)
        # encoder for action and intent classification (only on the first user utterance)
        self._sentence_encoder = nn.Sequential(
                                    nn.Linear(self.bert_hidden_dim, 500),
                                    nn.ReLU(),
                                    nn.Linear(500, 300),
                                    nn.ReLU(),
                                )
        self._intent_classifier = nn.Linear(300, self.n_intents)
        self._action_classifier = nn.Linear(300, 2)
        self._softmax = nn.functional.softmax


    
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
        hidden = self.init_hidden()
        hidden = hidden.to(device)

        # PRE PROCESSING for obtaining window tensors
        # from single input sentence to window packed sentences
        windows_l = self._sliding_win.pack_windows(input, turns)
        # returns a list of tensors to be padded and aggregated
        t_list = tensor_builder.build_tensor(windows_l, device)
        bert_input = self._build_bert_input(t_list)
        
        attention_mask, segment_mask = tensor_builder.build_attention_and_segment(bert_input)
        attention_mask = attention_mask.to(device)
        segment_mask = segment_mask.to(device)

        if str(device) == 'cuda:0':
            torch.cuda.empty_cache()

        ### BERT
        # only for PC debug: BERT too computationally expensive on cpu and out of mem on GPU    
        if str(device) == 'cpu':
            bert_cls_out = torch.randn((self._windows_per_batch, 768))
            bert_hiddens = torch.randn((self._windows_per_batch, self._window_length, 768))
        else:
            # bert_input.shape == [WIN_PER_DIALOGUE x WIN_LENGTH]
            bert_hiddens, cls_out = self._bert(input_ids = bert_input,
                                                segment_mask = segment_mask,
                                                attention_mask = attention_mask)

        pdb.set_trace()

                    ### EOD GRU
        # bert_cls_out is a tensor of shape `B x W_PER_DIALOGUE x 768`
        gru_out, hidden = self._gru(bert_cls_out.unsqueeze(0), hidden)
        

        ### SENTENCE ENCODER FOR INTENT AND ACTION
        # here compute average and send to sentence encoder
        # bert_hiddens has dimension WIN_PER_DIALOGUE x WIN_LENGTH x 768
        # sentence_avg has dimension WIN_PER_DIALOGUE x WIN_LENGTH
        sentence_avg = compute_average(bert_hiddens, attention_mask)
        sentence_out = self._sentence_encoder(sentence_avg)


        ### LOGITS and predictions
        logits_eod = self._eod_classifier(gru_out)
        logits_intent = self._intent_classifier()
        logits_action = self._action_classifier()

        #logits = logits.squeeze(0) # now logits has dim `B x 3` (batch_size * num_labels)
        prediction_eod = self._softmax(logits_eod, dim=1)
        prediction_intent = self._softmax(logits_intent, dim=1)
        prediction_action = self._softmax(logits_action, dim=1)

        return prediction_eod, prediction_intent, prediction_action,\
                logits_eod, logits_intent, logits_action, hidden

                



    def _build_bert_input(self, t_list):
        # apply padding (both single utterance and dialogue padding)
        for win_idx, _ in enumerate(t_list):
            win_residual = self._window_length - len(t_list[win_idx])
            assert win_residual > 0, '[ASSERT FAILED] -- max window length not enough'
            t_list[win_idx] = F.pad(t_list[win_idx], (0, win_residual), 'constant', 0)
        # now shape will be ? x W_LENGTH : dialogue padding already to perform
        bert_input = torch.stack(t_list)
        #padding for the entire batch (dialogue)
        batch_residual = self._windows_per_batch - len(t_list)
        # now shape will be W_P_DIALOGUE x W_LENGTH 
        bert_input = F.pad(bert_input, (0, 0, 0, batch_residual), 'constant', 0)
        return bert_input




    def init_hidden(self):
        # here the second dimension is 1 because we use the hidden sequentially for each dialog window
        return torch.zeros(self.num_layers, 1, self.gru_hidden_dim)



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
        This method creates the dialogue input concatenating in the proper way the utterances

        [CLS] <Q(t-1)> <R(t-1)> [SEP] <Q(t)> [SEP]

        The window is single intent, so the turn as to be considered "cyclically" inside a single dialogue.
        When a new dialogue is detected the window is flushed and it restart with the turn counter t = 1.

        Input:
            batch : the set of dialogues in the batch having shape B x D_LEN x S_LEN
            dialogue_ids : the ids of the dialogue
        Output:
            curr_dialog_window
            segment : segment vector for the Bert model (sentence A and B)

        ------------------    
        Example with windows_size = 3:
            At turn t = 1 will have the following shape

                    [CLS] Q(t=1) [SEP]
            
            At turn t = t' | t' > 1 will have the following shape

                    [CLS] <Q(t'-1)> <R(t'-1)> [SEP] <Q(t')> [SEP]

            During training phase when a new dialogue (here called 'b') starts we have the following behaviour

                    [CLS] <Qa(t=n_a)> <Ra(t=n_a)> [SEP] <Qb(t=1)> [SEP] -> new dialogue has to be detected
                        ==> flush and come back to t=1 for the new dialogue 'b'
                    [CLS] Qb(t=1) [SEP]

                (in the above example n_a is the length of dialog 'a')
                    
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
