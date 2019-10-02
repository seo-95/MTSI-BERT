
import pdb

import torch
import torch.nn.functional as F
from torch import nn


class BaseLine(nn.Module):


    def __init__(self, num_layers_encoder, num_layers_eod, n_intents, seed):

        super(BaseLine, self).__init__()

        torch.manual_seed(seed)
        
        self._n_intents = n_intents

        # sentence encoder parameters
        self._encoder_num_layers = num_layers_encoder
        self._encoder_input_dim = 300
        self._encoder_hidden_dim = 1024

        # build nn stack
        self.__build_eos_nn()
        self.__build_intent_nn()
        self.__build_action_nn()

        self._softmax = F.softmax



    def __build_eos_nn(self):
        """
        Architecture stack
        """
        
        # sentence encoder for action and intent
        self._eos_encoderbiLSTM = nn.LSTM(self._encoder_input_dim,
                                    self._encoder_hidden_dim,
                                    num_layers=self._encoder_num_layers,
                                    batch_first=True,
                                    bidirectional=True)

        # EOS FFNN
        self._eos_ffnn = nn.Sequential(nn.Linear(3*2*self._encoder_hidden_dim, 3*2*self._encoder_hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(3*2*self._encoder_hidden_dim, 2*self._encoder_hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(2*self._encoder_hidden_dim, 768),
                                        nn.ReLU())
    
        
        # classifiers
        self._eos_classifier = nn.Linear(in_features = 768,
                                        out_features = 2)


    def __build_intent_nn(self):
        """
        Architecture stack
        """
        
        # sentence encoder for action and intent
        self._intent_encoderbiLSTM = nn.LSTM(self._encoder_input_dim,
                                            self._encoder_hidden_dim,
                                            num_layers=self._encoder_num_layers,
                                            batch_first=True,
                                            bidirectional=True)

        # INTENT FFNN
        self._intent_ffnn = nn.Sequential(nn.Linear(2*self._encoder_hidden_dim, 2*self._encoder_hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(2*self._encoder_hidden_dim, self._encoder_hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(self._encoder_hidden_dim, 768),
                                        nn.ReLU())
    
        
        # classifiers
        self._intent_classifier = nn.Linear(in_features = 768,
                                            out_features = self._n_intents)


    def __build_action_nn(self):
        """
        Architecture stack
        """
        
        # sentence encoder for action and intent
        self._action_encoderbiLSTM = nn.LSTM(self._encoder_input_dim,
                                            self._encoder_hidden_dim,
                                            num_layers=self._encoder_num_layers,
                                            batch_first=True,
                                            bidirectional=True)

        # ACTION FFNN
        self._action_ffnn = nn.Sequential(nn.Linear(2*self._encoder_hidden_dim, 2*self._encoder_hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(2*self._encoder_hidden_dim, self._encoder_hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(self._encoder_hidden_dim, 768),
                                        nn.ReLU())

        
        # classifiers
        self._action_classifier = nn.Linear(in_features = 768,
                                            out_features = 2)
        
    



    def forward(self, input, turns, dialogue_ids, seqs_len, device='cpu'):

        """
        It works only with batch size B=1
        Input:
            input : padded tensor input of dim `1 x D_LEN x S_LEN`
            turns : the turns for the input dialogue having shape `1 x 14`
            dialogue_ids : list of string ids of len `B` (B=1)
            device : the device on which to work

        Output:
            prediction : tensor having shape `B x NUM_CLASSES`
            hidden : tensor having shape `NUM_LAYERS x 1 X 768`
        """
        # remove the batch dimension
        input = input.squeeze(0)
        turns = turns.squeeze(0)
        seqs_len = seqs_len.squeeze(0)

        
        eos_res = self._eos_compute(input, seqs_len)
        intent_res = self._intent_compute(input[0].unsqueeze(0), seqs_len[0].unsqueeze(0))
        action_res = self._action_compute(input[0].unsqueeze(0), seqs_len[0].unsqueeze(0))

        return eos_res, intent_res, action_res





    def _eos_compute(self, input, seqs_len):

         # ENCODE the sentence
        packed_encoder_input = torch.nn.utils.rnn.pack_padded_sequence(input, seqs_len,
                                                                        batch_first=True, enforce_sorted=False)

        # hidden shape == NUM_LAYERS * NUM_DIRECTIONS x BATCH x HIDDEN_SIZE
        packed_out, (hidden, cell) = self._eos_encoderbiLSTM(packed_encoder_input)
        last_state_forward = hidden[self._encoder_num_layers-1, :, :]
        last_state_backward = hidden[2*self._encoder_num_layers-1, :, :]
        # now concatenate the last of forward and the last of backward
        enc_sentence = torch.cat((last_state_forward, last_state_backward), dim=1)
        
        # group windows of size 3
        win_l = []
        for idx in range(0, len(enc_sentence), 2):
            if idx + 3 > len(enc_sentence):
                break
            win_l.append(enc_sentence[idx:idx+3])
        win_tensor = torch.stack(win_l)

        # concatenate internal window items
        win_tensor = win_tensor.view(-1, 2048*3)

        # FFNN
        eos_out = self._eos_ffnn(win_tensor)

        ### LOGITS and predictions
        logits_eos = self._eos_classifier(eos_out)
        
        prediction_eos = self._softmax(logits_eos, dim=1)

        return {'logit': logits_eos, 'prediction': prediction_eos}



    def _intent_compute(self, input, seq_len):
        
        # ENCODE the sentence
        packed_encoder_input = torch.nn.utils.rnn.pack_padded_sequence(input, seq_len, batch_first=True, enforce_sorted=False)
        
        # hidden shape == NUM_LAYERS * NUM_DIRECTIONS x BATCH x HIDDEN_SIZE
        packed_out, (hidden, cell) = self._intent_encoderbiLSTM(packed_encoder_input)
        last_state_forward = hidden[self._encoder_num_layers-1, :, :]
        last_state_backward = hidden[2*self._encoder_num_layers-1, :, :]
        # now concatenate the last of forward and the last of backward
        enc_sentence = torch.cat((last_state_forward, last_state_backward), dim=1)

        # FFNN
        intent_out = self._intent_ffnn(enc_sentence)
        
        ### LOGITS and predictions
        logits_intent = self._intent_classifier(intent_out)
        
        prediction_intent = self._softmax(logits_intent, dim=1)

        return {'logit': logits_intent, 'prediction': prediction_intent}



    def _action_compute(self, input, seq_len):
        
        # ENCODE the sentence
        packed_encoder_input = torch.nn.utils.rnn.pack_padded_sequence(input, seq_len, batch_first=True, enforce_sorted=False)
        
        # hidden shape == NUM_LAYERS * NUM_DIRECTIONS x BATCH x HIDDEN_SIZE
        packed_out, (hidden, cell) = self._action_encoderbiLSTM(packed_encoder_input)
        last_state_forward = hidden[self._encoder_num_layers-1, :, :]
        last_state_backward = hidden[2*self._encoder_num_layers-1, :, :]
        # now concatenate the last of forward and the last of backward
        enc_sentence = torch.cat((last_state_forward, last_state_backward), dim=1)

        # FFNN
        action_out = self._action_ffnn(enc_sentence)
        
        ### LOGITS and predictions
        logits_action = self._action_classifier(action_out)
        
        prediction_action = self._softmax(logits_action, dim=1)

        return {'logit': logits_action, 'prediction': prediction_action}