
import pdb
from collections import deque

import torch
from pytorch_transformers import BertModel, BertTokenizer
from torch import nn


class MTSIBert(nn.Module):
    """Implementation of MTSI-Bert"""
    _BERT_H = 768 
    _BERT_CLS_IDX = 101
    _BERT_SEP_IDX = 102
    _BERT_MASK_IDX = 103


    def __init__(self, num_layers, n_labels, batch_size, pretrained, seed, window_size):
        super(MTSIBert, self).__init__()

        torch.manual_seed(seed)

        self.num_layers = num_layers
        self.n_labels = n_labels
        self.batch_size = batch_size
        self.bert_hidden_dim = MTSIBert._BERT_H
        self.gru_hidden_dim = MTSIBert._BERT_H

        self._sliding_win = SlidingWindow(window_size)

        # architecture stack
        self._bert = BertModel.from_pretrained(pretrained)
        self._gru = nn.GRU(input_size = self.bert_hidden_dim,\
                            hidden_size = self.gru_hidden_dim,\
                            num_layers = self.num_layers,\
                            batch_first = True)
        self._classifier = nn.Linear(in_features = self.gru_hidden_dim,\
                                out_features = self.n_labels)
        self._softmax = nn.functional.softmax        


    
    def forward(self, input, hidden, turns, dialogue_ids, persistence = False, device='cpu'):

        """
        Input:
            input : padded tensor input of dim `B x D_LEN x S_LEN`
            hidden : the hidden state for the model
            dialogue_ids : list of string ids of len `B`
            device : the device on which to work

        Output:
            prediction : tensor having shape `B x NUM_CLASSES`
            hidden : tensor having shape `NUM_LAYERS x 1 X 768`
        """
        cls_batch = []
        new_dialogue = False


        # TRY
        self._sliding_win.pack_dialogs(input, dialogue_ids, turns, persistence)
        #TODO here call the tensor builder


        for curr_dialogue, curr_id in zip(input, dialogue_ids):
            for curr_sentence in curr_dialogue:
                pdb.set_trace()

                if new_dialogue:
                    #TODO here set the EOD label and flush the window
                    new_dialogue = False

                bert_input, segments = self.add_to_dialog_window(curr_sentence, device)
                #print(bert_input.shape)
                hidden_states, cls_out = self._bert(input_ids = bert_input.unsqueeze(0),\
                                                    token_type_ids = segments.unsqueeze(0))
                # cls_out = batch_sizex768
                cls_out = cls_out.unsqueeze(0)
                cls_batch.append(cls_out)

                #if end of dialog then flush the window
                if self._curr_dialog_id and curr_dialog != self._curr_dialog_id:
                    self.dialogue_input_flush()
                    # here re-insert the last sentence (the first of the new dialogue)
                    bert_input, _ = self.add_to_dialog_window(curr_sentence, device)
                self._curr_dialog_id = curr_dialog
            new_dialogue = True


        # cls_batch is a list of list having len `B`. Interl lists length is 768
        gru_input = torch.stack(cls_batch).squeeze(1).squeeze(1).unsqueeze(0)
        # gru input is a tensor of shape `1 x B x 768`
        gru_out, hidden = self._gru(gru_input, hidden)
        logits = self._classifier(gru_out)

        logits = logits.squeeze(0) # now logits has dim `B x 3` (batch_size * num_labels)
        prediction = self._softmax(logits, dim=1)

        return prediction, logits, hidden
        





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


    def pack_dialogs(self, batch, dialogue_ids, batch_turns, persistence=False, device='cpu'):
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

        batch_pack = []
        for idx, (curr_dialogue, dialogue_turns) in enumerate(zip(batch, batch_turns)):
            curr_id = dialogue_ids[idx]
            window_list = []
            for utt, turn in zip(curr_dialogue, dialogue_turns):
                # user
                if turn == 1:
                    window_list.append(self.add_to_dialog_window(utt))
                else:
                    # agent
                    if turn == 2:
                        # insert in the window but not use it as input (agent utterances has not to be classified)
                        self.add_to_dialog_window(utt)
                    # else turn == 0 is dialogue padding and we discard it

            # if persistence enabled and not last dialogue:
            if persistence and idx < len(batch) - 1:
                # then add the first utterance of the next dialogue as last turn
                next_dialogue_utt = batch[idx+1][0]
                window_list.append(self.add_to_dialog_window(next_dialogue_utt))

            batch_pack.append({'id': curr_id,\
                                'windows': window_list})
            self.sliding_window_flush()

        #TODO here return !


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
                





    """

        # ------------- OLD ----------------


        has_more_sentences = False
        # append [CLS] at the beginning
        if(len(self._curr_dialog_window) == 0):
            self._curr_dialog_window = torch.tensor(MTSIBert._BERT_CLS_IDX).reshape(1).to(device)
        else:
            has_more_sentences = True
            # remove the old separator
            self._curr_dialog_window = self._curr_dialog_window[self._curr_dialog_window != MTSIBert._BERT_SEP_IDX]
            #append the new separator
            self._curr_dialog_window = torch.cat((self._curr_dialog_window,\
                                                 torch.tensor(MTSIBert._BERT_SEP_IDX).reshape(1).to(device)))

        self._curr_dialog_window = torch.cat((self._curr_dialog_window, input))

        #remove padding
        self._curr_dialog_window = self._curr_dialog_window[self._curr_dialog_window != 0]

        #compute the segment for one or multiple sentences
        segment = torch.zeros(len(self._curr_dialog_window), dtype=torch.long).to(device)
        if has_more_sentences:
            for idx in reversed(range(len(self._curr_dialog_window))): #TODO avoid to overwrite pos 0
                if self._curr_dialog_window[idx] != MTSIBert._BERT_SEP_IDX:
                    segment[idx] = 1 
                else:
                    break
        
        return self._curr_dialog_window, segment
    """


    def sliding_window_flush(self):
        """
        Flush the current dialogue window
        """
        self._curr_window.clear()
        return self._curr_window
