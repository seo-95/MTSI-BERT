
import torch
import pdb
from torch import nn
from pytorch_transformers import BertModel, BertTokenizer


class MTSIBert(nn.Module):
    """Implementation of MTSI-Bert"""
    _BERT_H = 768 
    _BERT_CLS_IDX = 101
    _BERT_SEP_IDX = 102
    _BERT_MASK_IDX = 103

    def __init__(self, num_layers, n_labels, batch_size, pretrained, seed):
        super(MTSIBert, self).__init__()

        torch.manual_seed(seed)

        self.num_layers = num_layers
        self.n_labels = n_labels
        self.batch_size = batch_size
        self.bert_hidden_dim = MTSIBert._BERT_H
        self.gru_hidden_dim = MTSIBert._BERT_H

        # init dialog status
        self._curr_dialog_window = []
        self._curr_dialog_id = ''

        # architecture stack
        self._bert = BertModel.from_pretrained(pretrained)
        self._gru = nn.GRU(input_size = self.bert_hidden_dim,\
                            hidden_size = self.gru_hidden_dim,\
                            num_layers = self.num_layers,\
                            batch_first = True)
        self._classifier = nn.Linear(in_features = self.gru_hidden_dim,\
                                out_features = self.n_labels)
        self._softmax = nn.functional.softmax        


    
    def forward(self, input, hidden, dialogue_ids):

        """
        Input:
            input : padded tensor input of dim `B x S`
            labels : tensor of dim `1 x B`
            hidden : the hidden state for the model
            dialogue_ids : list of string ids of len `B`
        """
        cls_batch = []
        for curr_sentence, curr_dialog in zip(input, dialogue_ids):
            bert_input, segments = self.add_to_dialog_window(curr_sentence)

            hidden_states, cls_out = self._bert(input_ids = bert_input.unsqueeze(0),\
                                                 token_type_ids = segments.unsqueeze(0))
            # cls_out = batch_sizex768
            cls_out = cls_out.unsqueeze(0)
            cls_batch.append(cls_out)

            #if end of dialog then flush the window
            if self._curr_dialog_id and curr_dialog != self._curr_dialog_id:
                self.dialogue_input_flush()
                # here re-insert the last sentence (the first of the new dialogue)
                bert_input, _ = self.add_to_dialog_window(curr_sentence)
            self._curr_dialog_id = curr_dialog

        # cls_batch is a list of list having len `B`. Interl lists length is 768
        gru_input = torch.stack(cls_batch).squeeze(1).squeeze(1).unsqueeze(0)
        # gru input is a tensor of shape `1 x B x 768`
        gru_out, hidden = self._gru(gru_input, hidden)
        logits = self._classifier(gru_out)

        logits = logits.squeeze(0) # now logits has dim `B x 3` (batch_size * num_labels)
        prediction = self._softmax(logits, dim=1)

        return prediction, hidden






    def init_hidden(self):
        # here the second dimension is 1 because we use the hidden sequentially for each dialog window
        return torch.zeros(self.num_layers, 1, self.gru_hidden_dim)



    def add_to_dialog_window(self, input):
        """
        This method creates the dialogue input concatenating in the proper way the utterances

        Input:
            input : the set of utterances to append to the dialogue window
        Output:
            curr_dialog_window
            segment : segment vector for the Bert model (sentence A and B)

        ------------------    
        Example:
            a dialogue composed of 3 utterance at turn = 3 will have the following shape
                    concat(U1, U2) [SEP] U3

        """

        has_more_sentences = False
        # append [CLS] at the beginning
        if(len(self._curr_dialog_window) == 0):
            self._curr_dialog_window = torch.tensor(MTSIBert._BERT_CLS_IDX).reshape(1)
        else:
            has_more_sentences = True
            # remove the old separator
            self._curr_dialog_window = self._curr_dialog_window[self._curr_dialog_window != MTSIBert._BERT_SEP_IDX]
            #append the new separator
            self._curr_dialog_window = torch.cat((self._curr_dialog_window,\
                                                 torch.tensor(MTSIBert._BERT_SEP_IDX).reshape(1)))
        self._curr_dialog_window = torch.cat((self._curr_dialog_window, input))

        #remove padding
        self._curr_dialog_window = self._curr_dialog_window[self._curr_dialog_window != 0]

        #compute the segment for one or multiple sentences
        segment = torch.zeros(len(self._curr_dialog_window), dtype=torch.long)
        if has_more_sentences:
            for idx in reversed(range(len(self._curr_dialog_window))): #TODO avoid to overwrite pos 0
                if self._curr_dialog_window[idx] != MTSIBert._BERT_SEP_IDX:
                    segment[idx] = 1 
                else:
                    break
        
        return self._curr_dialog_window, segment


    def dialogue_input_flush(self):
        """
        Flush the current dialogue window
        """
        self._curr_dialog_window = []
        return self._curr_dialog_window





