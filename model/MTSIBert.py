
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

    def __init__(self, num_layers, n_labels, batch_size, pretrained):
        super(MTSIBert, self).__init__()

        self.num_layers = num_layers
        self.n_labels = n_labels
        self.batch_size = batch_size
        self.bert_hidden_dim = MTSIBert._BERT_H
        self.gru_hidden_dim = MTSIBert._BERT_H

        # init dialog status
        self._curr_dialog = []

        # architecture stack
        self._bert = BertModel.from_pretrained(pretrained)
        self._gru = nn.GRU(input_size = self.bert_hidden_dim,\
                            hidden_size = self.gru_hidden_dim,\
                            num_layers = self.num_layers,\
                            batch_first = True)
        self._classifier = nn.Linear(in_features = self.gru_hidden_dim,\
                                out_features = self.n_labels)
        self._softmax = nn.functional.softmax        


    
    def forward(self, input, labels, hidden, segments):

        """
        Input:
            input : padded tensor input of dim `B x S`
            labels : tensor of dim `1 x B`
            hidden : the hidden state for the model
            segment : the segment vector for the input
            mask : tensor of dim `B x S`
            dialogue_ids : list of string ids of len `B`
        """
        

        hidden_states, cls_out = self._bert(input, segments)
        # cls_out = batch_sizex768
        cls_out = cls_out.unsqueeze(0)

        gru_out, hidden = self._gru(cls_out, hidden)
        logits = self._classifier(gru_out)

        # TODO try it with batch_size > 1
        logits = logits.squeeze(1) # now logits has dim 1x3 (batch_size * num_labels)
        #pdb.set_trace()
        prediction = self._softmax(logits, dim=1)

        return prediction, hidden






    def init_hidden(self):

        return torch.zeros(self.num_layers, self.batch_size, self.gru_hidden_dim)



    def dialogue_input_generator(self, input, turn):
        """
        This method creates the dialogue input concatenating in the proper way the utterances

        Input:
            input : the set of utterances in the dialogue
            turn : the actual turn of the dialogue

        Output:
            curr_dialog
            segment : segment vector for the Bert model (sentence A and B)

        ------------------    
        Example:
            a dialogue composed of 3 utterance at turn = 3 will have the following shape
                    concat(U1, U2) [SEP] U3

        """
        has_more_sentences = False
        # append [CLS] at the beginning
        if(len(self._curr_dialog) == 0):
            self._curr_dialog = torch.tensor(MTSIBert._BERT_CLS_IDX).reshape(1)
        else:
            has_more_sentences = True
            # remove the old separator
            self._curr_dialog = self._curr_dialog[self._curr_dialog != MTSIBert._BERT_SEP_IDX]
            #append the new separator
            self._curr_dialog = torch.cat((self._curr_dialog, torch.tensor(MTSIBert._BERT_SEP_IDX).reshape(1)))
        self._curr_dialog = torch.cat((self._curr_dialog, input[turn]))

        #remove padding
        self._curr_dialog = self._curr_dialog[self._curr_dialog != 0]

        #compute the segment for one or multiple sentences
        segment = torch.zeros(len(self._curr_dialog), dtype=torch.long)
        if has_more_sentences:
            for idx in reversed(range(len(self._curr_dialog))): #TODO avoid to overwrite pos 0
                if self._curr_dialog[idx] != MTSIBert._BERT_SEP_IDX:
                    segment[idx] = 1 
                else:
                    break

        return self._curr_dialog, segment


    def dialogue_input_flush(self):
        """
        Flush the current dialogue window
        """
        self._curr_dialog = []
        return self._curr_dialog





