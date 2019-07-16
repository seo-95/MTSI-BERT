
import torch
import pdb
from torch import nn
from pytorch_transformers import BertModel, BertTokenizer


class MTSIBert(nn.Module):
    """Implementation of MTSI-Bert"""
    _BERT_H = 768 

    def __init__(self, num_layers, n_labels, batch_size, pretrained):
        super(MTSIBert, self).__init__()

        self.num_layers = num_layers
        self.n_labels = n_labels
        self.batch_size = batch_size
        self.bert_hidden_dim = MTSIBert._BERT_H
        self.gru_hidden_dim = MTSIBert._BERT_H

        # stack architecture
        self._bert = BertModel.from_pretrained(pretrained)
        self._gru = nn.GRU(input_size = self.bert_hidden_dim,\
                            hidden_size = self.gru_hidden_dim,\
                            num_layers = self.num_layers,\
                            batch_first = True)
        self._classifier = nn.Linear(in_features = self.gru_hidden_dim,\
                                out_features = self.n_labels)
        self._softmax = nn.functional.softmax


    def forward(self, input, segments, hidden):

        hidden_states, cls_out = self._bert(input, segments)
        # cls_out = batch_sizex768
        cls_out = cls_out.unsqueeze(0)

        gru_out, hidden = self._gru(cls_out, hidden)
        logits = self._classifier(gru_out)

        # TODO try it with batch_size > 1
        logits = logits.squeeze(1) # now logits has dim 1x3 (batch_size * num_labels)
        pdb.set_trace()
        prediction = self._softmax(logits, dim=1)

        return prediction, hidden


    def init_hidden(self):

        return torch.zeros(self.num_layers, self.batch_size, self.gru_hidden_dim)