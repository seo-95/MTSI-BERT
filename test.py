import datetime
import logging
import os
import pdb
import sys

import numpy as np
import torch
from sklearn.metrics import classification_report
from torch import nn
from torch.utils.data import DataLoader

from model import (KvretConfig, KvretDataset, BLAdapterDataset, BaseLine,
                   BaselineKvretConfig)



def remove_dataparallel(load_checkpoint_path):
    # original saved file with DataParallel
    state_dict = torch.load(load_checkpoint_path)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    # load params
    return new_state_dict


def compute_f1(model, data_generator, device):

    # initializes statistics
    true_eos = []
    pred_eos = []
    true_action = []
    pred_action = []
    true_intent = []
    pred_intent = []


    with torch.no_grad():
        for local_batch, local_turns, seqs_len, local_intents, local_actions, dialogue_ids in data_generator:
            
            # local_batch.shape == B x D_LEN x U_LEN
            # local_intents.shape == B
            # local_actions.shape == B
            # local_eod_label.shape == B x D_PER_WIN
            local_batch = local_batch.to(device)
            local_intents = local_intents.to(device)
            local_actions = local_actions.to(device)

            eos, intent, action = model(local_batch,
                                        local_turns, 
                                        dialogue_ids,
                                        seqs_len,
                                        device)

            eos_label = torch.tensor([0]*eos['logit'].shape[0]).to(device)
            eos_label[-1] = 1                            

            # take the predicted label
            eos_predicted = torch.argmax(eos['prediction'], dim=-1)
            action_predicted = torch.argmax(action['prediction'], dim=-1)
            intent_predicted = torch.argmax(intent['prediction'], dim=-1)
            true_eos += eos_label.tolist()
            pred_eos += eos_predicted.tolist()
            true_action += local_actions.tolist()
            pred_action.append(action_predicted.item())
            true_intent += local_intents.tolist()
            pred_intent.append(intent_predicted.item())

    
    print('--EOD score:')
    print(classification_report(true_eos, pred_eos, target_names=['NON-EOD', 'EOD']))
    print('--Action score:')
    print(classification_report(true_action, pred_action, target_names=['FETCH', 'INSERT']))
    print('--Intent score:')
    print(classification_report(true_intent, pred_intent, target_names=['SCHEDULE', 'WEATHER', 'NAVIGATE']))




def test(load_checkpoint_path):
    """
    Test utility
    """
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print('active device = '+str(device))
 

    # Model preparation
    model = BaseLine(num_layers_encoder = BaselineKvretConfig._ENCODER_LAYERS_NUM,
                    num_layers_eod = BaselineKvretConfig._EOD_LAYERS_NUM,
                    n_intents = BaselineKvretConfig._N_INTENTS,
                    seed = BaselineKvretConfig._SEED)
    # work on multiple GPUs when availables
    if torch.cuda.device_count() > 1:
        print('active devices = '+str(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    print('model loaded from: '+load_checkpoint_path)
    model.load_state_dict(torch.load(load_checkpoint_path))
    #new_state_dict = remove_dataparallel(load_checkpoint_path)
    #model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()


    # Parameters
    params = {'batch_size': BaselineKvretConfig._BATCH_SIZE,
            'shuffle': False,
            'num_workers': 0}

    # f1-score on test set
    test_set = KvretDataset(KvretConfig._KVRET_TEST_PATH)
    test_set.remove_subsequent_actor_utterances()
    badapter_test = BLAdapterDataset(test_set, 
                                    KvretConfig._KVRET_MAX_BERT_TOKENS_PER_TRAIN_SENTENCE + 1,
                                    KvretConfig._KVRET_MAX_BERT_SENTENCES_PER_TRAIN_DIALOGUE+2)
    test_generator = DataLoader(badapter_test, **params)
    print('### TEST SET:')
    compute_f1(model, test_generator, device)


    # f1-score on validation set
    val_set = KvretDataset(KvretConfig._KVRET_VAL_PATH)
    val_set.remove_subsequent_actor_utterances()
    badapter_val = BLAdapterDataset(val_set, 
                                    KvretConfig._KVRET_MAX_BERT_TOKENS_PER_TRAIN_SENTENCE + 1,
                                    KvretConfig._KVRET_MAX_BERT_SENTENCES_PER_TRAIN_DIALOGUE+2)
    val_generator = DataLoader(badapter_val, **params)
    print('### VALIDATION SET:')
    compute_f1(model, val_generator, device)






if __name__ == '__main__':
    test(load_checkpoint_path='../dict_archive/baseline/state_dict.pt')
