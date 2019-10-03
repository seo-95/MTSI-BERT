import datetime
import logging
import os
import pdb
import sys

import numpy as np
import torch
from pytorch_transformers import BertTokenizer
from sklearn.metrics import classification_report, recall_score, f1_score, precision_score
from torch import nn
from torch.utils.data import DataLoader

from model import (KvretConfig, KvretDataset, MTSIAdapterDataset, MTSIBert,
                   MTSIKvretConfig, TwoSepTensorBuilder)



def get_eod(turns, win_size, windows_per_dialogue):
    
    res = torch.zeros((len(turns), windows_per_dialogue), dtype=torch.long)
    user_count = 0
    for idx, curr_dial in enumerate(turns):
        for t in curr_dial:
            if t == 1:
                user_count += 1
        res[idx][user_count-1] = 1

    return res, user_count-1


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
    true_eod = []
    pred_eod = []
    true_action = []
    pred_action = []
    true_intent = []
    pred_intent = []

    tensor_builder = TwoSepTensorBuilder()

    #model.eval()
    with torch.no_grad():
        for local_batch, local_turns, local_intents, local_actions, dialogue_ids in data_generator:
            
            # 0 = intra dialogue ; 1 = eod
            eod_label, eod_idx = get_eod(local_turns, MTSIKvretConfig._WINDOW_SIZE,\
                                windows_per_dialogue=KvretConfig._KVRET_MAX_USER_SENTENCES_PER_TRAIN_DIALOGUE + 2)
            
            # local_batch.shape == B x D_LEN x U_LEN
            # local_intents.shape == B
            # local_actions.shape == B
            # local_eod_label.shape == B x D_PER_WIN
            local_batch = local_batch.to(device)
            local_intents = local_intents.to(device)
            local_actions = local_actions.to(device)
            eod_label = eod_label.to(device)

            eod, intent, action = model(local_batch,
                                        local_turns, 
                                        dialogue_ids,
                                        tensor_builder,
                                        device)
            
            # take the predicted label
            eod_predicted = torch.argmax(eod['prediction'], dim=-1)
            action_predicted = torch.argmax(action['prediction'], dim=-1)
            intent_predicted = torch.argmax(intent['prediction'], dim=-1)
            true_eod += eod_label[0][:eod_idx+1].tolist()
            pred_eod += eod_predicted.tolist()
            true_action += local_actions.tolist()
            pred_action.append(action_predicted.item())
            true_intent += local_intents.tolist()
            pred_intent.append(intent_predicted.item())

    print('macro scores:')
    print('--EOD score:')
    #print(classification_report(true_eod, pred_eod, target_names=['NON-EOD', 'EOD']))
    print('precision: '+str(precision_score(true_eod, pred_eod, average='macro')))
    print('recall: '+str(recall_score(true_eod, pred_eod, average='macro')))
    print('f1: '+str(f1_score(true_eod, pred_eod, average='macro')))
    
    print('--Action score:')
    #print(classification_report(true_action, pred_action, target_names=['FETCH', 'INSERT']))
    print('precision: '+str(precision_score(true_action, pred_action, average='macro')))
    print('recall: '+str(recall_score(true_action, pred_action, average='macro')))
    print('f1: '+str(f1_score(true_action, pred_action, average='macro')))
    
    print('--Intent score:')
    #print(classification_report(true_intent, pred_intent, target_names=['SCHEDULE', 'WEATHER', 'NAVIGATE']))
    print('precision: '+str(precision_score(true_intent, pred_intent, average='micro')))
    print('recall: '+str(recall_score(true_intent, pred_intent, average='micro')))
    print('f1: '+str(f1_score(true_intent, pred_intent, average='micro')))
    




def test(load_checkpoint_path):
    """
    Test utility
    """
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print('active device = '+str(device))

    # Dataset preparation


    # Bert adapter for dataset
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case = False)
    # pass max_len + 1 (to pad of 1 also the longest sentence, a sort of EOS) + 1 (random last sentence from other)
 

    # Model preparation
    model = MTSIBert(num_layers_encoder = MTSIKvretConfig._ENCODER_LAYERS_NUM,
                    num_layers_eod = MTSIKvretConfig._EOD_LAYERS_NUM,
                    n_intents = MTSIKvretConfig._N_INTENTS,
                    batch_size = MTSIKvretConfig._BATCH_SIZE,
                    pretrained = 'bert-base-cased',
                    seed = MTSIKvretConfig._SEED,
                    window_size = MTSIKvretConfig._WINDOW_SIZE)
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
    params = {'batch_size': MTSIKvretConfig._BATCH_SIZE,
            'shuffle': False,
            'num_workers': 0}

    # f1-score on test set
    test_set = KvretDataset(KvretConfig._KVRET_TEST_PATH)
    test_set.remove_subsequent_actor_utterances()
    badapter_test = MTSIAdapterDataset(test_set, 
                                        tokenizer,
                                        KvretConfig._KVRET_MAX_BERT_TOKENS_PER_TRAIN_SENTENCE + 1,
                                        KvretConfig._KVRET_MAX_BERT_SENTENCES_PER_TRAIN_DIALOGUE+2)
    test_generator = DataLoader(badapter_test, **params)
    print('### TEST SET:')
    compute_f1(model, test_generator, device)


    # f1-score on validation set
    val_set = KvretDataset(KvretConfig._KVRET_VAL_PATH)
    val_set.remove_subsequent_actor_utterances()
    badapter_val = MTSIAdapterDataset(val_set, 
                                        tokenizer,
                                        KvretConfig._KVRET_MAX_BERT_TOKENS_PER_TRAIN_SENTENCE + 1,
                                        KvretConfig._KVRET_MAX_BERT_SENTENCES_PER_TRAIN_DIALOGUE+2)
    val_generator = DataLoader(badapter_val, **params)
    print('### VALIDATION SET:')
    compute_f1(model, val_generator, device)






if __name__ == '__main__':
    test(load_checkpoint_path='dict_archive/MINI_BATCH16/100epochs/eod_no_RNN/state_dict.pt')
