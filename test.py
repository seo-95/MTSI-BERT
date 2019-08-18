import datetime
import logging
import os
import pdb
import sys

import numpy as np
import torch
from pytorch_transformers import BertTokenizer
from torch.utils.data import DataLoader
from torch import nn

from model import (MTSIAdapterDataset, KvretConfig, KvretDataset, MTSIBert,
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



def test(load_checkpoint_path):
    """
    Test utility
    """
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print('active device = '+str(device))

    # Dataset preparation
    test_set = KvretDataset(KvretConfig._KVRET_TEST_PATH)
    test_set.remove_subsequent_actor_utterances()

    # Bert adapter for dataset
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case = False)
    # pass max_len + 1 (to pad of 1 also the longest sentence, a sort of EOS) + 1 (random last sentence from other)
    badapter_test = MTSIAdapterDataset(test_set, tokenizer,\
                                    KvretConfig._KVRET_MAX_BERT_TOKENS_PER_TRAIN_SENTENCE + 1,\
                                    KvretConfig._KVRET_MAX_BERT_SENTENCES_PER_TRAIN_DIALOGUE+2)

    # Parameters
    params = {'batch_size': MTSIKvretConfig._BATCH_SIZE,
            'shuffle': True,
            'num_workers': 0}

    test_generator = DataLoader(badapter_test, **params)

    # Model preparation
    model = MTSIBert(num_layers = MTSIKvretConfig._LAYERS_NUM,
                    n_intents = MTSIKvretConfig._N_INTENTS,
                    batch_size = MTSIKvretConfig._BATCH_SIZE,
                    # length of a single tensor: (max_tokens+1) + 3bert_tokens which are 1[CLS] and 2[SEP]
                    window_length = KvretConfig._KVRET_MAX_BERT_TOKENS_PER_WINDOWS + 1,
                    # user utterances for this dialogue + first user utterance of the next
                    windows_per_batch = KvretConfig._KVRET_MAX_USER_SENTENCES_PER_TRAIN_DIALOGUE + 2,
                    pretrained = 'bert-base-cased',
                    seed = MTSIKvretConfig._SEED,
                    window_size = MTSIKvretConfig._WINDOW_SIZE)
    # work on multiple GPUs when availables
    if torch.cuda.device_count() > 1:
        print('active devices = '+str(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    print('model loaded from: '+load_checkpoint_path)
    model.load_state_dict(torch.load(load_checkpoint_path))
    model.to(device)
    # initializes statistics
    test_len = test_set.__len__()
    model.eval()
    with torch.no_grad:
        for local_batch, local_turns, local_intents, local_actions, dialogue_ids in test_generator:

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

            eod, intent, action, hidden = model(local_batch,\
                                            local_turns, dialogue_ids,\
                                            tensor_builder,\
                                            device)
            pdb.set_trace()
            # count correct predictions
            predictions = torch.argmax(output, dim=1)
            test_correctly_predicted += (predictions == local_labels).sum().item()



if __name__ == '__main__':
    test(load_checkpoint_path='savings/2019-08-14T09:33:09.821063/state_dict.pt')