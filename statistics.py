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



def print_statistics_per_set(curr_set):
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case = False)

    tok, sentences, id = curr_set.get_max_tokens_per_dialogue(tokenizer)
    print('\n--- max tokens per dialogue ---')
    print('num tokens = ' + str(tok) + ', \nnum sentences = ' + str(sentences) + ', \nid = ' + id)

    tot_num = curr_set.get_total_num_utterances()
    print('\n--- total num utterances ---')
    print('num = ' + str(tot_num) + ', dialogues num = ' + str(curr_set.__len__()))

    tok, sentences, id = curr_set.get_max_tokens_per_sentence(tokenizer)
    print('\n--- max tokens per sentence ---')
    print('num tokens = ' + str(tok) + ', \nnum sentences = ' + str(sentences) + ', \nid = ' + id)

    sentences, id = curr_set.get_max_utterances_per_dialogue()
    print('\n--- max sentences per dialogue ---')
    print('num sentences = ' + str(sentences) + ', \nid = ' + id)

    sentences, id = curr_set.get_max_num_user_utterances()
    print('\n--- max user utterances per dialogue ---')
    print('num user utterances = ' + str(sentences) + ', \nid = ' + id)

    fetch_count, insert_count = curr_set.get_action_frequency()
    id_l = curr_set.get_dialogues_with_subsequent_same_actor_utterances()
    print('\n--- check dialogues with subsequent same actor utterances ---')
    print('ids = ' + str(id_l))
    print('\n--- action frequency ---')
    print('num fetch = ' + str(fetch_count) + ' vs num insert = ' + str(insert_count))


def statistics(remove_subsequent_actor=False):
    """
    To retrieve the max sentence lenth in order to apply the rigth amount of padding
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case = False)

    print('\n\t\t\t### TRAINING ###')
    training_set = KvretDataset(KvretConfig._KVRET_TRAIN_PATH)
    if remove_subsequent_actor:
        training_set.remove_subsequent_actor_utterances()
    print_statistics_per_set(training_set)
    
    print('\n\t\t\t### VALIDATION ###')
    validation_set = KvretDataset(KvretConfig._KVRET_VAL_PATH)
    if remove_subsequent_actor:
        validation_set.remove_subsequent_actor_utterances()
    print_statistics_per_set(validation_set)

    print('\n\t\t\t### TEST ###')
    test_set = KvretDataset(KvretConfig._KVRET_TEST_PATH)
    if remove_subsequent_actor:
        test_set.remove_subsequent_actor_utterances()
    print_statistics_per_set(test_set)





if __name__ == '__main__':
    statistics(True)
