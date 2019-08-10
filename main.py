import datetime
import logging
import os
import pdb
import sys

import GPUtil
import numpy as np
import torch
from pytorch_transformers import BertTokenizer
from torch.utils.data import DataLoader

from model import (FriendlyBert, KvretConfig, KvretDataset, MTSIBert,
                   MTSIKvretConfig, TwoSepTensorBuilder)

_N_EPOCHS = 20

def train(load_checkpoint_path=None):

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print('active device = '+str(device))

    # Dataset preparation
    training_set = KvretDataset(KvretConfig._KVRET_TRAIN_PATH)
    training_set.remove_subsequent_actor_utterances()
    validation_set = KvretDataset(KvretConfig._KVRET_VAL_PATH)
    validation_set.remove_subsequent_actor_utterances()

    # Bert adapter for dataset
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case = False)
    # pass max_len + 1 (to pad of 1 also the longest sentence, a sort of EOS)
    badapter_train = FriendlyBert(training_set, tokenizer,\
                                    KvretConfig._KVRET_MAX_BERT_TOKENS_PER_TRAIN_SENTENCE + 1,\
                                    KvretConfig._KVRET_MAX_BERT_SENTENCES_PER_TRAIN_DIALOGUE+1)
    badapter_val = FriendlyBert(validation_set, tokenizer,\
                                    KvretConfig._KVRET_MAX_BERT_TOKENS_PER_VAL_SENTENCE + 1,\
                                    KvretConfig._KVRET_MAX_BERT_SENTENCES_PER_VAL_DIALOGUE+1)

    # Parameters
    params = {'batch_size': MTSIKvretConfig._BATCH_SIZE,
            'shuffle': False,
            'num_workers': 0}

    training_generator = DataLoader(badapter_train, **params)
    validation_generator = DataLoader(badapter_val, **params)

    # Model preparation
    model = MTSIBert(num_layers = MTSIKvretConfig._LAYERS_NUM,
                    n_labels = MTSIKvretConfig._N_LABELS,
                    batch_size = MTSIKvretConfig._BATCH_SIZE,
                    # length of a single tensor: (max_tokens+1) + 3bert_tokens which are 1[CLS] and 2[SEP]
                    window_length = 3*(KvretConfig._KVRET_MAX_BERT_TOKENS_PER_TRAIN_SENTENCE + 1) + 3,
                    # user utterances for this dialogue + first user utterance of the next
                    windows_per_batch = KvretConfig._KVRET_MAX_USER_SENTENCES_PER_TRAIN_DIALOGUE + 1,
                    pretrained = 'bert-base-cased',
                    seed = MTSIKvretConfig._SEED,
                    window_size = MTSIKvretConfig._WINDOW_SIZE)
    if load_checkpoint_path != None:
        print('model loaded from: '+load_path)
        model.load_state_dict(torch.load(load_checkpoint_path))
    if torch.cuda.device_count() > 1:
        print('active devices = '+str(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    model.to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = MTSIKvretConfig._LEARNING_RATE)
    
    # creates the directory for the checkpoints     
    os.makedirs(os.path.dirname(MTSIKvretConfig._SAVING_PATH), exist_ok=True)
    curr_date = datetime.datetime.now().isoformat()
    os.makedirs(os.path.dirname(MTSIKvretConfig._SAVING_PATH+curr_date+'/'), exist_ok=True)
    #if not os.path.exists(MTSIKvretConfig._SAVING_PATH+curr_date+'/'+'MTSI.log'):
    #  with open(MTSIKvretConfig._SAVING_PATH+curr_date+'/'+'MTSI.log', 'w'): pass
    #logging.basicConfig(filename=MTSIKvretConfig._SAVING_PATH+curr_date+'/'+'MTSI.log', filemode='a', level=logging.INFO)
    
    # initializes statistics
    train_len = training_set.__len__()
    val_len = validation_set.__len__()
    best_loss = 100


    # ------------- TRAINING ------------- 

    tensor_builder = TwoSepTensorBuilder()

    for epoch in range(_N_EPOCHS):
        model.train()
        train_losses = []
        train_correctly_predicted = 0
        val_correctly_predicted = 0
        hidden = model.init_hidden()
        hidden = hidden.to(device)
        
        for local_batch, local_turns, local_intents, local_actions, dialogue_ids in training_generator:
            
            # local_batch.size() == B x D_LEN x U_LEN
            # local_intents = B x D_LEN
            # local_actions = B x D_LEN
            local_batch = local_batch.to(device)
            local_intents = local_intents.to(device)
            local_actions = local_actions.to(device)

            optimizer.zero_grad()

            output, logits, hidden = model(local_batch, hidden,\
                                            local_turns, dialogue_ids,\
                                            tensor_builder,\
                                            persistence = True, device=device)

            loss = loss_fn(logits, local_labels)
            train_losses.append(loss.item())
            loss.backward()

            if str(device) == 'cuda:0':
                occupied_mem_before = round(torch.cuda.memory_allocated()/1000000000, 2)
                occupied_cache_before = round(torch.cuda.memory_cached()/1000000000, 2)
            
            pdb.set_trace()
            if str(device) == 'cuda:0':
                torch.cuda.empty_cache()
            #clipping_value = 5
            #torch.nn.utils.clip_grad_norm_(model.parameters(), clipping_value) TODO only on GRU
            optimizer.step()
            # detach the hidden after each batch to avoid infinite gradient graph
            hidden.detach_()

            # count correct predictions
            #predictions = torch.argmax(output, dim=1)
            #train_correctly_predicted += (predictions == local_labels).sum().item()
        
            if str(device) == 'cuda:0':
                torch.cuda.empty_cache()
                occupied_mem_after = round(torch.cuda.memory_allocated()/1000000000, 2)
                occupied_cache_after = round(torch.cuda.memory_cached()/1000000000, 2)
                print('##[BEFORE] : MEM='+str(occupied_mem_before)+' CACHE='+str(occupied_cache_before)+'\n\t'+\
                        '[AFTER] : MEM='+str(occupied_mem_after)+' CACHE='+str(occupied_cache_after))
            
            
        #end of epoch
















        # ------------- VALIDATION ------------- 
        val_losses = []
        with torch.no_grad():
            model.eval()
            hidden = model.init_hidden()
            hidden = hidden.to(device)
            for local_batch, local_labels, dialogue_ids in validation_generator:
                    
                    local_batch = local_batch.to(device)
                    local_labels = local_labels.to(device)
                    
                    output, logits, hidden = model(local_batch, hidden, turns, dialogue_ids, device)
                    torch.cuda.empty_cache()
                    
                    loss = loss_fn(logits, local_labels)
                    val_losses.append(loss.item())
                    
                    # count correct predictions
                    #predictions = torch.argmax(output, dim=1)
                    #val_correctly_predicted += (predictions == local_labels).sum().item()


        #train_accuracy = round(train_correctly_predicted/train_len * 100, 2)
        #val_accuracy = round(val_correctly_predicted/val_len * 100, 2)
        train_mean_loss = round(np.mean(train_losses), 4)
        val_mean_loss = round(np.mean(val_losses), 4)
        
        # check if new best model
        if val_mean_loss < best_loss:
          #saves the model weights
          best_loss = val_mean_loss
          torch.save(model.state_dict(),\
                       MTSIKvretConfig._SAVING_PATH+curr_date+'/state_dict.pt')
        
        #log_str = '### EPOCH '+str(epoch+1)+'/'+str(_N_EPOCHS)+':: TRAIN LOSS = '+str(train_mean_loss)+', TRAIN ACCURACY= '+str(train_accuracy)+'%'+\
        #                            '\n\t\t\t || VAL LOSS = '+str(val_mean_loss)+', VAL ACCURACY= '+str(val_accuracy)+'%'
        log_str = '### EPOCH '+str(epoch+1)+'/'+str(_N_EPOCHS)+':: TRAIN LOSS = '+str(train_mean_loss)+\
                                                                '\n\t\t\t || VAL LOSS = '+str(val_mean_loss)
        print(log_str)
        logging.info(log_str)




def print_statistics_per_set(curr_set):
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case = False)

    tok, sentences, id = curr_set.get_max_tokens_per_dialogue(tokenizer)
    print('\n--- max tokens per dialogue ---')
    print('num tokens = ' + str(tok) + ', \nnum sentences = ' + str(sentences) + ', \nid = ' + id)
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
    #statistics(True)
    #train(load_checkpoint_path='savings/2019-08-04T17:43:29.354518/state_dict.pt')
    train()
