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

_N_EPOCHS = 15
_OPTIMIZER_STEP_RATE = 16 # how many samples has to be computed before the optimizer.step()


def get_eod(turns, win_size, windows_per_dialogue):
    
    res = torch.zeros((len(turns), windows_per_dialogue), dtype=torch.long)
    user_count = 0
    for idx, curr_dial in enumerate(turns):
        for t in curr_dial:
            if t == 1:
                user_count += 1
        res[idx][user_count-1] = 1

    return res, user_count-1


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
    # pass max_len + 1 (to pad of 1 also the longest sentence, a sort of EOS) + 1 (random last sentence from other)
    badapter_train = MTSIAdapterDataset(training_set, tokenizer,\
                                    KvretConfig._KVRET_MAX_BERT_TOKENS_PER_TRAIN_SENTENCE + 1,\
                                    KvretConfig._KVRET_MAX_BERT_SENTENCES_PER_TRAIN_DIALOGUE+2)
    # for validation keep using the train max tokens for model compatibility
    badapter_val = MTSIAdapterDataset(validation_set, tokenizer,\
                                    KvretConfig._KVRET_MAX_BERT_TOKENS_PER_TRAIN_SENTENCE + 1,\
                                    KvretConfig._KVRET_MAX_BERT_SENTENCES_PER_VAL_DIALOGUE+2)

    # Parameters
    params = {'batch_size': MTSIKvretConfig._BATCH_SIZE,
            'shuffle': True,
            'num_workers': 0}

    training_generator = DataLoader(badapter_train, **params)
    validation_generator = DataLoader(badapter_val, **params)

    # Model preparation
    model = MTSIBert(num_layers_encoder = MTSIKvretConfig._ENCODER_LAYERS_NUM,
                    num_layers_eod = MTSIKvretConfig._EOD_LAYERS_NUM,
                    n_intents = MTSIKvretConfig._N_INTENTS,
                    batch_size = MTSIKvretConfig._BATCH_SIZE,
                    pretrained = 'bert-base-cased',
                    seed = MTSIKvretConfig._SEED,
                    window_size = MTSIKvretConfig._WINDOW_SIZE)

    if load_checkpoint_path != None:
        print('model loaded from: '+load_checkpoint_path)
        model.load_state_dict(torch.load(load_checkpoint_path))
    # work on multiple GPUs when availables
    if torch.cuda.device_count() > 1:
        print('active devices = '+str(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    model.to(device)

    # this weights are needed because of unbalancing between 0 and 1 for action and eod
    loss_eod_weights = torch.tensor([1, 2.6525])
    loss_action_weights = torch.tensor([1, 4.8716])
    loss_eod = torch.nn.CrossEntropyLoss(weight=loss_eod_weights).to(device)
    loss_action = torch.nn.CrossEntropyLoss(weight=loss_action_weights).to(device)
    loss_intent = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = MTSIKvretConfig._LEARNING_RATE, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = [4,8,12], gamma = 0.5)
    
    # creates the directory for the checkpoints
    os.makedirs(os.path.dirname(MTSIKvretConfig._SAVING_PATH), exist_ok=True)
    curr_date = datetime.datetime.now().isoformat()
    os.makedirs(os.path.dirname(MTSIKvretConfig._SAVING_PATH+curr_date+'/'), exist_ok=True)
    
    # initializes statistics
    train_len = training_set.__len__()
    val_len = validation_set.__len__()
    best_loss = 100


    # ------------- TRAINING ------------- 

    tensor_builder = TwoSepTensorBuilder()

    for epoch in range(_N_EPOCHS):
        model.train()
        t_eod_losses = []
        t_intent_losses = []
        t_action_losses = []
        train_correctly_predicted = 0
        val_correctly_predicted = 0
        idx = 0

        for local_batch, local_turns, local_intents, local_actions, dialogue_ids in training_generator:

            # 0 = intra dialogue ; 1 = eod
            eod_label, eod_idx = get_eod(local_turns, MTSIKvretConfig._WINDOW_SIZE,
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
                                        local_turns, dialogue_ids,
                                        tensor_builder,
                                        device)

            # compute loss only on real dialogue (exclude padding)
            loss1 = loss_eod(eod['logit'].squeeze(0)[:eod_idx+1], eod_label.squeeze(0)[:eod_idx+1])
            loss2 = loss_intent(intent['logit'].unsqueeze(0), local_intents)
            loss3 = loss_action(action['logit'].unsqueeze(0), local_actions)
            tot_loss = (loss1 + loss2 + loss3)/3
            tot_loss.backward()

            #save results
            t_eod_losses.append(loss1.item())
            t_intent_losses.append(loss2.item())
            t_action_losses.append(loss3.item())

            if idx % _OPTIMIZER_STEP_RATE == 0 or idx == badapter_train.__len__()-1:
                optimizer.step()
                optimizer.zero_grad()
            # detach the hidden after each batch to avoid infinite gradient graph
            #hidden.detach_()

            # count correct predictions
            # predictions = torch.argmax(output, dim=1)
            # train_correctly_predicted += (predictions == local_labels).sum().item()
        
            if 'cuda' in str(device):
                torch.cuda.empty_cache()
            
            
        #end of epoch


        # ------------- VALIDATION ------------- 
        val_losses = []
        with torch.no_grad():
            model.eval()
            v_eod_losses = []
            v_intent_losses = []
            v_action_losses = []
            
            for local_batch, local_turns, local_intents, local_actions, dialogue_ids in validation_generator:
                
                # 0 = intra dialogue ; 1 = eod
                eod_label, eod_idx = get_eod(local_turns, MTSIKvretConfig._WINDOW_SIZE,\
                                    windows_per_dialogue=KvretConfig._KVRET_MAX_USER_SENTENCES_PER_TRAIN_DIALOGUE + 1)
                
                # local_batch.shape == B x D_LEN x U_LEN
                # local_intents.shape == B
                # local_actions.shape == B
                # local_eod_label.shape == B x D_PER_WIN
                local_batch = local_batch.to(device)
                local_intents = local_intents.to(device)
                local_actions = local_actions.to(device)
                eod_label = eod_label.to(device)
                    

                eod, intent, action = model(local_batch,
                                            local_turns, dialogue_ids,
                                            tensor_builder,
                                            device)
                if 'cuda' in str(device):
                    torch.cuda.empty_cache()
                
                loss1 = loss_eod(eod['logit'].squeeze(0)[:eod_idx+1], eod_label.squeeze(0)[:eod_idx+1])
                loss2 = loss_intent(intent['logit'].unsqueeze(0), local_intents)
                loss3 = loss_action(action['logit'].unsqueeze(0), local_actions)

                #save results
                v_eod_losses.append(loss1.item())
                v_intent_losses.append(loss2.item())
                v_action_losses.append(loss3.item())
                
                # count correct predictions
                #predictions = torch.argmax(output, dim=1)
                #val_correctly_predicted += (predictions == local_labels).sum().item()


        #train_accuracy = round(train_correctly_predicted/train_len * 100, 2)
        #val_accuracy = round(val_correctly_predicted/val_len * 100, 2)
        train_mean_loss = round(np.mean([t_eod_losses, t_action_losses, t_intent_losses]), 4)
        val_mean_loss = round(np.mean([v_eod_losses, v_action_losses, v_intent_losses]), 4)
        
        # check if new best model
        if val_mean_loss < best_loss:
          #saves the model weights
          best_loss = val_mean_loss 
          # save using model.cpu to allow the further loading also on cpu or single-GPU
          torch.save(model.cpu().state_dict(),\
                       MTSIKvretConfig._SAVING_PATH+curr_date+'/state_dict.pt')
        model.to(device)
        

        log_str = '### EPOCH '+str(epoch+1)+'/'+str(_N_EPOCHS)+':: TRAIN LOSS = '+str(train_mean_loss)+\
                                                                '[eod = '+str(round(np.mean(t_eod_losses), 4))+'], '+\
                                                                '[action = '+str(round(np.mean(t_action_losses), 4))+'], '+\
                                                                '[intent = '+str(round(np.mean(t_intent_losses), 4))+'], '+\
                                                                '\n\t\t\t || VAL LOSS = '+str(val_mean_loss)+\
                                                                '[eod = '+str(round(np.mean(v_eod_losses), 4))+'], '+\
                                                                '[action = '+str(round(np.mean(v_action_losses), 4))+'], '+\
                                                                '[intent = '+str(round(np.mean(v_intent_losses), 4))+']'
        print(log_str)
        scheduler.step()







if __name__ == '__main__':
    train()