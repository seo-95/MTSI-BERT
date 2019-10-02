import datetime
import logging
import os
import pdb
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from model import (KvretConfig, KvretDataset, BLAdapterDataset, BaseLine,
                   BaselineKvretConfig)

_N_EPOCHS = 20
_OPTIMIZER_STEP_RATE = 16 # how many samples has to be computed before the optimizer.step()




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

    # pass max_len + 1 (to pad of 1 also the longest sentence, a sort of EOS) + 1 (random last sentence from other)
    badapter_train = BLAdapterDataset(training_set,
                                    KvretConfig._KVRET_MAX_BERT_TOKENS_PER_TRAIN_SENTENCE + 1,\
                                    KvretConfig._KVRET_MAX_BERT_SENTENCES_PER_TRAIN_DIALOGUE+2)
    # for validation keep using the train max tokens for model compatibility
    badapter_val = BLAdapterDataset(validation_set,
                                    KvretConfig._KVRET_MAX_BERT_TOKENS_PER_TRAIN_SENTENCE + 1,\
                                    KvretConfig._KVRET_MAX_BERT_SENTENCES_PER_VAL_DIALOGUE+2)

    # Parameters
    params = {'batch_size': BaselineKvretConfig._BATCH_SIZE,
            'shuffle': True,
            'num_workers': 0}

    training_generator = DataLoader(badapter_train, **params)
    validation_generator = DataLoader(badapter_val, **params)

    # Model preparation
    model = BaseLine(num_layers_encoder = BaselineKvretConfig._ENCODER_LAYERS_NUM,
                    num_layers_eod = BaselineKvretConfig._EOD_LAYERS_NUM,
                    n_intents = BaselineKvretConfig._N_INTENTS,
                    seed = BaselineKvretConfig._SEED)

    if load_checkpoint_path != None:
        print('model loaded from: '+load_checkpoint_path)
        new_state_dict = remove_dataparallel(load_checkpoint_path)
        model.load_state_dict(new_state_dict)
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

    optimizer = torch.optim.Adam(model.parameters(), lr = BaselineKvretConfig._LEARNING_RATE, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = [5,10,15,20,30,40,50,75], gamma = 0.5)
    
    # creates the directory for the checkpoints
    os.makedirs(os.path.dirname(BaselineKvretConfig._SAVING_PATH), exist_ok=True)
    curr_date = datetime.datetime.now().isoformat()
    os.makedirs(os.path.dirname(BaselineKvretConfig._SAVING_PATH+curr_date+'/'), exist_ok=True)
    # creates the directory for the plots figure
    os.makedirs(os.path.dirname(BaselineKvretConfig._PLOTS_SAVING_PATH), exist_ok=True)
    
    # initializes flag and list of overall losses
    best_loss = 100
    train_global_losses = []
    val_global_losses = []
    eod_val_global_losses = []
    action_val_global_losses = []
    intent_val_global_losses = []


    # ------------- TRAINING ------------- 

    for epoch in range(_N_EPOCHS):
        model.train()
        t_eod_losses = []
        t_intent_losses = []
        t_action_losses = []
        idx = 0

        for local_batch, local_turns, seqs_len, local_intents, local_actions, dialogue_ids in training_generator:
            
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

            # compute loss only on real dialogue (exclude padding)
            loss1 = loss_eod(eos['logit'], eos_label)
            loss2 = loss_intent(intent['logit'], local_intents)
            loss3 = loss_action(action['logit'], local_actions)

            loss1.backward()
            loss2.backward()
            loss3.backward()

            #save results
            t_eod_losses.append(loss1.item())
            t_intent_losses.append(loss2.item())
            t_action_losses.append(loss3.item())

            if idx != 0 and idx % _OPTIMIZER_STEP_RATE == 0 or idx == badapter_train.__len__()-1:
                optimizer.step()
                optimizer.zero_grad()

            if 'cuda' in str(device):
                torch.cuda.empty_cache()
            
            idx += 1
            
        #end of epoch


        # ------------- VALIDATION ------------- 
        val_losses = []
        with torch.no_grad():
            model.eval()
            v_eod_losses = []
            v_intent_losses = []
            v_action_losses = []
            
            for local_batch, local_turns, seqs_len, local_intents, local_actions, dialogue_ids in validation_generator:
                
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

                # compute loss only on real dialogue (exclude padding)
                loss1 = loss_eod(eos['logit'], eos_label)
                loss2 = loss_intent(intent['logit'], local_intents)
                loss3 = loss_action(action['logit'], local_actions)

                #save results
                v_eod_losses.append(loss1.item())
                v_intent_losses.append(loss2.item())
                v_action_losses.append(loss3.item())

                if 'cuda' in str(device):
                    torch.cuda.empty_cache()


        # compute the mean for each loss in the current epoch
        t_eod_curr_mean = round(np.mean(t_eod_losses), 4)
        t_action_curr_mean = round(np.mean(t_action_losses), 4)
        t_intent_curr_mean = round(np.mean(t_intent_losses), 4)

        v_eod_curr_mean = round(np.mean(v_eod_losses), 4)
        v_action_curr_mean = round(np.mean(v_action_losses), 4)
        v_intent_curr_mean = round(np.mean(v_intent_losses), 4)

        train_mean_loss = round(np.mean([t_eod_curr_mean, t_action_curr_mean, t_intent_curr_mean]), 4)
        val_mean_loss = round(np.mean([v_eod_curr_mean, v_action_curr_mean, v_intent_curr_mean]), 4)


        # accumulate losses for plotting
        eod_val_global_losses.append(v_eod_curr_mean)
        action_val_global_losses.append(v_action_curr_mean)
        intent_val_global_losses.append(v_intent_curr_mean)
        train_global_losses.append(train_mean_loss)
        val_global_losses.append(val_mean_loss)

        
        # check if new best model
        if val_mean_loss < best_loss:
          #saves the model weights
          best_loss = val_mean_loss 
          # save using model.cpu to allow the further loading also on cpu or single-GPU
          torch.save(model.cpu().state_dict(),\
                       BaselineKvretConfig._SAVING_PATH+curr_date+'/state_dict.pt')
        model.to(device)
        
        curr_lr = optimizer.param_groups[0]['lr']
        log_str = '### EPOCH '+str(epoch+1)+'/'+str(_N_EPOCHS)+'(nn_lr='+str(curr_lr)+'):: TRAIN LOSS = '+str(train_mean_loss)+\
                                                                '[eod = '+str(round(np.mean(t_eod_losses), 4))+'], '+\
                                                                '[action = '+str(round(np.mean(t_action_losses), 4))+'], '+\
                                                                '[intent = '+str(round(np.mean(t_intent_losses), 4))+'], '+\
                                                                '\n\t\t\t || VAL LOSS = '+str(val_mean_loss)+\
                                                                '[eod = '+str(round(np.mean(v_eod_losses), 4))+'], '+\
                                                                '[action = '+str(round(np.mean(v_action_losses), 4))+'], '+\
                                                                '[intent = '+str(round(np.mean(v_intent_losses), 4))+']'
        print(log_str)
        # step of scheduler to reduce the lr each milestone
        scheduler.step()


    # ------------ FINAL PLOTS ------------

    epoch_list = np.arange(1, _N_EPOCHS+1)

    # plot train vs validation
    plt.plot(epoch_list, train_global_losses, color='blue', label='train loss')
    plt.plot(epoch_list, val_global_losses, color='red', label='validation loss')
    
    plt.title('train vs validation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.savefig(BaselineKvretConfig._PLOTS_SAVING_PATH+'train_vs_val.png') 

    # clean figure
    plt.clf()

    # plot eod vs action vs intent
    plt.plot(epoch_list, eod_val_global_losses, color='red', label='eod loss')
    plt.plot(epoch_list, action_val_global_losses, color='green', label='action loss')
    plt.plot(epoch_list, intent_val_global_losses, color='blue', label='intent loss')

    plt.title('eod vs action vs intent')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Loss')
    plt.legend(loc='best')
    plt.savefig(BaselineKvretConfig._PLOTS_SAVING_PATH+'validation_losses.png')






if __name__ == '__main__':
    start = time.time()
    train()
    end = time.time()
    h_count = (end-start)/60/60
    print('training time: '+str(h_count)+'h')
