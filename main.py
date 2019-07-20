import torch
from model import (KvretDataset, KvretConfig, MTSIKvretConfig, FriendlyBert, MTSIBert)
from pytorch_transformers import BertTokenizer
from torch.utils.data import DataLoader
import datetime
import os
import pdb


_N_EPOCHS = 4

def main():

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # Dataset preparation
    training_set = KvretDataset(KvretConfig._KVRET_TRAIN_PATH)
    validation_set = KvretDataset(KvretConfig._KVRET_VAL_PATH)

    # Bert adapter for dataset
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case = False)
    # pass max_len + 1 (to pad of 1 also the longest sentence, a sort of EOS)
    badapter_train = FriendlyBert(training_set, tokenizer, KvretConfig._KVRET_MAX_BERT_TOKEN_PER_TRAIN_SENTENCE + 1)
    badapter_val = FriendlyBert(validation_set, tokenizer, KvretConfig._KVRET_MAX_BERT_TOKEN_PER_VAL_SENTENCE + 1)

    # Parameters
    params = {'batch_size': MTSIKvretConfig._BATCH_SIZE,
            'shuffle': False,
            'num_workers': 0}

    training_generator = DataLoader(badapter_train, **params)
    validation_generator = DataLoader(badapter_val, **params)

    # Model preparation
    model = MTSIBert(num_layers = MTSIKvretConfig._LAYERS_NUM,\
                    n_labels = MTSIKvretConfig._N_LABELS,\
                    batch_size = MTSIKvretConfig._BATCH_SIZE,\
                    pretrained = 'bert-base-cased',\
                    seed = MTSIKvretConfig._SEED)
    model.to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = MTSIKvretConfig._LEARNING_RATE)
    
    # creates the directory for the checkpoints     
    os.makedirs(os.path.dirname(MTSIKvretConfig._SAVING_PATH), exist_ok=True)


    # ------------- TRAINING ------------- 
    for epoch in range(_N_EPOCHS):
        model.train()
        train_losses = []
        hidden = model.init_hidden()
        for local_batch, local_labels, dialogue_ids in training_generator:
            
            optimizer.zero_grad()
            #compute the mask
            #attention_mask = [[float(idx>0) for idx in sentence]for sentence in local_batch]
            #mask = torch.tensor(attention_mask)
            hidden = hidden.to(device)
            local_batch = local_batch.to(device)
            local_labels = local_labels.to(device)
            
            output, hidden = model(local_batch, hidden, dialogue_ids, device)
            loss = loss_fn(output, local_labels)
            train_losses.append(loss.item())
            # detach the hidden after each batch to avoid infinite gradient graph
            hidden.detach_()
            loss.backward()
            clipping_value = 1
            torch.nn.utils.clip_grad_norm_(model.parameters(), clipping_value)
            optimizer.step()
            
            occupied_mem_before = round(torch.cuda.memory_allocated()/1000000000, 2)
            occupied_cache_before = round(torch.cuda.memory_cached()/1000000000, 2)
            torch.cuda.empty_cache()
            occupied_mem_after = round(torch.cuda.memory_allocated()/1000000000, 2)
            occupied_cache_after = round(torch.cuda.memory_cached()/1000000000, 2)
            #pdb.set_trace()

        #end of epoch
        #saves the model weights
        curr_date = datetime.datetime.now().isoformat()
        torch.save(model.state_dict(),\
                     MTSIKvretConfig._SAVING_PATH+curr_date+'/state_dict.pt')


        # ------------- VALIDATION ------------- 
        val_losses = []
        with torch.no_grad:
            model.eval()
            hidden = model.init_hidden()
            for local_batch, local_labels, dialogue_ids in validation_generator:
                    #compute the mask
                    #attention_mask = [[float(idx>0) for idx in sentence]for sentence in local_batch]
                    #mask = torch.tensor(attention_mask)
                    hidden = hidden.to(device)
                    local_batch = local_batch.to(device)
                    local_labels = local_labels.to(device)
                    
                    output, hidden = model(local_batch, hidden, dialogue_ids, device)
                    
                    loss = loss_fn(output, local_labels)
                    val_losses.append(loss.item())

        res_str = 'EPOCH '+str(epoch)+' || TRAIN LOSS ==> '+str(np.mean(train_losses))+\
                                    ' || VALIDATION LOSS ==> '+str(np.mean(val_losses))
        print(res_str)
        os.makedirs(os.path.dirname(MTSIKvretConfig._SAVING_PATH+curr_date+'/'), exist_ok=True)
        # write result for current checkpoint
        with open(MTSIKvretConfig._SAVING_PATH+curr_date+'/checkpoint.txt', "w+") as f:
            f.write('checkpoint '+curr_date+'\n')

            

            

            





def statistics():
    """
    To retrieve the max sentence lenth in order to apply the rigth amount of padding
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case = False)
    print('### TRAINING: ')
    training_set = KvretDataset(KvretConfig._KVRET_TRAIN_PATH)
    #tok, sentences, id = training_set.get_max_tokens_per_dialogue(tokenizer)
    tok, sentences, id = training_set.get_max_tokens_per_sentence(tokenizer)
    print('num tokens = ' + str(tok) + ', \nnum sentences = ' + str(sentences) + ', \nid = ' + id)

    print('### VALIDATION: ')
    validation_set = KvretDataset(KvretConfig._KVRET_VAL_PATH)
    #tok, sentences, id = validation_set.get_max_tokens_per_dialogue(tokenizer)
    tok, sentences, id = validation_set.get_max_tokens_per_sentence(tokenizer)
    print('num tokens = ' + str(tok) + ', \nnum sentences = ' + str(sentences) + ', \nid = ' + id)

    print('### TEST: ')
    test_set = KvretDataset(KvretConfig._KVRET_TEST_PATH)
    #tok, sentences, id = test_set.get_max_tokens_per_dialogue(tokenizer)
    tok, sentences, id = test_set.get_max_tokens_per_sentence(tokenizer)
    print('num tokens = ' + str(tok) + ', \nnum sentences = ' + str(sentences) + ', \nid = ' + id)




if __name__ == '__main__':
    #statistics()
    main()



