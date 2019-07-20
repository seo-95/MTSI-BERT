import torch
from model import (KvretDataset, KvretConfig, FriendlyBert, MTSIBert)
from pytorch_transformers import BertTokenizer
from torch.utils.data import DataLoader
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
    params = {'batch_size': KvretConfig._BATCH_SIZE,
            'shuffle': False,
            'num_workers': 0}

    training_generator = DataLoader(badapter_train, **params)
    validation_generator = DataLoader(badapter_val, **params)

    # Model preparation
    model = MTSIBert(num_layers = KvretConfig._LAYERS_NUM,\
                    n_labels = KvretConfig._N_LABELS,\
                    batch_size = KvretConfig._BATCH_SIZE,\
                    pretrained = 'bert-base-cased',\
                    seed = KvretConfig._SEED)


    # ------------- TRAINING ------------- 
    hidden = model.init_hidden()
    for epoch in range(_N_EPOCHS):
       for local_batch, local_labels, dialogue_ids in training_generator:
            #compute the mask
            #attention_mask = [[float(idx>0) for idx in sentence]for sentence in local_batch]
            #mask = torch.tensor(attention_mask)
            
            out, hidden = model(local_batch, hidden, dialogue_ids)
            pdb.set_trace()
            

            

            





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



