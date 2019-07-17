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
    badapter_train = FriendlyBert(training_set, tokenizer, KvretConfig._KVRET_MAX_BERT_TOKEN_PER_TRAIN_SENTENCE + 1) #TODO pass max_len + 1 (to pad of 1 also the longest sentence, a sort of EOS)
    badapter_val = FriendlyBert(validation_set, tokenizer, KvretConfig._KVRET_MAX_BERT_TOKEN_PER_VAL_SENTENCE + 1)

    # Parameters
    params = {'batch_size': 4,
            'shuffle': False,
            'num_workers': 0}

    training_generator = DataLoader(badapter_train, **params)
    validation_generator = DataLoader(badapter_val, **params)

    # Model preparation
    model = MTSIBert(num_layers = KvretConfig._LAYERS_NUM,\
                    n_labels = KvretConfig._N_LABELS,\
                    batch_size = KvretConfig._BATCH_SIZE,\
                    pretrained = 'bert-base-cased')

    for epoch in range(_N_EPOCHS):
       for local_batch, local_labels in training_generator:
           #None
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
    """
    text = "It's me Mario![MASK]"
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case = False)
    res = tokenizer.tokenize(text)
    print(res)
    """



