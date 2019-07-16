import torch
from model import (KvretDataset, KvretConfig, FriendlyBert, MTSIBert)
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
    badapter_train = FriendlyBert(training_set, 'bert-base-cased', do_lower_case = False)
    badapter_val = FriendlyBert(validation_set, 'bert-base-cased', do_lower_case = False)

    # Parameters
    params = {'batch_size': 2,
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
           pdb.set_trace()
            









if __name__ == '__main__':
    main()

