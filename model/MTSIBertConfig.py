
"""
A module containing all the infos for the project
"""


class KvretConfig():
    """
    Kvret Dataset configuration parameters
    """
    _KVRET_TRAIN_PATH = 'dataset/kvret_dataset_public/kvret_train_public.json'
    _KVRET_VAL_PATH = 'dataset/kvret_dataset_public/kvret_dev_public.json'
    _KVRET_TEST_PATH = 'dataset/kvret_dataset_public/kvret_test_public.json'
    _KVRET_ENTITIES_PATH = 'dataset/kvret_dataset_public/kvret_entities.json'
    
    _KVRET_MAX_BERT_TOKEN_PER_TRAIN_SENTENCE = 113
    _KVRET_MAX_BERT_TOKEN_PER_VAL_SENTENCE = 111
    _KVRET_MAX_BERT_TOKEN_PER_TEST_SENTENCE = 50

    _KVRET_MAX_BERT_SENTENCE_PER_TRAIN_DIALOGUE = 12
    _KVRET_MAX_BERT_SENTENCE_PER_VAL_DIALOGUE = 12
    _KVRET_MAX_BERT_SENTENCE_PER_TEST_DIALOGUE = 12


class MTSIKvretConfig:
    """
    MTSI-Bert model parameters for Kvret dataset
    """
    _N_LABELS = 3
    _BATCH_SIZE = 3
    _LAYERS_NUM = 2
    _SEED = 1 # for reroducibility of results
    _LEARNING_RATE = 0.005
    _WINDOW_SIZE = 3 # tipically odd number [Q(t-1), R(t-1), Q(t)]

    _SAVING_PATH = 'savings/'






