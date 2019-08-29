
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
    
    _KVRET_MAX_BERT_TOKENS_PER_TRAIN_SENTENCE = 113
    _KVRET_MAX_BERT_TOKENS_PER_VAL_SENTENCE = 111
    _KVRET_MAX_BERT_TOKENS_PER_TEST_SENTENCE = 50

    _KVRET_MAX_BERT_SENTENCES_PER_TRAIN_DIALOGUE = 12
    _KVRET_MAX_BERT_SENTENCES_PER_VAL_DIALOGUE = 12
    _KVRET_MAX_BERT_SENTENCES_PER_TEST_DIALOGUE = 12

    _KVRET_MAX_USER_SENTENCES_PER_TRAIN_DIALOGUE = 6
    _KVRET_MAX_USER_SENTENCES_PER_VALIDATION_DIALOGUE = 6
    _KVRET_MAX_USER_SENTENCES_PER_TEST_DIALOGUE = 6 # 7 if subsequent utterances not removed

    _KVRET_MAX_BERT_TOKENS_PER_WINDOWS = 129 # see kvret statistics


class MTSIKvretConfig:
    """
    MTSI-Bert model parameters for Kvret dataset
    """
    _N_INTENTS = 3 # number of intents
    _BATCH_SIZE = 1
    _ENCODER_LAYERS_NUM = 1
    _EOD_LAYERS_NUM = 1
    _SEED = 26 # for reproducibility of results
    _LEARNING_RATE = 5e-5
    _WINDOW_SIZE = 3 # tipically odd number [Q(t-1), R(t-1), Q(t)]

    _MODEL_SAVING_PATH = 'savings/'
    _PLOTS_SAVING_PATH = 'plots/'





