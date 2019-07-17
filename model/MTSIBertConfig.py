
"""
A module containing all the infos for the project
"""


class KvretConfig():
    """
    Kvret Dataset configuration parameters
    """
    _KVRET_TRAIN_PATH = 'MTSI-BERT/dataset/kvret_dataset_public/kvret_train_public.json'
    _KVRET_VAL_PATH = 'MTSI-BERT/dataset/kvret_dataset_public/kvret_dev_public.json'
    _KVRET_TEST_PATH = 'MTSI-BERT/dataset/kvret_dataset_public/kvret_test_public.json'
    _KVRET_ENTITIES_PATH = 'MTSI-BERT/dataset/kvret_dataset_public/kvret_entities.json'
    
    _KVRET_MAX_BERT_TOKEN_PER_TRAIN_SENTENCE = 113
    _KVRET_MAX_BERT_TOKEN_PER_VAL_SENTENCE = 111
    _KVRET_MAX_BERT_TOKEN_PER_TEST_SENTENCE = 50


    """
    MTSI-Bert model parameters for Kvret dataset
    """
    _N_LABELS = 3
    _BATCH_SIZE = 128
    _LAYERS_NUM = 2






