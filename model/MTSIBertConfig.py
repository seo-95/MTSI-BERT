
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

    """
    MTSI-Bert model parameters for Kvret dataset
    """
    _N_LABELS = 3
    _BATCH_SIZE = 128
    _LAYERS_NUM = 2






