import os
import datetime
import pdb
import sys
sys.path.insert(1, 'model/')
import time

from torch.utils.data import DataLoader

from MTSIBertConfig import KvretConfig
from ner_kvret import NERKvretDataset

import spacy
import random




_N_EPOCHS = 120
_SPACY_MODEL_SAVING_PATH = 'ner/spacy_savings/'
_BATCH_SIZE = 32


def spacy_train(data):
    

    nlp = spacy.blank('en')  # create blank Language class
    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)

    # add labels
    for _, annotations in data:
         for ent in annotations.get('entities'):
            ner.add_label(ent[2])


 # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        optimizer = nlp.begin_training()

        for epoch in range(_N_EPOCHS):
            random.shuffle(straining_set)
            losses = {}
            curr_batch_text = []
            curr_batch_label = []
            for idx, (text, annotations) in enumerate(data):
                if idx % _BATCH_SIZE == 0 or idx == len(data) - 1:
                    curr_batch_text.append(text)
                    curr_batch_label.append(annotations)
                nlp.update(
                    curr_batch_text,  # batch of texts
                    curr_batch_label,  # batch of annotations
                    drop=0.2,  # dropout
                    sgd=optimizer,  # callable to update weights
                    losses=losses)
                
                curr_batch_text = []
                curr_batch_label = []

            log_str = '### EPOCH '+str(epoch+1)+'/'+str(_N_EPOCHS)+':: TRAIN LOSS = '+str(losses)
            print(log_str)

    return nlp











if __name__ == '__main__':
    start = time.time()
    curr_date = datetime.datetime.now().isoformat()
    # creates the directory for the checkpoints
    os.makedirs(os.path.dirname(_SPACY_MODEL_SAVING_PATH), exist_ok=True)
    
    training_set = NERKvretDataset(KvretConfig._KVRET_TRAIN_PATH)
    straining_set = training_set.build_spacy_dataset()

    validation_set = NERKvretDataset(KvretConfig._KVRET_VAL_PATH)
    svalidation_set = validation_set.build_spacy_dataset()

    # train on both training and validation set
    spacy_model = spacy_train(straining_set+svalidation_set)
    spacy_model.to_disk(_SPACY_MODEL_SAVING_PATH+'spacy_'+curr_date)


    end = time.time()
    h_count = (end-start)/60/60
    print('training time: '+str(h_count)+'h')
