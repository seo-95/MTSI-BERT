import pdb
import sys
sys.path.insert(1, 'model/')
import spacy
from ner_kvret import NERKvretDataset
from MTSIBertConfig import KvretConfig
from nltk.translate.bleu_score import sentence_bleu
import numpy as np

_SPACY_MODEL_SAVING_PATH = 'spaCy/spaCy_savings/'


def spacy_test(spacy_model, data):

    doc = nlp('Where is the nearest gas station?')

    missing_entities = 0
    bleu_scores = []

    for t_sample in data:
        utt = t_sample[0]
        ents_l = t_sample[1]['entities']
        ents_l.sort(key=lambda tup: tup[0])  # sorts in place the entities list

        doc = nlp(utt) # make prediction

        #if len(doc.ents) > len(ents_l):
            #pdb.set_trace()
        #assert len(ents_l) >= len(doc.ents), 'PREDICTED MORE ENTITIES THAN REQUESTED'
        if len(doc.ents) > len(ents_l):
            missing_entities += (len(doc.ents) - len(ents_l))

        for pred, truth in zip(doc.ents, ents_l):
            start_idx = truth[0]
            end_idx = truth[1]
            curr_bleu = sentence_bleu(references=utt[start_idx:end_idx], hypothesis=pred)
            bleu_scores.append(curr_bleu)

    print('BLEU: '+str(np.mean(bleu_scores)))
    print('missing: '+str(missing_entities))






if __name__ == '__main__':
    
    test_set = NERKvretDataset(KvretConfig._KVRET_TEST_PATH)
    stest_set = test_set.build_spacy_dataset()

    nlp = spacy.load(_SPACY_MODEL_SAVING_PATH+'spacy_2019-08-25T22:23:47.579104')

    spacy_test(nlp, stest_set)

