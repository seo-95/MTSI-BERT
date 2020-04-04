
import json
import os
import sys
import pdb

sys.path.append(os.path.realpath('..'))
print(sys.path)

from model import (KvretConfig, KvretDataset, MTSIAdapterDataset, MTSIBert,
                   MTSIKvretConfig, TwoSepTensorBuilder)









def identify_sessions(dialogue):
    pass
    #split the dialoue into triplets
    #fed these triplets to MTSI-BERT
    #collect the sessions inside a JSON file

    





if __name__ == '__main__':

    with open('input_dialogue.json') as json_dialogue:
        dialogue_data = json.load(json_dialogue)
    pdb.set_trace()
    identify_sessions(dialogue_data)


    #run flask on localhost
    #allocate the model before any call
    #inside the flask call do:
        #read input json
        #make forward
        #return the json
