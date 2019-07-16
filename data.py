import pandas as pd
import json


_DATASET = '../dataset/kvret_dataset_public/kvret_dev_public.json'

with open(_DATASET) as json_dataset:
    data = json.load(json_dataset)

df = pd.read_json(_DATASET)

print(list(df))
print('end')