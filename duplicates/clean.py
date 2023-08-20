from transformers import DistilBertTokenizer
from src import to_work_with_model
import pandas as pd
import random

tokenizer = DistilBertTokenizer.from_pretrained(
    "distilbert-base-uncased"
)

model = to_work_with_model.Model()

# example of dirty files
data2 = pd.read_csv('datasets/train.csv')

data_true = data2[data2['is_duplicate'] == 1]
data_false = data2[data2['is_duplicate'] == 0]
data_true.iloc[2]

noizy_entity = []
for i in random.sample(range(1, 100),73):
    noizy_entity.append(data_true.iloc[i]['name_1'])
    noizy_entity.append(data_true.iloc[i]['name_2'])
    noizy_entity.append(data_false.iloc[i]['name_1'])
    noizy_entity.append(data_false.iloc[i]['name_2'])

#print(noizy_entity)

print(to_work_with_model.clean_entity(noizy_entity, model))
