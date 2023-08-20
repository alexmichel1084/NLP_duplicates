import torch
import pickle
from scipy.special import softmax
from fuzzywuzzy import fuzz
from transformers import DistilBertTokenizer


def clean_entity(noizy_entity, model):
    copy = {}
    for i in range(len(noizy_entity)):
        if i + 1 == len(noizy_entity):
            break
        for j in range(i, len(noizy_entity)):
            if model.score_phrase(noizy_entity[i], noizy_entity[j]) == 1:
                copy[noizy_entity[i]] = noizy_entity[j]
                noizy_entity = noizy_entity[:j] + noizy_entity[j:]
            if j + 1 == len(noizy_entity):
                break
    return noizy_entity, copy


def create_token(text1, text2, tokenizer):
    tokens1 = tokenizer.tokenize(text1)
    tokens2 = tokenizer.tokenize(text2)

    distance = fuzz.ratio(text1.lower(), text2.lower())
    partial_distance = fuzz.partial_ratio(text1.lower(), text2.lower())

    tokens = ['[CLS]'] + tokens1 + ['[SEP]'] + tokens2 + ['[SEP]'] + [distance] + ['[SEP]'] + [partial_distance] + [
        '[SEP]']

    return {'input_ids': torch.tensor([tokenizer.convert_tokens_to_ids(tokens)]),
            'attention_mask': torch.tensor([[1] * len(tokenizer.convert_tokens_to_ids(tokens))])}


class Model:
    def __init__(self, model_path="models/model"):
        with open(model_path, "rb") as file:
            self.model = pickle.load(file)
        self.tokenizer = DistilBertTokenizer.from_pretrained(
            "distilbert-base-uncased"
        )

    def score_phrase(self, text1, text2):
        output = self.model(**create_token(text1, text2, self.tokenizer))

        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        return scores.argmax()
