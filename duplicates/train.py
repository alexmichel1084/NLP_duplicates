import pandas as pd
import torch
import pickle
import sklearn
from transformers import DistilBertTokenizer, BertForSequenceClassification, AdamW
from src import get_data, to_train_model
from torch.utils.data import DataLoader

tokenizer = DistilBertTokenizer.from_pretrained(
    "distilbert-base-uncased"
)
config = {
    "num_classes": 2,
    "dropout_rate": 0.1,
}

model = BertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased"
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
model.to(device)

train_text1, val_text1, train_text2, val_text2, train_labels, val_labels = get_data.PrepareData(
    pd.read_csv('datasets/train.csv', nrows=1000), ).get_data()

# create object for training and test samples
train_dataset = to_train_model.DuplicateDataset(train_text1, train_text2, train_labels, tokenizer, device)
val_dataset = to_train_model.DuplicateDataset(val_text1, val_text2, val_labels, tokenizer, device)

# create DataLoader to train model
train_loader = DataLoader(train_dataset, batch_size=51, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=51, shuffle=False)

# define parameters to train
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=1e-2)
num_epochs = 2

with open('models/model2', 'wb') as f:
    pickle.dump(model, f)

to_train_model.train(num_epochs, model, train_loader, optimizer)
predict = to_train_model.eval(model, val_loader)
predict = torch.Tensor(predict)
val_labels = torch.Tensor(val_labels)
z = torch.zeros(predict.shape)
z[:int(val_labels.shape[0])] = val_labels
to_train_model.calculate(z, predict)

print(sklearn.metrics.classification_report(predict.tolist(), z.tolist()))
