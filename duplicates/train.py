import pandas as pd
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

train_text1, val_text1, train_text2, val_text2, train_labels, val_labels = get_data.PrepareData(
    pd.read_csv('datasets/train.csv', nrows=100)).get_data()

# create object for training and test samples
train_dataset = to_train_model.DuplicateDataset(train_text1, train_text2, train_labels, tokenizer, 128)
val_dataset = to_train_model.DuplicateDataset(val_text1, val_text2, val_labels, tokenizer, 128)

# create DataLoader to train model
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# define parameters to train
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=1e-2)
num_epochs = 2

to_train_model.train(num_epochs, model, train_loader, optimizer)
to_train_model.calculate(val_labels, to_train_model.eval(model, val_loader))

# safe model
import pickle
with open('models/model', 'wb') as f:
    pickle.dump(model, f)
