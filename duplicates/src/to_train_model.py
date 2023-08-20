import torch
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from fuzzywuzzy import fuzz


# class for dataset
class DuplicateDataset(Dataset):
    def __init__(self, text1, text2, labels, tokenizer, max_seq_len):
        self.text1 = text1
        self.text2 = text2
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.text1)

    def __getitem__(self, idx):
        tokens1 = self.tokenizer.tokenize(self.text1[idx])
        tokens2 = self.tokenizer.tokenize(self.text2[idx])

        distance = fuzz.ratio(self.text1[idx].lower(), self.text2[idx].lower())
        partial_distance = fuzz.partial_ratio(self.text1[idx].lower(), self.text2[idx].lower())

        tokens = ['[CLS]'] + tokens1 + ['[SEP]'] + tokens2 + ['[SEP]'] + [distance] + ['[SEP]'] + [partial_distance] + [
            '[SEP]']
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return {
            'text': str(self.text1[idx]) + str(self.text2[idx]),
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor([[1] * len(input_ids)]),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }


# func to train model
def train(num_epochs, model, train_loader, optimizer):
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for i, batch in enumerate(train_loader):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['label']

            optimizer.zero_grad()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

        print(f'Epoch: {epoch}, Training Loss: {train_loss / len(train_loader)}')


# Evaluation of the model on a test sample
def eval(model, val_loader):
    model.eval()
    with torch.no_grad():
        val_preds = []
        for batch in val_loader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            val_preds.extend(preds.cpu().detach().numpy().tolist())
    return val_preds


# Quality Assessment
def calculate(val_labels, val_preds):
    val_accuracy = accuracy_score(val_labels, val_preds)
    val_precision, val_recall, val_fscore, _ = precision_recall_fscore_support(val_labels, val_preds,
                                                                               average='weighted')
    print(f"Accuracy: {val_accuracy}, Precision: {val_precision}, Recall: {val_recall}")
