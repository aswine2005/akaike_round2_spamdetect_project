import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter
from model.spam_detector import SpamLSTM
from model.data_loader import SpamDataset
from utils.preprocessing import clean_text
from utils.metrics import evaluate_metrics
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
df = pd.read_csv("data/spam.csv", encoding="latin-1")
df = df[['v1','v2']]
df.columns = ['label','text']
df['text'] = df['text'].apply(clean_text)
df['label'] = df['label'].map({'ham':0,'spam':1})
texts = df['text'].tolist()
labels = df['label'].tolist()
counter = Counter()
for text in texts:
    counter.update(text.split())
vocab = {"<PAD>":0,"<UNK>":1}
for word in counter:
    vocab[word] = len(vocab)
_, X_temp, _, y_temp = train_test_split(texts, labels, test_size=0.3, stratify=labels)
_, X_test, _, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp)
test_dataset = SpamDataset(X_test, y_test, vocab)
test_loader = DataLoader(test_dataset, batch_size=64)
model = SpamLSTM(len(vocab)).to(device)
model.load_state_dict(torch.load("best_model.pth"))
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for texts_batch, labels_batch in test_loader:
        texts_batch = texts_batch.to(device)
        outputs = model(texts_batch)
        predictions = torch.sigmoid(outputs)
        predictions = (predictions > 0.5).int().cpu().tolist()
        all_preds.extend(predictions)
        all_labels.extend(labels_batch.tolist())
precision, recall, f1, cm = evaluate_metrics(all_labels, all_preds)
print("\n------------")
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("Confusion Matrix:\n", cm)