import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from model.spam_detector import SpamLSTM
from model.data_loader import SpamDataset
from utils.class_weights import compute_class_weight
from utils.metrics import evaluate_metrics
from utils.preprocessing import clean_text
from collections import Counter
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load-------------------------------------------------------------------------------
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
X_train, X_temp, y_train, y_temp = train_test_split(texts, labels, test_size=0.3, stratify=labels)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp)
train_dataset = SpamDataset(X_train, y_train, vocab)
val_dataset = SpamDataset(X_val, y_val, vocab)
test_dataset = SpamDataset(X_test, y_test, vocab)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=64)
model = SpamLSTM(len(vocab)).to(device)
class_weights = compute_class_weight(y_train).to(device)
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
best_val_loss = float('inf')
patience = 3
counter_stop = 0
for epoch in range(15):
    model.train()
    total_loss = 0
    for texts_batch, labels_batch in train_loader:
        texts_batch, labels_batch = texts_batch.to(device), labels_batch.to(device)
        optimizer.zero_grad()
        outputs = model(texts_batch)
        loss = criterion(outputs, labels_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Train Loss: {total_loss/len(train_loader):.4f}")
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for texts_batch, labels_batch in val_loader:
            texts_batch, labels_batch = texts_batch.to(device), labels_batch.to(device)
            outputs = model(texts_batch)
            loss = criterion(outputs, labels_batch)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    print(f"Validation Loss: {val_loss:.4f}")
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_model.pth")
        counter_stop = 0
    else:
        counter_stop += 1
        if counter_stop >= patience:
            print("Early stopping triggered")
            break