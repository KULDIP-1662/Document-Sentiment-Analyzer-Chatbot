from preprocessing import preprocess_data
from dataset import TextDataset
from model import ModernBertClassifier
from train import train_model
from config import *
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from evaluate import evaluate_model

df, label_encoder = preprocess_data(DATA_PATH)

train_texts, temp_texts, train_labels, temp_labels = train_test_split(df['text'].tolist(), df['label'].tolist(), test_size=0.3, random_state=1)
val_texts, test_texts, val_labels, test_labels = train_test_split(temp_texts, temp_labels, test_size=0.5, random_state=42)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_dataset = TextDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
val_dataset = TextDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)
test_dataset = TextDataset(test_texts, test_labels, tokenizer, MAX_LENGTH)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

model = ModernBertClassifier(len(label_encoder.classes_), MODEL_NAME).to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.005)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 15], gamma=0.1)
criterion = torch.nn.CrossEntropyLoss()

train_model(model, train_loader, val_loader, EPOCHS, optimizer, criterion, scheduler)
evaluate_model(model, test_loader, DEVICE)