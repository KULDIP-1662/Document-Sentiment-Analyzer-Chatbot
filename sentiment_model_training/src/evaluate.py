import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
# from src.config import DEVICE

def evaluate_model(model, test_loader, device):
    model.eval()
    all_labels, all_preds = [], []

    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)

            outputs = model(**inputs)
            preds = outputs.argmax(dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    cm = confusion_matrix(all_labels, all_preds)

    print('\nEvaluation Results:\n')
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Macro F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)

