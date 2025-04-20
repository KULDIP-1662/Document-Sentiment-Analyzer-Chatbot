import os
import torch
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from config import DEVICE, CHECKPOINT_DIR, FINE_TUNED_MODEL_PATH
import dill

def train_model(model, train_loader, val_loader, epochs, optimizer, criterion, scheduler, resume=False, checkpoint_path=None):
    start_epoch = 1

    if resume and checkpoint_path:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f'Resumed from epoch {start_epoch}')

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        total_train_loss = 0
        train_correct = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            inputs, labels = batch
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(**inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            train_correct += (outputs.argmax(dim=1) == labels).sum().item()

        train_acc = train_correct / len(train_loader.dataset)

        # Validation
        model.eval()
        val_correct, val_loss = 0, 0
        all_labels, all_preds = [], []

        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                labels = labels.to(DEVICE)

                outputs = model(**inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_correct += (outputs.argmax(dim=1) == labels).sum().item()

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(outputs.argmax(dim=1).cpu().numpy())

        scheduler.step()
        val_acc = val_correct / len(val_loader.dataset)
        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        cm = confusion_matrix(all_labels, all_preds)

        print('\nTraining Results: \n')
        print(f"Epoch {epoch} - Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, F1: {macro_f1:.4f}")
        print(f"Confusion Matrix:\n{cm}")

    torch.save(model, FINE_TUNED_MODEL_PATH, pickle_module=dill) 
        # if epoch % 3 == 0:
        #     torch.save({
        #         'epoch': epoch,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #     }, os.path.join(CHECKPOINT_DIR, f'model_epoch_{epoch}.pth'))
