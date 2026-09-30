import time
from pathlib import Path
import torch
import torch.nn as nn


def train_model(model, loaders, device, *, epochs, learning_rate, weight_decay=0.0,
                scheduler_patience=2, scheduler_factor=0.5, min_lr=1e-6,
                early_stopping_patience=5, checkpoint_path=None):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=scheduler_factor, patience=scheduler_patience, min_lr=min_lr
    )
    if checkpoint_path:
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

    best_val = -1.0
    best_epoch = 0
    no_improve = 0
    history = []
    start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = train_correct = train_total = 0
        for images, labels in loaders["train"]:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * labels.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()
            train_total += labels.size(0)

        model.eval()
        val_loss_sum = val_correct = val_total = 0
        with torch.no_grad():
            for images, labels in loaders["validation"]:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss_sum += loss.item() * labels.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        train_loss = train_loss_sum / train_total
        train_acc = train_correct / train_total
        val_loss = val_loss_sum / val_total
        val_acc = val_correct / val_total
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)

        row = {"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_acc,
               "validation_loss": val_loss, "validation_accuracy": val_acc, "learning_rate": current_lr,
               "generalization_gap": train_acc - val_acc}
        history.append(row)
        print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | LR: {current_lr:.2e}")

        if val_acc > best_val:
            best_val, best_epoch, no_improve = val_acc, epoch, 0
            if checkpoint_path:
                torch.save(model.state_dict(), checkpoint_path)
                print(f"  Best model saved (Val Acc = {val_acc:.4f})")
        else:
            no_improve += 1
            if no_improve >= early_stopping_patience:
                print(f"  Early stopping at epoch {epoch} (patience={early_stopping_patience})")
                break

    return {"training_time_seconds": time.perf_counter() - start,
            "best_validation_accuracy": best_val, "best_epoch": best_epoch,
            "epochs_completed": len(history), "training_history": history}
