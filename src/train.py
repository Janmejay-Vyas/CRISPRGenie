"""Training function definitions"""

from tqdm import tqdm
import torch
from utils.loss import custom_loss
from utils.utils import calculate_accuracy
import json

# Define the training function
def train(model, train_loader, val_loader, tokenizer, optimizer, epochs, device, log_file_path):
    model.train()
    results = {
        'train_loss': [],
        'val_loss': [],
        'train_accuracy': [],
        'val_accuracy': []
    }

    for epoch in range(epochs):
        total_loss = 0
        total_accuracy = 0
        train_steps = len(train_loader)
        val_steps = len(val_loader)

        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training", unit="batch")
        
        for step, batch in enumerate(train_progress):
            inputs, labels = batch[:, :-1], batch[:, 1:]
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = custom_loss(outputs, labels, tokenizer)
            loss.backward()
            optimizer.step()

            accuracy = calculate_accuracy(outputs.logits, labels, tokenizer)

            total_loss += loss.item()
            total_accuracy += accuracy

            train_progress.set_postfix({"Step": step + 1, "Loss": loss.item(), "Accuracy": accuracy})

        avg_train_loss = total_loss / train_steps
        avg_train_accuracy = total_accuracy / train_steps
        results['train_loss'].append(avg_train_loss)
        results['train_accuracy'].append(avg_train_accuracy)

        model.eval()
        val_loss = 0
        val_accuracy = 0
        val_progress = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation", unit="batch")
        
        with torch.no_grad():
            for step, batch in enumerate(val_progress):
                inputs, labels = batch[:, :-1], batch[:, 1:]
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = custom_loss(outputs, labels, tokenizer)
                accuracy = calculate_accuracy(outputs.logits, labels, tokenizer)

                val_loss += loss.item()
                val_accuracy += accuracy
                val_progress.set_postfix({"Step": step + 1, "Loss": loss.item(), "Accuracy": accuracy})

        avg_val_loss = val_loss / val_steps
        avg_val_accuracy = val_accuracy / val_steps
        results['val_loss'].append(avg_val_loss)
        results['val_accuracy'].append(avg_val_accuracy)

        # Log results to a file after each epoch
        with open(log_file_path, 'a') as log_file:
            log_file.write(json.dumps({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'train_accuracy': avg_train_accuracy,
                'val_loss': avg_val_loss,
                'val_accuracy': avg_val_accuracy
            }) + '\n')

    return results