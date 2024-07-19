"""Utility functions"""

import torch

def calculate_accuracy(logits, labels, tokenizer):
    preds = torch.argmax(logits, dim=-1)
    mask = (labels != tokenizer.token_to_id['[PAD]']).float()
    correct = ((preds == labels) * mask).sum().item()
    total = mask.sum().item()
    return correct / total