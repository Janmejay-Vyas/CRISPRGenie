import torch
import torch.nn as nn
import torch.nn.functional as F

def custom_loss(model_output, labels, tokenizer):
    
    outputs = model_output.logits

    # Find indices of [SOS] and [EOS] tokens
    sos_id = tokenizer.token_to_id['[SOS]']
    eos_id = tokenizer.token_to_id['[EOS]']

    loss = 0
    batch_size = outputs.size(0)
    for i in range(batch_size):
        # Extract the sequence between [SOS] and [EOS]
        start = (labels[i] == sos_id).nonzero(as_tuple=True)[0]
        end = (labels[i] == eos_id).nonzero(as_tuple=True)[0]
        if start.nelement() == 0 or end.nelement() == 0:
            continue  # Skip if [SOS] or [EOS] not found
        if start.item() >= end.item():
            continue  # Ensure valid range
        # Calculate loss only within the [SOS] and [EOS] range
        relevant_outputs = outputs[i, start:end, :]
        relevant_labels = labels[i, start:end]
        loss += F.cross_entropy(relevant_outputs, relevant_labels, reduction='mean')
    return loss / batch_size