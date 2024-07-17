"""Script to create a dataset for the sgRNA sequences and their respective input labels"""

# importing the necessary libraries
import torch
from torch.utils.data import Dataset

class GeneSequenceDataset(Dataset):
    def __init__(self, gene_ids, sequences, tokenizer):
        self.gene_ids = gene_ids
        self.sequences = sequences
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.gene_ids)

    def __getitem__(self, idx):
        # Form the full sequence with special tokens
        full_sequence = f"[ID]{self.gene_ids[idx]}[SOS]{self.sequences[idx]}[EOS]"
        tokenized_sequence = self.tokenizer.encode(full_sequence)
        return torch.tensor(tokenized_sequence, dtype=torch.long)
    
def collate_fn(batch):
    batch_padded = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0)
    return batch_padded