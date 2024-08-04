"""Script to create a dataset for the sgRNA sequences and their respective input labels"""

# Importing the required libraries
import torch

class DataLoaderLite:
    def __init__(self, gene_ids, sequences, B, T, process_rank, num_processes, tokenizer, split):
        self.B = B
        self.T = T
        self.tokenizer = tokenizer
        self.current_position = 0
        self.pad_token_id = tokenizer.token_to_id['[PAD]']

        split_idx = int(len(gene_ids) * 0.8)
        if split == 'train':
            gene_ids = gene_ids[:split_idx]
            sequences = sequences[:split_idx]
        elif split == 'val':
            gene_ids = gene_ids[split_idx:]
            sequences = sequences[split_idx:]

        self.batches = []
        for i in range(0, len(gene_ids), B):
            batch_gene_ids = gene_ids[i:i+B]
            batch_sequences = sequences[i:i+B]
            max_len = 0
            batch_encoded = []

            for gene_id, seq in zip(batch_gene_ids, batch_sequences):
                formatted_seq = f'{gene_id}[SOS]{seq}[EOS]'
                encoded = tokenizer.encode(formatted_seq)
                max_len = max(max_len, len(encoded))
                batch_encoded.append(encoded)

            padded_batch = []
            for encoded in batch_encoded:
                padded_length = max_len - len(encoded)
                padded_seq = encoded + [self.pad_token_id] * padded_length
                padded_batch.append(padded_seq)

            self.batches.append(torch.tensor(padded_batch, dtype=torch.long))

        self.num_batches = len(self.batches)
        self.current_shard = 0
        self.process_rank = process_rank
        self.num_processes = num_processes

    def reset(self):
        self.current_position = 0

    def next_batch(self):
        if self.current_position >= self.num_batches:
            self.current_position = 0
            return None, None

        batch = self.batches[self.current_position]
        self.current_position += 1
        targets = batch.roll(-1, dims=1)
        return batch, targets

