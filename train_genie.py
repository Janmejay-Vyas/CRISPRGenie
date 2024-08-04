# NOTE: TEMP file, do not commit, delete later
import os
import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import pandas as pd

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    sos_token_id: int = 1
    eos_token_id: int = 2

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.ffn_head = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd),
            nn.ReLU(),
            nn.Linear(config.n_embd, config.n_embd)
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        sos_positions = (idx == self.config.sos_token_id).nonzero(as_tuple=True)
        gene_id_end_idx = sos_positions[1].min()
        gene_id_idx = idx[:, :gene_id_end_idx]
        sequence_idx = idx[:, gene_id_end_idx+1:]

        gene_id_emb = self.transformer.wte(gene_id_idx)
        gene_id_context = self.ffn_head(gene_id_emb.mean(dim=1))

        pos = torch.arange(0, sequence_idx.size(1), dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(sequence_idx)
        x = tok_emb + pos_emb.unsqueeze(0)
        x += gene_id_context.unsqueeze(1)

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        logits = logits.view(-1, self.config.vocab_size)

        loss = None
        if targets is not None:
            targets = targets[:, gene_id_end_idx+1:].contiguous().view(-1)
            if logits.size(0) != targets.size(0):
                raise ValueError(f"Logits and targets size mismatch: {logits.size(0)} vs {targets.size(0)}")
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if master_process:
            print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

# -----------------------------------------------------------------------------

import tiktoken
import numpy as np

class CustomTokenizer:
    def __init__(self):
        self.token_to_id = {
            '[PAD]': 0, '[SOS]': 1, '[EOS]': 2,
            'A': 3, 'T': 4, 'G': 5, 'C': 6,
            '[ID]': 7,
            '0': 8, '1': 9, '2': 10, '3': 11, '4': 12,
            '5': 13, '6': 14, '7': 15, '8': 16, '9': 17,
            '[PAD]': 18,
        }
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

    def encode(self, text):
        tokens = []
        i = 0
        while i < len(text):
            if text[i:i+4] == 'ENSG':
                tokens.append('[ID]')
                i += 4
            elif text[i] == '[':
                special_token_end = text.find(']', i)
                tokens.append(text[i:special_token_end+1])
                i = special_token_end + 1
            else:
                tokens.append(text[i])
                i += 1
        return [self.token_to_id.get(token, self.token_to_id['[PAD]']) for token in tokens]

    @property
    def vocab_size(self):
        return len(self.token_to_id)

    def decode(self, token_ids):
        return ''.join(self.id_to_token.get(token_id, '') for token_id in token_ids)

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

# -----------------------------------------------------------------------------
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    assert torch.cuda.is_available(), "CUDA is required for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

device_type = "cuda" if device.startswith("cuda") else "cpu"

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

enc = tiktoken.get_encoding("gpt2")

total_batch_size = 524288
B = 64
T = 1024
assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

df = pd.read_csv('./data/GenomeCRISPR.csv')
gene_ids = df['ensg'].tolist()
sequences = df['sequence'].tolist()
tokenizer = CustomTokenizer()

train_loader = DataLoaderLite(gene_ids, sequences, B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, tokenizer=tokenizer, split="train")
val_loader = DataLoaderLite(gene_ids, sequences, B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, tokenizer=tokenizer, split="val")

torch.set_float32_matmul_precision('high')

model = GPT(GPTConfig(vocab_size=tokenizer.vocab_size, block_size=T))
model.to(device)
use_compile = True # False
if use_compile:
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 19073
def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device_type)

log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f:
    pass

for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    # Evaluate validation loss
    if step % 250 == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                if x is None:
                    break
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"validation loss: {val_loss_accum.item():.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")
            if step > 0 and (step % 5000 == 0 or last_step):
                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'config': raw_model.config,
                    'step': step,
                    'val_loss': val_loss_accum.item()
                }
                torch.save(checkpoint, checkpoint_path)

    # TODO: Bug in lines 422-434, fix the bug and uncomment the code
    # # Generate text from the model once in a while
    # if (step > 0 and step % 2 == 0) or last_step: # TODO: testing generation with step % 2, replace with step % 250
    #     model.eval()
    #     num_return_sequences = 4
    #     max_length = 32
    #     tokens = tokenizer.encode("ENSG00000141510[SOS]")
    #     tokens = torch.tensor(tokens, dtype=torch.long)
    #     tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    #     xgen = tokens.to(device)
    #     print(xgen.shape)
    #     sample_rng = torch.Generator(device=device)
    #     sample_rng.manual_seed(42 + ddp_rank)
    #     while xgen.size(1) < max_length:
    #         with torch.no_grad():
    #             with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
    #                 xgen = model.transformer.wte(xgen) # Word token embeddings
    #                 pos = torch.arange(0, xgen.size(1), dtype=torch.long, device=device)
    #                 xgen = xgen + model.transformer.wpe(pos)
    #                 print(f"xgen shape after wte and wpe: {xgen.shape}")
    #                 for block in model.transformer.h:
    #                     xgen = block(x)
    #                 xgen = model.transformer.ln_f(xgen)
    #                 logits = model.lm_head(xgen)
    #             logits = logits[:, -1, :]
    #             probs = F.softmax(logits, dim=-1)
    #             topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
    #             ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
    #             xcol = torch.gather(topk_indices, -1, ix)
    #             xgen = torch.cat((xgen, xcol), dim=1)
    #     for i in range(num_return_sequences):
    #         tokens = xgen[i, :max_length].tolist()
    #         decoded = tokenizer.decode(tokens)
    #         print(f"rank {ddp_rank} sample {i}: {decoded}")

    # Training step
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        if x is None:
            train_loader.reset()
            x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    if device_type == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    if master_process:
        print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")

if ddp:
    destroy_process_group()