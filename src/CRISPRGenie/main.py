"""Main module for CRISPRGenie."""

# Importing the required libraries
import os
from datetime import datetime
import time
import torch
import pandas as pd
import math
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from CRISPRGenie.models.tokenizer import CustomTokenizer
from CRISPRGenie.utils import load_config, get_lr
from CRISPRGenie.models.gpt import GPT, GPTConfig, ddp, ddp_local_rank, ddp_world_size, master_process, device_type, device, ddp_rank
from CRISPRGenie.data.dataloader import DataLoaderLite

def train_genie():
    
    # Initializing the batch size and gradient accumulation steps
    total_batch_size = 524288
    B = 64
    T = 1024
    assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
    grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
    if master_process:
        print(f"total desired batch size: {total_batch_size}")
        print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    # Loading the dataset 
    df = pd.read_csv('src/CRISPRGenie/data/GenomeCRISPR.csv')
    gene_ids = df['ensg'].tolist()
    sequences = df['sequence'].tolist()
    tokenizer = CustomTokenizer()
    
    # Creating the train and validation dataloaders
    train_loader = DataLoaderLite(gene_ids, sequences, B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, tokenizer=tokenizer, split="train")
    val_loader = DataLoaderLite(gene_ids, sequences, B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, tokenizer=tokenizer, split="val")
    
    # Set float32 precision
    torch.set_float32_matmul_precision('high')

    # Define the GPT model
    model = GPT(GPTConfig(vocab_size=tokenizer.vocab_size, block_size=T))
    model.to(device)
    use_compile = False # True
    if use_compile:
        model = torch.compile(model)
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model

    # Setting the hyperparameters
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

    # Setting the optimizer
    optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device_type)

    # Logdir setup
    log_dir = "./src/CRISPRGenie/results/logs/"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{datetime.now().strftime('%Y-%m-%d')}_log.txt")
    with open(log_file, "w") as f:
        pass

    # Training the model
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
        dist.destroy_process_group()


if __name__ == "__main__":
    train_genie()