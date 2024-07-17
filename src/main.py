"""Main module"""

# Importing the required libraries
import os
import torch
import yaml
from tqdm import tqdm
from data.loader import load_data
from data.dataset import GeneSequenceDataset, collate_fn
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Config
from models.tokenizer import CustomTokenizer
from utils.loss import custom_loss
from utils.utils import calculate_accuracy
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from train import train
from datetime import datetime

def train_sequence_generation(config_path):
    """Train the autoregressive GPT model to generate sgRNA sequences"""

    # Define the device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')  

    # Load the arguments from config
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Load the dataset
    df_seq = load_data(config['data']['filepath'])

    # Split the data into train and val sets
    train_size = config['train_size']
    train_df, val_df = train_test_split(df_seq, test_size=train_size, random_state=42)
    

    # Extract the sgRNA sequence and gene ids
    train_sequences = train_df['sequence'].tolist()
    train_gene_ids = train_df['ensg'].tolist()

    valid_sequences = val_df['sequence'].tolist()
    valid_gene_ids = val_df['ensg'].tolist()
    
    # Define the model and tokenizer
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    tokenizer = CustomTokenizer()
    vocab_size = len(tokenizer.token_to_id)
    model.resize_token_embeddings(vocab_size)

    # Create the train and valid dataloader
    train_dataset = GeneSequenceDataset(train_gene_ids, train_sequences, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, collate_fn=collate_fn)

    val_dataset = GeneSequenceDataset(valid_gene_ids, valid_sequences, tokenizer)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=True, collate_fn=collate_fn)

    # Define the optimizer
    optimizer = AdamW(model.parameters(), lr=3e-5)

    # Get the current date
    current_date = datetime.now().strftime("%Y-%m-%d")

    # Train the model
    epochs = config['training_args']['epochs']
    results = train(model, train_loader, val_loader, tokenizer, optimizer, epochs, device, log_file_path=f"results/logs/{current_date}.log")

    # Save the trained model
    save_path = config['model']['save_path']
    model_path = f"{save_path}/{current_date}"
    os.makedir(model_path, exist_ok = True)
    torch.save(model, f"{model_path}/sgRNA_model.pth")

    print("Training Completed!")

if __name__ == "__main__":
    train_sequence_generation("../config/config.yaml")