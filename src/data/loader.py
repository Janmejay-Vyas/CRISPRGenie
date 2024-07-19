"""Module for loading the dataset"""

# Importing the required libraries
import pandas as pd

def load_data(filepath):
    """Load the CSV file containing the sgRNA sequences, ENSEMBL IDs and their respective 
    log2fc values"""
    df = pd.read_csv(filepath)
    df = df[['ensg', 'sequence', 'log2fc']]
    return df
