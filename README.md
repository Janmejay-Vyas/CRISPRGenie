# CRISPRGenie

## Description
This project aims to generate sgRNA sequences based on provided gene IDs using a GPT model. The model is trained to understand gene IDs and produce relevant sgRNA sequences, marked by special tokens to ensure proper format and context handling.

## Dataset
The dataset used for this project is downloaded from the [GenomeCRISPR Database](https://genomecrispr.dkfz.de/#!/download).

### Dataset Description
The data table contains the following columns:

* start: sgRNA start position
* end: sgRNA end position
* chr: Target chromosome
* strand: Target strand
* pubmed: PubMed ID of the screen's original publication
* cellline: The cell line model used in the screen.
* condition: The phenotype that was measured in the screen.
* sequence: The sgRNA sequence including PAM. The last three characters of the sequence are the PAM.
* symbol: HUGO gene symbol of the targeted gene
* ensg: ENSEMBL gene identifier of the targeted gene
* log2fc: Log2-scaled fold change between the end time point and time point zero for dropout (negative selection for viability) screens or treated and untreated * for positive selection screens (e.g. drug screens).
* rc_inital: Array of read counts at the 'initial' time point (either day 0 or untreated sample). There is one read count per technical replicate.
* rc_final: Array of read counts of the treated sample. There is one read count per technical replicate.
* effect: The sgRNA effect determined by GenomeCRISPR [-9, 9]
* cas: The Cas variant used in the screen
* screentype: The type of screen.


## Project Components

1. **Tokenizer**: Custom tokenizer to handle gene IDs and sgRNA sequences with special tokens.
2. **Dataset** Preparation: Formatting gene IDs and sequences into the required format for model training.
3. **Model Configuration**: Setting up a GPT model to process input sequences and generate sgRNA sequences.
4. **Training Loop**: Training the model with custom loss functions and ensuring predictions focus between [SOS] and [EOS] tokens.
5. **Sequence Generation**: Generating sgRNA sequences based on provided gene IDs during inference.

