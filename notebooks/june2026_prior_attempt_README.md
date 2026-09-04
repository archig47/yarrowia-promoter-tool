# yeast-promoter-lm

Fine-tuning a pretrained DNA language model to predict promoter strength 
in *Saccharomyces cerevisiae* using LoRA and synthetic data augmentation.

## Project Overview

This project investigates whether LoRA fine-tuning of a pretrained DNA 
language model (DNABERT-2 or Nucleotide Transformer) on a large-scale 
yeast promoter dataset can accurately predict promoter strength from 
sequence alone. A secondary aim is to evaluate whether augmenting the 
real training data with synthetically generated sequences improves 
model performance on held-out experimental data.

The project is motivated by the data scarcity problem in non-conventional 
yeasts (e.g. *Yarrowia lipolytica*), where the approach developed here 
on *S. cerevisiae* could in principle be extended with limited labelled data.

## Aims

1. Fine-tune a pretrained DNA LM on the de Boer et al. 2020 yeast promoter 
   dataset using LoRA and establish baseline prediction accuracy
2. Generate synthetic promoter sequences using the pretrained model and 
   annotate them with predicted expression values
3. Evaluate whether augmenting real training data with synthetic sequences 
   improves model performance on held-out experimental data
4. Validate learned representations biologically by comparing model 
   attribution scores to known TF binding site positions reported in 
   de Boer et al.

## Dataset

- **Source:** de Boer et al. 2020, Nature — "Deciphering eukaryotic 
  gene regulatory logic with 100 million random promoters"
- **GEO accession:** GSE104878
- **Sequences used:** pTpA scaffold, glucose condition
- **Input:** 80bp random DNA inserts
- **Label:** log2(YFP/RFP) — normalised continuous expression level

## Methods

- Pretrained model: DNABERT-2 / Nucleotide Transformer (HuggingFace)
- Fine-tuning: LoRA via HuggingFace PEFT library
- Framework: PyTorch, HuggingFace Transformers
- Compute: Google Colab Pro (T4/A100 GPU)

## Repository Structure
yeast-promoter-lm/

├── data/           # raw and processed dataset files

├── notebooks/      # Colab notebooks for each stage

├── src/            # reusable scripts (data loading, model, training)

├── results/        # evaluation outputs, figures

└── README.md

## Status

✅ Environment set up — data downloaded and inspected

## References

- de Boer et al. (2020). Deciphering eukaryotic gene regulatory logic 
  with 100 million random promoters. *Nature*.
- Ji et al. (2021). DNABERT: pre-trained Bidirectional Encoder 
  Representations from Transformers model for DNA-language in genome.
  *Bioinformatics*.
- Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models.
  *ICLR 2022*.
