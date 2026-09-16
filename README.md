# BacPT
**Bacterial Proteome Foundation Model for Enhanced Functional Prediction**

Code for the manuscript: _Bacterial proteome foundation model enhances functional prediction from enzymes to ecological interactions_ ([preprint](references/BacPT_reference.pdf))

BacPT is a proteome foundation model trained on tens of thousands of complete bacterial genomes. It represents a genome as an ordered sequence of ESM2 protein embeddings and learns contextualized, genome-aware gene representations through a self-supervised reconstruction objective. Two model variants are described in the paper:

- **BacPT-small** — RoBERTa backbone, relative key-query position embeddings, trained end-to-end at whole-genome scale (up to 5,000 genes).
- **BacPT-large** — RoFormer backbone with Rotary Position Embeddings (RoPE), trained in two stages (short-contig pretraining followed by whole-genome fine-tuning).

---

## Repository structure

```
BacPT/
├── src/            # Model architectures and dataset classes
├── training/        # Pretraining scripts for BacPT-small and BacPT-large
├── inference/        # Model loading and embedding-generation utilities
├── data/             # Small demo data for the quickstart examples
├── notebooks/        # Downstream analyses, one directory per application
└── references/       # Manuscript PDF
```

### Downstream applications (`notebooks/`)

Each subdirectory corresponds to one of the downstream applications described in the Results section of the manuscript:

| Directory | Paper section / figure |
|---|---|
| `notebooks/enzyme_activity/` | Enzyme activity prediction (Figure 2) |
| `notebooks/operon_classification/` | Operon classification (Figure 3A) |
| `notebooks/gene_clusters_bgc/` | Biosynthetic gene cluster identification (Figure 3E–H) |
| `notebooks/gene_interactions/` | Gene interaction / Jacobian analysis and STRING benchmarking (Figure 3B–D) |
| `notebooks/trait_prediction/` | Bacterial metabolic trait prediction (Figure 4) |
| `notebooks/ecological_interactions/` | Ecological interaction outcome prediction (Figure 5) |
| `notebooks/genome_scaffolding/` | Genome scaffolding from contigs (Figure 1H) |

Directories without notebooks yet are placeholders (marked with `.gitkeep`) pending upload.

---

## Model weights

BacPT model weights are public now:

- [BacPT-small](https://huggingface.co/palsetuf/BacPT-small)
- [BacPT-large](https://huggingface.co/palsetuf/BacPT-large)

Please follow the instructions in the corresponding Hugging Face repository to
install the required packages and run each model.

---

## Citation

[Citation information will be added upon publication]
