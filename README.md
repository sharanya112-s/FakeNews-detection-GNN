# FakeNews-detection-GNN

# Propagation-Aware Fake News Detection Using Graph Neural Networks

## Project Overview

This project investigates the use of Graph Neural Networks (GNNs) for fake news detection by combining textual information with social-media propagation patterns.

We compare different GNN architectures and investigate whether additional propagation-structure statistics can improve classification performance.

## Datasets

We use the UPFD benchmark datasets:

- PolitiFact
- GossipCop

These are graph-based subsets derived from the FakeNewsNet collection.

The datasets provide news propagation graphs, node features, labels, and predefined data splits.

See [data/README.md](data/README.md) for dataset information and download instructions.

## Models

The planned experiments include:

1. Text-only MLP baseline
2. Propagation-statistics Random Forest baseline
3. Graph Convolutional Network (GCN)
4. Graph Attention Network (GAT)
5. GraphSAGE
6. Extended model combining the selected GNN with explicit propagation statistics

## Repository Structure

- `data/`: Dataset instructions and local dataset storage
- `configs/`: Dataset and experiment configurations
- `notebooks/`: Jupyter notebooks for exploration and demonstration
- `src/`: Reusable project implementation
- `scripts/`: Experiment execution scripts
- `tests/`: Basic implementation tests
- `results/`: Experimental metrics and visualizations
- `docs/`: Project documentation
- `deliverables/`: Final report, presentation, and video details

## Installation

Installation instructions will be added after the project environment is tested and the software versions are finalized.

## Execution

Detailed dataset setup, training, and evaluation instructions will be provided here.

## Research References

The project uses the UPFD framework and associated research implementations as its foundation.

See [REFERENCES.md](REFERENCES.md).

## Project Status

Under development. Experimental results will be added only after successful execution and evaluation.
