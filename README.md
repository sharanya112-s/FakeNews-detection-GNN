# Propagation-Aware Fake News Detection Using Graph Neural Networks

This project studies whether combining news text with the way news spreads on social media improves fake news detection, using the UPFD PolitiFact and GossipCop datasets. It compares two baselines and three Graph Neural Network (GNN) architectures, then tests whether adding explicit propagation statistics improves the best GNN.

The implementation is based on the UPFD framework ([GNN-FakeNews](https://github.com/safe-graph/GNN-FakeNews)). This is a comparative experimental study with an extension. It does not propose a new detection algorithm.

## Research questions

1. How effective is news text alone for detecting fake news?
2. How much useful information does the propagation structure contain?
3. Does combining text and propagation information improve classification?
4. Do explicit propagation statistics add value when combined with the best GNN?

## Experiments

| Experiment | Model | Information used |
|---|---|---|
| Baseline 1 | Text-only MLP | BERT vector of the news item |
| Baseline 2 | Random Forest | Propagation statistics |
| Comparison | GCN, GAT, GraphSAGE | BERT features and propagation graph |
| Extension | Validation-selected GNN + statistics | BERT features, propagation graph and explicit statistics |

All models use the official data splits and five random seeds (42 to 46). The best GNN is chosen by validation macro-F1, not by test results.

## Repository structure

```text
.
├── README.md
├── REFERENCES.md
├── requirements.txt
├── data/                  Dataset location (see data/README.md)
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_baselines.ipynb
│   ├── 03_gnn_comparison.ipynb
│   ├── 04_propagation_extension.ipynb
│   ├── 05_final_results_demo.ipynb
│   └── upfd_common.py     Shared settings, data loader and graph statistics
└── results/
    ├── figures/
    ├── logs/
    └── metrics/
```

Model checkpoint files are not stored in the repository. Running the notebooks recreates them in `results/checkpoints/`.

## Setup

The experiments were run on Python 3.13 on a CPU. A GPU is used automatically if one is available.

```bash
git clone <repository-url>
cd <repository-folder>
python -m venv .venv
```

Activate the environment:

```bash
# Windows
.venv\Scripts\activate

# macOS or Linux
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Then download the dataset by following [data/README.md](data/README.md).

## Running the project

Open the notebooks from the `notebooks` folder and run them in order. Paths are detected automatically, so no paths need to be edited.

| Notebook | Purpose | Needs the dataset? |
|---|---|---|
| 01 | Dataset checks and exploration | Yes |
| 02 | Text-only and propagation-only baselines | Yes |
| 03 | GCN, GAT and GraphSAGE comparison | Yes |
| 04 | Propagation-statistics extension | Yes |
| 05 | Final results from the saved metrics | No |

To view the final results only, run notebook 05. Running notebooks 02 to 04 again overwrites the saved results. Exact values can vary slightly between machines and library versions.

## Results

Mean test results over five seeds:

| Model | PolitiFact macro-F1 | PolitiFact ROC-AUC | GossipCop macro-F1 | GossipCop ROC-AUC |
|---|---:|---:|---:|---:|
| Text-only MLP | 0.812 | 0.863 | 0.739 | 0.811 |
| Propagation-only Random Forest | 0.486 | 0.559 | 0.815 | 0.888 |
| GCN | 0.824 | 0.888 | 0.949 | 0.985 |
| GAT | 0.822 | 0.881 | 0.963 | 0.991 |
| GraphSAGE | 0.822 | 0.885 | 0.966 | 0.994 |

GraphSAGE was selected by validation macro-F1 on both datasets.

Propagation-statistics extension (GraphSAGE, test macro-F1):

| Dataset | Without statistics | With statistics | Mean paired difference | Paired t-test p |
|---|---:|---:|---:|---:|
| PolitiFact | 0.828 | 0.811 | -0.017 | 0.559 |
| GossipCop | 0.966 | 0.965 | -0.0005 | 0.827 |

### Main findings

- Text alone performs well on PolitiFact but is weaker on GossipCop.
- Propagation structure alone is close to chance on PolitiFact and much stronger on GossipCop.
- Combining text and propagation clearly improves GossipCop. On PolitiFact the gain over text alone is small and within the variation between seeds.
- The explicit propagation statistics gave no consistent benefit. The PolitiFact decrease comes mainly from one seed.

## Limitations

- PolitiFact is small (62 training, 31 validation and 221 test graphs), so differences between models there are uncertain.
- Five seeds give limited statistical power, so the paired t-tests are exploratory.
- The meaning of class 0 and class 1 (fake or real) was not independently verified.
- Some graphs share an identical news vector, so similar news content may appear more than once.
- The MLP stops early on validation loss, while the GNNs stop early on validation macro-F1.
- This implementation keeps the edge direction as stored in the dataset and uses early stopping, so its results are not an exact replication of the published UPFD numbers.
- The GraphSAGE runs in notebooks 03 and 04 are separate runs, so their values differ slightly (PolitiFact macro-F1 0.822 and 0.828).

## Reference

Dou, Y., Shu, K., Xia, C., Yu, P. S., and Sun, L. (2021). *User Preference-aware Fake News Detection.* SIGIR 2021. Further references are in [REFERENCES.md](REFERENCES.md).
