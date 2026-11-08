# Data

This folder holds the UPFD dataset used by the notebooks. The dataset files are not stored in this repository because they are too large for GitHub (for example, `gossipcop/new_bert_feature.npz` is about 874 MB). This README is the only tracked file in the folder.

## Download

1. Download the UPFD data (about 1.2 GB) from the link in the official repository: <https://github.com/safe-graph/GNN-FakeNews>
   Direct Google Drive folder: <https://drive.google.com/drive/folders/1OslTX91kLEYIi2WBnwuFtXsVz5SS_XeR?usp=sharing>
2. Unzip the PolitiFact and GossipCop archives.
3. Place the files as shown below.

## Expected layout

```text
data/
└── upfd/
    ├── politifact/
    │   ├── A.txt
    │   ├── graph_labels.npy
    │   ├── new_bert_feature.npz
    │   ├── node_graph_id.npy
    │   ├── train_idx.npy
    │   ├── val_idx.npy
    │   └── test_idx.npy
    └── gossipcop/
        └── (the same seven files)
```

The loader also accepts these files inside a `raw/` subfolder, for example `data/upfd/politifact/raw/`. The download also contains `new_content_feature.npz`, `new_profile_feature.npz` and `new_spacy_feature.npz`. This project does not use them.

## What each file contains

| File | Contents |
|---|---|
| `A.txt` | Edge list of all propagation graphs, one edge per line |
| `node_graph_id.npy` | The graph that each node belongs to |
| `graph_labels.npy` | The class label (0 or 1) of each graph |
| `new_bert_feature.npz` | 768-dimensional BERT vector for each node |
| `train_idx.npy`, `val_idx.npy`, `test_idx.npy` | Official train, validation and test graph indices |

## Dataset size

| Dataset | Graphs | Nodes | Edges | Train / validation / test |
|---|---:|---:|---:|---:|
| PolitiFact | 314 | 41,054 | 40,740 | 62 / 31 / 221 |
| GossipCop | 5,464 | 314,262 | 308,798 | 1,092 / 546 / 3,826 |

Each graph is one news item and the users who shared it on Twitter. Node 0 is the news item and the remaining nodes are users.

Class labels are reported as class 0 and class 1 because this project did not independently verify which class means fake and which means real.

## Source

Dou, Y., Shu, K., Xia, C., Yu, P. S., and Sun, L. (2021). *User Preference-aware Fake News Detection.* SIGIR 2021. <https://arxiv.org/abs/2104.12259>
