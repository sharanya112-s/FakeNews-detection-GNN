"""Settings, paths, data loading and graph statistics shared by notebooks 01 to 04.

Keep this file in the same folder as the notebooks.
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import networkx as nx
from tqdm.auto import tqdm


# Settings shared by all notebooks.
SEED = 42
FEATURE = "bert"
EXPECTED_FEATURE_DIM = 768
DATASETS = ["politifact", "gossipcop"]
DATASET_TITLES = {"politifact": "PolitiFact", "gossipcop": "GossipCop"}
SPLITS = ["train", "val", "test"]


# The fake/real meaning of labels 0 and 1 is unverified, so classes are reported by index.
# Set this to 0 or 1 only once the mapping has been confirmed.
FAKE_CLASS_INDEX = None
if FAKE_CLASS_INDEX is None:
    CLASS_NAMES = {0: "class 0", 1: "class 1"}
else:
    CLASS_NAMES = {FAKE_CLASS_INDEX: "fake", 1 - FAKE_CLASS_INDEX: "real"}
CLASS_COLORS = {0: "#0072B2", 1: "#D55E00"}


# Find the project folder by walking up from the working directory.
def find_project_root(start=None):
    start = Path(start or Path.cwd()).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "data" / "upfd").is_dir():
            return candidate
    for candidate in [start, *start.parents]:
        if (candidate / "notebooks").is_dir() and (candidate / "README.md").is_file():
            return candidate
    return start.parent if start.name == "notebooks" else start


PROJECT_ROOT = find_project_root()
UPFD_ROOT = PROJECT_ROOT / "data" / "upfd"
RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"
LOGS_DIR = RESULTS_DIR / "logs"
CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"
for _d in (METRICS_DIR, FIGURES_DIR, LOGS_DIR, CHECKPOINTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

print("Working directory:", Path.cwd())
print("Project root     :", PROJECT_ROOT)
print("UPFD data root   :", UPFD_ROOT, "(found)" if UPFD_ROOT.is_dir() else "(NOT FOUND)")
if not UPFD_ROOT.is_dir():
    raise FileNotFoundError(
        f"Could not find '{UPFD_ROOT}'. Start Jupyter from inside the project folder "
        "(e.g. `cd FakeNews-detection-GNN` then `jupyter lab`) so that data/upfd/ can be located."
    )


# Convert NumPy values and paths so that they can be written to JSON.
def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


# Helpers that create the target folder and then save the file.
def save_figure(fig, path, dpi=150):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def save_dataframe(df, path, index=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    return path


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default), encoding="utf-8")
    return path


REQUIRED_FILES = [
    "A.txt", "graph_labels.npy", "node_graph_id.npy",
    "train_idx.npy", "val_idx.npy", "test_idx.npy",
]
FEATURE_FILE = f"new_{FEATURE}_feature.npz"


# Find the folder holding the dataset files, either <name>/raw or <name>.
def detect_layout(name, root=UPFD_ROOT):
    base = Path(root) / name
    required = REQUIRED_FILES + [FEATURE_FILE]
    problems = []
    for label, directory in (("raw", base / "raw"), ("flat", base)):
        if not directory.is_dir():
            problems.append(f"  - {directory}: directory does not exist")
            continue
        missing = [f for f in required if not (directory / f).is_file()]
        if not missing:
            return label, directory
        problems.append(f"  - {directory}: missing {missing}")
    raise FileNotFoundError(
        f"Could not find a complete UPFD '{name}' dataset under {base}.\n"
        + "\n".join(problems)
        + "\nThe files must be either directly inside that folder or inside its 'raw/' sub-folder."
    )


# Load the node feature matrix as a sparse float32 matrix.
def load_feature_matrix(path):
    path = Path(path)
    try:
        X = sp.load_npz(path)
        kind = f"scipy.sparse ({X.format})"
    except Exception as sparse_error:
        try:
            with np.load(path, allow_pickle=False) as archive:
                arrays = {k: archive[k] for k in archive.files}
        except Exception as dense_error:
            raise ValueError(
                f"{path.name} is neither a SciPy sparse archive ({sparse_error!r}) "
                f"nor a readable NumPy archive ({dense_error!r})."
            )
        candidates = [k for k, v in arrays.items() if v.ndim == 2]
        if len(candidates) != 1:
            raise ValueError(
                f"{path.name} is not a SciPy sparse archive ({sparse_error!r}) and contains "
                f"{len(candidates)} two-dimensional arrays (keys: {list(arrays)}), so the feature "
                "matrix cannot be identified unambiguously."
            )
        X = sp.csr_matrix(arrays[candidates[0]])
        kind = f"dense array stored under key '{candidates[0]}'"
    return X.tocsr().astype(np.float32, copy=False), kind


# Read one UPFD dataset and validate it. All nodes are stored in one long matrix;
# node_slice and edge_slice give the rows that belong to each graph.
def load_upfd_raw(name, load_features=True):
    layout, d = detect_layout(name)

    # Node to graph mapping, labels and per-graph node ranges.
    node_graph_id = np.load(d / "node_graph_id.npy", allow_pickle=False).reshape(-1).astype(np.int64)
    n_nodes = int(node_graph_id.shape[0])

    y_raw = np.load(d / "graph_labels.npy", allow_pickle=False).reshape(-1)
    classes_raw, y = np.unique(y_raw, return_inverse=True)
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    n_graphs = int(y.shape[0])

    if node_graph_id.min() < 0 or node_graph_id.max() >= n_graphs:
        raise ValueError(
            f"[{name}] node_graph_id.npy contains ids in [{node_graph_id.min()}, {node_graph_id.max()}] "
            f"but graph_labels.npy describes {n_graphs} graphs (valid ids: 0..{n_graphs - 1})."
        )
    if np.any(np.diff(node_graph_id) < 0):
        raise ValueError(
            f"[{name}] node_graph_id.npy is not sorted, i.e. the nodes of one graph are not contiguous. "
            "The PyG-style slicing used here (and by PyG itself) would be wrong."
        )
    node_counts = np.bincount(node_graph_id, minlength=n_graphs)
    if np.any(node_counts == 0):
        raise ValueError(f"[{name}] {(node_counts == 0).sum()} graph id(s) own zero nodes.")
    node_slice = np.concatenate([[0], np.cumsum(node_counts)]).astype(np.int64)


    # Edge list: validate, remove duplicates and group by graph.
    edges_raw = pd.read_csv(d / "A.txt", header=None, sep=",", skipinitialspace=True).to_numpy()
    if edges_raw.ndim != 2 or edges_raw.shape[1] != 2:
        raise ValueError(f"[{name}] A.txt should have exactly two comma-separated columns, got shape {edges_raw.shape}.")
    edges_raw = edges_raw.astype(np.int64)
    if edges_raw.min() < 0 or edges_raw.max() >= n_nodes:
        raise ValueError(
            f"[{name}] A.txt node ids range over [{edges_raw.min()}, {edges_raw.max()}] but there are "
            f"{n_nodes} nodes (valid: 0..{n_nodes - 1}). If the ids start at 1 the file is 1-indexed."
        )
    n_edges_raw = int(len(edges_raw))
    n_self_loops = int((edges_raw[:, 0] == edges_raw[:, 1]).sum())
    edges = np.unique(edges_raw, axis=0)
    n_duplicates = n_edges_raw - int(len(edges))
    key = edges[:, 0] * n_nodes + edges[:, 1]
    rev_key = edges[:, 1] * n_nodes + edges[:, 0]
    n_reciprocal = int((np.isin(rev_key, key) & (edges[:, 0] != edges[:, 1])).sum())

    src_gid = node_graph_id[edges[:, 0]]
    dst_gid = node_graph_id[edges[:, 1]]
    n_cross = int((src_gid != dst_gid).sum())
    if n_cross:
        raise ValueError(f"[{name}] {n_cross} edge(s) connect nodes that belong to different graphs.")
    if np.any(np.diff(src_gid) < 0):
        raise ValueError(f"[{name}] edges are not grouped by graph after sorting; slicing would be wrong.")
    edge_counts = np.bincount(src_gid, minlength=n_graphs)
    edge_slice = np.concatenate([[0], np.cumsum(edge_counts)]).astype(np.int64)


    # BERT features.
    X, feature_kind = (None, "not loaded")
    if load_features:
        X, feature_kind = load_feature_matrix(d / FEATURE_FILE)
        if X.shape[0] != n_nodes:
            raise ValueError(
                f"[{name}] {FEATURE_FILE} has {X.shape[0]} rows but node_graph_id.npy has {n_nodes} nodes."
            )


    # Official train, validation and test indices.
    splits = {}
    for s in SPLITS:
        idx = np.load(d / f"{s}_idx.npy", allow_pickle=False).reshape(-1).astype(np.int64)
        if len(idx) and (idx.min() < 0 or idx.max() >= n_graphs):
            raise ValueError(f"[{name}] {s}_idx.npy has indices outside 0..{n_graphs - 1}.")
        splits[s] = idx

    return {
        "name": name, "layout": layout, "dir": d,
        "X": X, "feature_kind": feature_kind,
        "edges": edges, "edge_slice": edge_slice, "node_slice": node_slice,
        "node_graph_id": node_graph_id, "y": y, "y_raw": y_raw, "classes_raw": classes_raw,
        "splits": splits, "n_nodes": n_nodes, "n_graphs": n_graphs,
        "audit": {
            "edges_in_A_txt": n_edges_raw, "unique_edges_after_coalesce": int(len(edges)),
            "duplicate_edges": n_duplicates, "self_loops": n_self_loops,
            "reciprocal_directed_edges": n_reciprocal, "cross_graph_edges": n_cross,
            "min_node_id": int(edges_raw.min()), "max_node_id": int(edges_raw.max()),
        },
    }


# Number of nodes in graph i.
def graph_num_nodes(raw, i):
    return int(raw["node_slice"][i + 1] - raw["node_slice"][i])


# Edges of graph i, renumbered so that the root is node 0.
def local_edges(raw, i):
    return raw["edges"][raw["edge_slice"][i]:raw["edge_slice"][i + 1]] - raw["node_slice"][i]


# BERT vectors of the root (news) nodes of the given graphs.
def root_features(raw, graph_indices):
    rows = raw["node_slice"][np.asarray(graph_indices, dtype=np.int64)]
    return raw["X"][rows].toarray()


# Check the official splits for duplicates, invalid indices and overlap.
def check_splits(raw):
    G = raw["n_graphs"]
    sets = {s: set(raw["splits"][s].tolist()) for s in SPLITS}
    rows = []
    for s in SPLITS:
        idx = raw["splits"][s]
        rows.append({
            "split": s, "n_indices": len(idx), "n_unique": len(sets[s]),
            "duplicates": len(idx) - len(sets[s]),
            "min_index": int(idx.min()) if len(idx) else None,
            "max_index": int(idx.max()) if len(idx) else None,
            "out_of_range": int(((idx < 0) | (idx >= G)).sum()),
        })
    overlaps = {f"{a} & {b}": len(sets[a] & sets[b]) for a, b in [("train", "val"), ("train", "test"), ("val", "test")]}
    union = set().union(*sets.values())
    coverage = {"graphs_in_union": len(union), "total_graphs": G, "graphs_in_no_split": G - len(union)}
    return pd.DataFrame(rows), overlaps, coverage


# Record, for each graph, whether node 0 is the only source or the only sink.
def root_convention_check(raw):
    rows = []
    for i in range(raw["n_graphs"]):
        n = graph_num_nodes(raw, i)
        e = local_edges(raw, i)
        indeg = np.bincount(e[:, 1], minlength=n)
        outdeg = np.bincount(e[:, 0], minlength=n)
        rows.append((i, n, len(e), int(indeg[0]), int(outdeg[0]), int((indeg == 0).sum()), int((outdeg == 0).sum())))
    df = pd.DataFrame(rows, columns=["graph_idx", "num_nodes", "num_edges", "node0_in_degree",
                                     "node0_out_degree", "n_sources", "n_sinks"])
    df["node0_unique_source"] = (df["node0_in_degree"] == 0) & (df["n_sources"] == 1)
    df["node0_unique_sink"] = (df["node0_out_degree"] == 0) & (df["n_sinks"] == 1)
    return df


# Summarise the root check and stop if node 0 is not the root.
def summarize_root_check(df, name):
    frac_source = float(df["node0_unique_source"].mean())
    frac_sink = float(df["node0_unique_sink"].mean())
    direction = ("edges point from node 0 (root) outwards: parent -> child"
                 if frac_source >= frac_sink else
                 "edges point towards node 0 (root): child -> parent")
    summary = {
        "dataset": name, "graphs": len(df),
        "node0_is_unique_source": f"{int(df['node0_unique_source'].sum())} / {len(df)}",
        "node0_is_unique_sink": f"{int(df['node0_unique_sink'].sum())} / {len(df)}",
        "edge_direction_inferred": direction,
        "graphs_with_edges_eq_nodes_minus_1": f"{int((df['num_edges'] == df['num_nodes'] - 1).sum())} / {len(df)}",
    }
    if max(frac_source, frac_sink) < 0.95:
        raise RuntimeError(
            f"[{name}] local node 0 is the unique root in only {max(frac_source, frac_sink):.1%} of graphs. "
            "The assumption 'graph.x[0] is the news node' is NOT supported by this data; stop and investigate."
        )
    if max(frac_source, frac_sink) < 1.0:
        bad = df.loc[~(df["node0_unique_source"] if frac_source >= frac_sink else df["node0_unique_sink"]), "graph_idx"]
        warnings.warn(f"[{name}] root convention violated in graphs {bad.head(10).tolist()} (showing up to 10).")
    return summary


# Propagation statistics of one graph. Depth is measured from the root (node 0).
def compute_graph_statistics(n_nodes, edges):
    n_edges = int(len(edges))
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    G.add_edges_from((int(a), int(b)) for a, b in edges if a != b)
    degree_map = dict(G.degree())
    degrees = np.array([degree_map[v] for v in range(n_nodes)], dtype=np.int64)
    depth_map = nx.single_source_shortest_path_length(G, 0)
    non_root_depths = np.array([dep for v, dep in depth_map.items() if v != 0], dtype=np.float64)
    connected = len(depth_map) == n_nodes
    leaf_count = int((degrees[1:] == 1).sum()) if n_nodes > 1 else 0
    return {
        "num_nodes": int(n_nodes),
        "num_edges": n_edges,
        "max_depth": int(max(depth_map.values())),
        "avg_depth": float(non_root_depths.mean()) if len(non_root_depths) else 0.0,
        "root_degree": int(degrees[0]),
        "max_degree": int(degrees.max()),
        "avg_degree": float(degrees.mean()),
        "degree_std": float(degrees.std()),
        "leaf_count": leaf_count,
        "leaf_proportion": leaf_count / n_nodes,
        "is_connected": bool(connected),
        "is_tree": bool(connected and G.number_of_edges() == n_nodes - 1),
    }


# Statistics of every graph in a dataset, with its split and label.
def compute_all_graph_statistics(raw, desc=None):
    rows = []
    for i in tqdm(range(raw["n_graphs"]), desc=desc or raw["name"], leave=False):
        row = compute_graph_statistics(graph_num_nodes(raw, i), local_edges(raw, i))
        row["graph_idx"] = i
        rows.append(row)
    df = pd.DataFrame(rows)
    gi = df["graph_idx"].to_numpy()
    df["label"] = raw["y"][gi]
    df["label_raw"] = raw["y_raw"][gi]
    split = np.full(raw["n_graphs"], "unassigned", dtype=object)
    for s_name, idx in raw["splits"].items():
        split[idx] = s_name
    df["split"] = split[gi]
    first = ["graph_idx", "split", "label", "label_raw"]
    return df[first + [c for c in df.columns if c not in first]]
