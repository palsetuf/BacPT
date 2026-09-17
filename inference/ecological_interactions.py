"""Predict pairwise ecological interaction outcomes from BacPT genome embeddings.

For a pair of species (A, B) in a given environment, builds the feature
vector [eA, eB, eA*eB] from genome-mean embeddings (Methods: "Ecological
interaction prediction pipeline") and trains a linear probe to predict the
interaction outcome class. Two evaluation strategies are supported, matching
the manuscript: a random train/test split over pairs, and a species-disjoint
split where no species in the validation set appears in training.

Ported from the original research notebooks (linear_probe_randomsplit.ipynb,
linear_probe_tough_split.ipynb -- identical pipelines differing only in
which split strategy they use). One simplification versus the originals:
the per-environment feature vector there also concatenated a one-hot
environment indicator, but every training call already filters to a single
environment first, making that indicator a constant column across the
whole fit -- confirmed empirically, not a modeling choice, so it's dropped
here rather than ported as dead weight.
"""

import argparse
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from inference.context import prepare_2d_context
from inference.generate_esm2 import generate_embeddings
from inference.load_bacpt import load_bacpt_release


def embed_genome_all_layers(model, normalized, device=None):
    """Mean-pool every hidden layer's output over the genome's real positions.

    Returns a [num_layers, 480] array (num_layers = num_hidden_layers + 1,
    matching the embedding-layer-plus-transformer-layers convention of
    output_hidden_states).
    """
    device = device or next(model.parameters()).device
    inputs, mask, count = prepare_2d_context(normalized, device)
    model.eval()
    with torch.inference_mode():
        _, hidden_states = model(inputs_embeds=inputs, attention_mask=mask)
    return np.stack([h[0, :count].float().mean(0).cpu().numpy() for h in hidden_states])


def embed_all_species(fasta_dir, scaler_path, device=None):
    """Generate ESM2 and BacPT (small + large, all layers) genome-mean embeddings
    for every FASTA file in `fasta_dir` (species ID = filename stem without extension).

    Returns {species_id: {"esm": [480], "small": [11, 480], "large": [20, 480]}}.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    mean, scale = np.load(scaler_path)["mean"], np.load(scaler_path)["scale"]

    models = {variant: load_bacpt_release(variant, device=device)[0] for variant in ("small", "large")}

    embeddings = {}
    for fasta_path in sorted(Path(fasta_dir).glob("*.fa*")):
        species_id = fasta_path.stem
        _, raw_esm, _ = generate_embeddings(fasta_path, device=str(device))
        normalized = ((raw_esm.astype(np.float64) - mean) / scale).astype(np.float32)
        embeddings[species_id] = {"esm": raw_esm.mean(0)}
        for variant, model in models.items():
            embeddings[species_id][variant] = embed_genome_all_layers(model, normalized, device)
    return embeddings


def build_pair_features(emb_a, emb_b):
    """[eA, eB, eA*eB] -- the manuscript's feature formula, no one-hot term."""
    return np.concatenate([emb_a, emb_b, emb_a * emb_b])


def build_dataset(pairs_df, embeddings, model_key, layer, environment,
                   species_a_col="species_a", species_b_col="species_b",
                   environment_col="environment", label_col="label"):
    """Build (X, y, species_pairs) for one (model, layer, environment) config."""
    df_env = pairs_df[pairs_df[environment_col] == environment]
    X, y, species_pairs = [], [], []
    for _, row in df_env.iterrows():
        a, b = row[species_a_col], row[species_b_col]
        emb_a = embeddings[a]["esm"] if model_key == "esm" else embeddings[a][model_key][layer]
        emb_b = embeddings[b]["esm"] if model_key == "esm" else embeddings[b][model_key][layer]
        X.append(build_pair_features(emb_a, emb_b))
        y.append(row[label_col])
        species_pairs.append((a, b))
    return np.asarray(X, dtype=np.float32), np.asarray(y), species_pairs


def filter_rare_classes(X, y, species_pairs, k):
    counts = Counter(y.tolist())
    keep = {cls for cls, c in counts.items() if c >= k}
    mask = np.isin(y, list(keep))
    species_pairs = [sp for sp, m in zip(species_pairs, mask) if m]
    return X[mask], y[mask], species_pairs


def random_split_cv(X, y, Cs=np.logspace(-3, 3, 10), k=5, max_iter=5000, seed=42):
    """Standard k-fold cross-validated logistic regression (manuscript's "random split")."""
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    grid = GridSearchCV(LogisticRegression(penalty="l2", solver="lbfgs", max_iter=max_iter),
                         {"C": Cs}, scoring="f1_macro", cv=cv, n_jobs=-1, refit=True)
    grid.fit(X, y)
    best_c = grid.best_params_["C"]

    f1s = []
    for train_idx, val_idx in cv.split(X, y):
        m = LogisticRegression(C=best_c, penalty="l2", solver="lbfgs", max_iter=max_iter)
        m.fit(X[train_idx], y[train_idx])
        f1s.append(f1_score(y[val_idx], m.predict(X[val_idx]), average="macro"))
    return {"F1": float(np.mean(f1s))}


def species_disjoint_cv(X, species_pairs, y, Cs=np.logspace(-3, 3, 10),
                         k=3, max_iter=5000, seed=42, max_tries=300):
    """Species-disjoint 80/20 split, repeated k times (manuscript's "species-disjoint split").

    Retries a random species partition up to max_tries times per split until
    every class appears in both the train and validation sides.
    """
    rng = np.random.default_rng(seed)
    all_classes = set(np.unique(y))
    all_species = sorted({s for pair in species_pairs for s in pair})

    f1s = []
    for split_num in range(k):
        for _attempt in range(max_tries):
            species = list(all_species)
            rng.shuffle(species)
            split_idx = int(len(species) * 0.8)
            train_species, val_species = set(species[:split_idx]), set(species[split_idx:])

            train_mask = np.array([a in train_species and b in train_species for a, b in species_pairs])
            val_mask = np.array([a in val_species or b in val_species for a, b in species_pairs])
            train_idx, val_idx = np.where(train_mask)[0], np.where(val_mask)[0]
            if len(train_idx) == 0 or len(val_idx) == 0:
                continue
            if all_classes.issubset(set(y[train_idx])) and all_classes.issubset(set(y[val_idx])):
                break
        else:
            raise RuntimeError(f"Could not find a valid species-disjoint split ({split_num + 1}/{k}) "
                                f"after {max_tries} tries -- likely too few species/classes for this config")

        X_train, y_train = X[train_idx], y[train_idx]
        min_class_count = min(Counter(y_train.tolist()).values())
        inner_cv = min(2, min_class_count) if min_class_count < 3 else 3

        model = LogisticRegression(penalty="l2", solver="lbfgs", max_iter=max_iter)
        if inner_cv >= 2:
            cv_splitter = StratifiedKFold(n_splits=inner_cv, shuffle=True, random_state=seed + split_num)
            grid = GridSearchCV(model, {"C": Cs}, scoring="f1_macro", cv=cv_splitter, n_jobs=-1, refit=True)
            grid.fit(X_train, y_train)
            best_c = grid.best_params_["C"]
        else:
            best_c = Cs[len(Cs) // 2]

        m = LogisticRegression(C=best_c, penalty="l2", solver="lbfgs", max_iter=max_iter)
        m.fit(X_train, y_train)
        f1s.append(f1_score(y[val_idx], m.predict(X[val_idx]), average="macro"))

    return {"F1": float(np.mean(f1s))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta-dir", type=Path, required=True,
                         help="directory of one ordered protein FASTA per species; "
                              "filename stem (without extension) is used as the species ID")
    parser.add_argument("--pairs-csv", type=Path, required=True,
                         help="CSV with columns species_a, species_b, environment, label")
    parser.add_argument("--scaler", type=Path, required=True)
    parser.add_argument("--split", choices=("random", "species-disjoint"), default="random")
    parser.add_argument("--seeds", type=int, action="append", default=None,
                         help="repeatable; defaults to a single seed (42)")
    parser.add_argument("--min-class-count", type=int, default=3)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, required=True, help="output pickle")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")

    seeds = args.seeds or [42]
    pairs_df = pd.read_csv(args.pairs_csv)
    embeddings = embed_all_species(args.fasta_dir, args.scaler, device=device)

    num_layers = {"small": embeddings[next(iter(embeddings))]["small"].shape[0],
                  "large": embeddings[next(iter(embeddings))]["large"].shape[0],
                  "esm": 1}
    cv_fn = random_split_cv if args.split == "random" else species_disjoint_cv

    results = {model_key: {} for model_key in ("esm", "small", "large")}
    for model_key in results:
        for environment in pairs_df["environment"].unique():
            for layer in range(num_layers[model_key]):
                X, y, species_pairs = build_dataset(pairs_df, embeddings, model_key, layer, environment)
                try:
                    X, y, species_pairs = filter_rare_classes(X, y, species_pairs, args.min_class_count)
                    if len(set(y.tolist())) < 2:
                        continue
                    f1s = []
                    for seed in seeds:
                        if args.split == "random":
                            f1s.append(cv_fn(X, y, seed=seed)["F1"])
                        else:
                            f1s.append(cv_fn(X, species_pairs, y, seed=seed)["F1"])
                    results[model_key].setdefault(environment, {})[layer] = float(np.mean(f1s))
                except (ValueError, RuntimeError) as exc:
                    print(f"skipping {model_key}/{environment}/layer={layer}: {exc}")
                    continue

    import pickle
    with open(args.output, "wb") as f:
        pickle.dump(results, f)
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
