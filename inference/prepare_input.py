"""Align ordered FASTA proteins to precomputed ESM2 vectors and scale them.

This prepares input for BacPT from a trusted `[proteins, 480]` ESM2 archive.
FASTA-to-ESM2 generation and BacPT model loading are separate release steps.
"""

import argparse
from pathlib import Path

import numpy as np


def read_fasta(path, max_length=2250):
    """Return retained (ID, sequence) records and omitted (ID, length) records."""
    retained = []
    omitted = []
    protein_id = None
    chunks = []
    seen = set()

    def add_record():
        if protein_id is None:
            return
        sequence = "".join(chunks)
        if not sequence:
            raise ValueError(f"Empty sequence for {protein_id}")
        if protein_id in seen:
            raise ValueError(f"Duplicate protein ID: {protein_id}")
        seen.add(protein_id)
        if len(sequence) <= max_length:
            retained.append((protein_id, sequence))
        else:
            omitted.append((protein_id, len(sequence)))

    with Path(path).open() as stream:
        for line in stream:
            if line.startswith(">"):
                add_record()
                protein_id = line[1:].split()[0]
                if not protein_id:
                    raise ValueError("FASTA header has no protein ID")
                chunks = []
            elif line.strip():
                if protein_id is None:
                    raise ValueError("FASTA sequence appeared before its header")
                chunks.append(line.strip())
    add_record()
    if not retained:
        raise ValueError("No proteins remain after length filtering")
    return retained, omitted


def prepare_input(fasta, esm2_archive, scaler_archive, max_length=2250):
    """Return protein IDs, normalized float32 BacPT input, and omitted IDs."""
    proteins, omitted = read_fasta(fasta, max_length=max_length)
    protein_ids = [protein_id for protein_id, _ in proteins]
    with np.load(esm2_archive, allow_pickle=False) as data:
        if "protein_ids" not in data or "embeddings" not in data:
            raise ValueError("ESM2 archive needs protein_ids and embeddings arrays")
        archive_ids = data["protein_ids"].tolist()
        embeddings = data["embeddings"]
    if archive_ids != protein_ids:
        raise ValueError("ESM2 protein IDs differ from the filtered FASTA order")
    if embeddings.shape != (len(protein_ids), 480) or embeddings.dtype != np.float32:
        raise ValueError("ESM2 embeddings must be float32 [proteins, 480]")

    with np.load(scaler_archive, allow_pickle=False) as data:
        mean = data["mean"]
        scale = data["scale"]
    if mean.shape != (480,) or scale.shape != (480,) or np.any(scale <= 0):
        raise ValueError("Scaler archive must contain 480 means and positive scales")
    if not np.isfinite(embeddings).all() or not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise ValueError("Input arrays contain non-finite values")
    normalized = ((embeddings.astype(np.float64) - mean) / scale).astype(np.float32)
    return protein_ids, normalized, omitted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--esm2", type=Path, required=True, help="precomputed ESM2 NPZ with protein_ids")
    parser.add_argument("--scaler", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=2250)
    args = parser.parse_args()
    protein_ids, embeddings, omitted = prepare_input(args.fasta, args.esm2, args.scaler, args.max_length)
    np.savez_compressed(args.output, protein_ids=np.asarray(protein_ids), embeddings=embeddings)
    noun = "protein" if len(omitted) == 1 else "proteins"
    print(f"Prepared {len(protein_ids)} BacPT inputs; omitted {len(omitted)} long {noun}")


if __name__ == "__main__":
    main()
