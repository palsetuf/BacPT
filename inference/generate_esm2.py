"""Generate BacPT's raw ESM2 protein embeddings from one ordered FASTA file.

Uses the original FAIR ESM2 model and averages layer-12 amino-acid vectors.
The output is a pickle-free NPZ with protein IDs and float32 embeddings.
"""

import argparse
from contextlib import nullcontext
from pathlib import Path

import numpy as np

from inference.prepare_input import read_fasta


ESM2_LAYER = 12
MAX_PROTEIN_LENGTH = 2250


def embed_proteins(proteins, model, alphabet, device):
    """Average layer-12 amino-acid representations for ordered proteins."""
    import torch

    model = model.to(device).eval()
    batch_converter = alphabet.get_batch_converter()
    protein_ids = []
    vectors = []
    with torch.no_grad():
        for protein_id, sequence in proteins:
            # The development script replaced the rare J residue with one mask token.
            sequence = sequence.replace("J", "<mask>")
            _, _, tokens = batch_converter([(protein_id, sequence)])
            tokens = tokens.to(device)
            representation = model(tokens, repr_layers=[ESM2_LAYER],
                                   return_contacts=False)["representations"][ESM2_LAYER]
            # One beginning and one end token surround the amino-acid positions.
            mean_vector = representation[0, 1:-1].mean(0)
            protein_ids.append(protein_id)
            vectors.append(mean_vector.cpu().numpy())
    return protein_ids, np.stack(vectors).astype(np.float32, copy=False)


def generate_embeddings(fasta, device=None, weights=None):
    """Return retained protein IDs, raw ESM2 embeddings, and omitted records."""
    try:
        import esm
        import torch
    except ImportError as exc:
        raise RuntimeError("FASTA embedding needs PyTorch and fair-esm==2.0.0") from exc

    proteins, omitted = read_fasta(fasta, max_length=MAX_PROTEIN_LENGTH)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if weights is None:
        model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
    else:
        # FAIR's local checkpoint contains argparse.Namespace in its metadata.
        # Newer PyTorch versions require that class to be explicitly allowed.
        allowed = (torch.serialization.safe_globals([argparse.Namespace])
                   if hasattr(torch.serialization, "safe_globals") else nullcontext())
        with allowed:
            model, alphabet = esm.pretrained.load_model_and_alphabet(str(weights))
    protein_ids, embeddings = embed_proteins(proteins, model, alphabet, device)
    if embeddings.shape != (len(protein_ids), 480):
        raise ValueError(f"Unexpected ESM2 output shape: {embeddings.shape}")
    return protein_ids, embeddings, omitted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta", required=True, type=Path,
                        help="one protein FASTA file, already in genomic order")
    parser.add_argument("--output", required=True, type=Path, help="raw ESM2 NPZ")
    parser.add_argument("--device", choices=("cpu", "cuda"),
                        help="defaults to CUDA if available, otherwise CPU")
    parser.add_argument("--esm-weights", type=Path,
                        help="optional local FAIR ESM2 checkpoint; its contact-regression file must be beside it")
    args = parser.parse_args()
    protein_ids, embeddings, omitted = generate_embeddings(args.fasta, args.device, args.esm_weights)
    np.savez_compressed(args.output, protein_ids=np.asarray(protein_ids), embeddings=embeddings)
    noun = "protein" if len(omitted) == 1 else "proteins"
    print(f"Saved {len(protein_ids)} ESM2 embeddings; omitted {len(omitted)} long {noun}")


if __name__ == "__main__":
    main()
