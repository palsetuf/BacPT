"""Compute BacPT's perturbation-based gene-interaction (Jacobian) matrix.

For each gene i, its embedding is replaced with a zero vector while every
other gene keeps its original embedding and the full genome stays as model
context. The influence of gene i on gene j is the L2 norm of the resulting
change in BacPT's predicted embedding for gene j (Methods: "Calculation of
Jacobian matrices"). This module and its post-processing were ported and
numerically validated against the original research notebook's cached
output, not re-derived from the manuscript's prose alone; see the project
plan for the validation details.

Unlike inference/bacpt.py, this uses a 2-D [batch, seq] attention mask, not
inference.bacpt.prepare_bacpt_context's 3-D mask -- the two were confirmed
to give different results, and only the 2-D mask reproduces the original
Jacobian notebook's cached output.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from inference.load_bacpt import load_bacpt_release
from inference.prepare_input import prepare_input

MAX_PROTEINS = 5000


def prepare_jacobian_context(normalized, device="cpu"):
    """Zero-pad to 5,000 positions and build a standard 2-D attention mask."""
    if not isinstance(normalized, np.ndarray):
        raise TypeError("BacPT input must be a NumPy array")
    if normalized.ndim != 2 or normalized.shape[1] != 480 or normalized.dtype != np.float32:
        raise ValueError("BacPT input must be float32 [proteins, 480]")
    count = min(normalized.shape[0], MAX_PROTEINS)
    inputs = torch.from_numpy(normalized[:count]).unsqueeze(0).to(device)
    inputs = F.pad(inputs, (0, 0, 0, MAX_PROTEINS - count))
    mask = torch.zeros((1, MAX_PROTEINS), dtype=torch.int, device=device)
    mask[:, :count] = 1
    return inputs, mask, count


def _run_batch(model, inputs_batch, mask_batch, count, device):
    with torch.inference_mode(), torch.amp.autocast(
        device_type="cuda" if device.type == "cuda" else "cpu",
        dtype=torch.float16,
        enabled=device.type == "cuda",
    ):
        reconstruction, _ = model(inputs_embeds=inputs_batch, attention_mask=mask_batch)
    return reconstruction[:, :count].float()


def compute_raw_jacobian(model, normalized, positions=None, batch_size=8, device=None):
    """Return the raw [len(positions), len(positions)] perturbation matrix.

    The same gene index set is used for both the perturbed genes (rows) and
    the genes whose predictions are measured (columns): the post-processing
    step's symmetrization (M @ M.T) requires a square matrix over one
    consistent gene set, matching how the reference notebook computed it.
    """
    device = device or next(model.parameters()).device
    inputs, mask, count = prepare_jacobian_context(normalized, device)
    if positions is None:
        positions = np.arange(count)
    else:
        positions = np.asarray(positions, dtype=np.int64)
        if positions.ndim != 1 or len(positions) == 0 or np.any(positions < 0) or np.any(positions >= count):
            raise ValueError("positions must be within the retained genome")

    model.eval()
    baseline = _run_batch(model, inputs, mask, count, device)[0]  # [count, 480]

    rows = []
    for start in range(0, len(positions), batch_size):
        batch_positions = positions[start:start + batch_size]
        batch_inputs = inputs.repeat(len(batch_positions), 1, 1).clone()
        for row, pos in enumerate(batch_positions):
            batch_inputs[row, pos, :] = 0.0
        batch_mask = mask.repeat(len(batch_positions), 1)
        predicted = _run_batch(model, batch_inputs, batch_mask, count, device)  # [B, count, 480]
        delta = baseline.unsqueeze(0) - predicted  # [B, count, 480]
        rows.append(torch.norm(delta, p=2, dim=2).cpu().numpy())  # [B, count]

    raw = np.concatenate(rows, axis=0)  # [len(positions), count]
    return raw[:, positions]  # restrict columns to the same gene set


def postprocess_jacobian(M):
    """Port of the reference notebook's transform_ppi_matrix/symmetrize/apc,
    applied in the exact order it was actually invoked there:
    mean_centered=True, sqr=True, sym=True, apc_=True, clip=True, filld=True.
    """
    M = np.asarray(M, dtype=np.float64).copy()

    # Sequential mean-centering: subtract column means, then row means of
    # the result (not a single double-centering formula with a grand-mean
    # term -- this is the actual reference implementation).
    M -= M.mean(axis=0, keepdims=True)
    M -= M.mean(axis=1, keepdims=True)

    # "sqr" in the reference is sqrt(square(x)), i.e. absolute value.
    M = np.abs(M)

    # Symmetrize via real matrix multiplication (not elementwise, not (M+M.T)/2).
    M = M @ M.T

    # Average product correction.
    total = np.sum(M)
    row_sums = np.sum(M, axis=1, keepdims=True)
    col_sums = np.sum(M, axis=0, keepdims=True)
    M = M - (row_sums @ col_sums) / total

    # Clip after APC, then zero the diagonal last.
    M = np.clip(M, 0, None)
    np.fill_diagonal(M, 0)
    return M


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--esm2", type=Path, required=True, help="ESM2 NPZ with IDs and raw embeddings")
    parser.add_argument("--scaler", type=Path, required=True)
    parser.add_argument("--variant", choices=("small", "large"), required=True)
    parser.add_argument("--start-gene", type=int, default=0,
                         help="first gene index to perturb/measure (0-based, genomic order)")
    parser.add_argument("--num-genes", type=int, default=50,
                         help="number of genes to perturb/measure, starting at --start-gene; "
                              "the full genome is always used as model context regardless")
    parser.add_argument("--batch-size", type=int, default=2,
                         help="perturbations computed per forward pass; BacPT-large at full "
                              "5,000-position context uses roughly 2.5-3GB of GPU memory per unit "
                              "of batch size, so raise this only if you have GPU memory to spare")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                         help="a GPU is strongly recommended: a full 5,000-length forward pass "
                              "can exceed 8GB of CPU memory")
    parser.add_argument("--output", type=Path, required=True, help="output NPZ")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")

    ids, normalized, omitted = prepare_input(args.fasta, args.esm2, args.scaler)
    model, _ = load_bacpt_release(args.variant, device=device)

    positions = np.arange(args.start_gene, args.start_gene + args.num_genes)
    raw = compute_raw_jacobian(model, normalized, positions=positions,
                                batch_size=args.batch_size, device=device)
    processed = postprocess_jacobian(raw)

    np.savez_compressed(
        args.output,
        raw=raw,
        processed=processed,
        positions=positions,
        protein_ids=np.asarray([ids[p] for p in positions]),
    )
    print(f"Saved {len(positions)}x{len(positions)} Jacobian matrix for genes "
          f"{args.start_gene}..{args.start_gene + args.num_genes - 1} to {args.output}")


if __name__ == "__main__":
    main()
