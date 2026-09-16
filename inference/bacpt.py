"""Run BacPT with the padding, mask shape, and precision of the saved workflow."""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from inference.load_bacpt import load_bacpt
from inference.prepare_input import prepare_input


MAX_PROTEINS = 5000


@dataclass
class BacPTOutput:
    protein_count: int
    reconstruction: np.ndarray
    last_hidden_state: np.ndarray
    hidden_states: np.ndarray | None = None


def prepare_bacpt_context(normalized, device="cpu"):
    """Zero-pad to 5,000 and make the notebook's [batch, query, 1] mask."""
    if not isinstance(normalized, np.ndarray):
        raise TypeError("BacPT input must be a NumPy array")
    if normalized.ndim != 2 or normalized.shape[1] != 480 or normalized.dtype != np.float32:
        raise ValueError("BacPT input must be float32 [proteins, 480]")
    if normalized.shape[0] == 0 or not np.isfinite(normalized).all():
        raise ValueError("BacPT input must contain finite protein vectors")
    protein_count = min(normalized.shape[0], MAX_PROTEINS)
    inputs = torch.from_numpy(normalized[:protein_count]).unsqueeze(0).to(device)
    inputs = torch.nn.functional.pad(inputs, (0, 0, 0, MAX_PROTEINS - protein_count))
    mask = torch.zeros((1, MAX_PROTEINS, 1), dtype=torch.int, device=device)
    mask[:, :protein_count, :] = 1
    return inputs, mask, protein_count


def run_bacpt(model, normalized, positions=None, all_layers=False):
    """Return representations for retained positions with historical inference settings.

    CUDA runs use float16 autocasting as in the development notebook. CPU runs
    keep the same padding and mask shape, but use float32 arithmetic.
    """
    device = next(model.parameters()).device
    inputs, mask, protein_count = prepare_bacpt_context(normalized, device)
    if positions is None:
        selection = slice(0, protein_count)
    else:
        positions = np.asarray(positions, dtype=np.int64)
        if positions.ndim != 1 or len(positions) == 0 or np.any(positions < 0) or np.any(positions >= protein_count):
            raise ValueError("Selected positions must be within the retained genome")
        selection = positions.tolist()
    model.eval()
    with torch.inference_mode(), torch.amp.autocast(
        device_type="cuda" if device.type == "cuda" else "cpu",
        dtype=torch.float16,
        enabled=device.type == "cuda",
    ):
        reconstruction, layers = model(inputs_embeds=inputs, attention_mask=mask)
    reconstruction = reconstruction[0, selection].float().cpu().numpy()
    last_hidden = layers[-1][0, selection].float().cpu().numpy()
    hidden_states = None
    if all_layers:
        hidden_states = np.stack([layer[0, selection].float().cpu().numpy() for layer in layers])
    return BacPTOutput(protein_count, reconstruction, last_hidden, hidden_states)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--esm2", type=Path, required=True, help="ESM2 NPZ with IDs and raw embeddings")
    parser.add_argument("--scaler", type=Path, required=True)
    parser.add_argument("--variant", choices=("small", "large"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--all-layers", action="store_true", help="Also save every BacPT layer")
    args = parser.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    ids, normalized, omitted = prepare_input(args.fasta, args.esm2, args.scaler)
    model, _ = load_bacpt(args.variant, args.checkpoint, device=device)
    output = run_bacpt(model, normalized, all_layers=args.all_layers)
    data = dict(
        protein_ids=np.asarray(ids[:output.protein_count]),
        reconstruction=output.reconstruction,
        last_hidden_state=output.last_hidden_state,
        omitted_protein_ids=np.asarray([protein_id for protein_id, _ in omitted], dtype=str),
        truncated_protein_ids=np.asarray(ids[output.protein_count:], dtype=str),
    )
    if args.all_layers:
        data["hidden_states"] = output.hidden_states
    np.savez_compressed(args.output, **data)
    noun = "protein" if len(omitted) == 1 else "proteins"
    print(
        f"Saved BacPT-{args.variant} representations for {output.protein_count} proteins; "
        f"omitted {len(omitted)} long {noun} and truncated {len(ids) - output.protein_count} beyond position 5000"
    )


if __name__ == "__main__":
    main()
