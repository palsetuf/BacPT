"""Load a BacPT-small or BacPT-large training checkpoint for inference."""

from pathlib import Path

import torch

from src.models import bacpt_model


def load_bacpt(variant, checkpoint_path, device="cpu"):
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
    if "model_state_dict" not in checkpoint:
        raise ValueError(f"No model_state_dict in {checkpoint_path}")
    model = bacpt_model(variant)
    model_keys = set(model.state_dict())
    state = {}
    for key, tensor in checkpoint["model_state_dict"].items():
        while key.startswith("module."):
            key = key[len("module."):]
        if key in state:
            raise ValueError(f"Duplicate model weight after prefix removal: {key}")
        # Newer Transformers registers this fixed position-index buffer as
        # non-persistent. All learned weights must still load strictly.
        if key == "embeddings.position_ids" and key not in model_keys:
            continue
        state[key] = tensor
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    return model, {"epoch": checkpoint.get("epoch"), "variant": variant}
