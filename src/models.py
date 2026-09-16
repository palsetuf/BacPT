"""Minimal BacPT architectures matching the released training checkpoints."""

from torch import nn
from transformers import RobertaConfig, RobertaModel, RoFormerConfig, RoFormerModel


class InputMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, vectors):
        return self.dense(vectors)


class ReconstructionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense1 = nn.Linear(config.hidden_size, config.pca_dim if config.pca else config.hidden_size)

    def forward(self, vectors):
        return self.dense1(vectors)


class BacPTSmall(RobertaModel):
    def __init__(self, config):
        super().__init__(config, add_pooling_layer=False)
        self.embeddings.word_embeddings = None
        self.mlp = InputMLP(config)
        self.lm_head = ReconstructionHead(config)
        self.post_init()

    def forward(self, inputs_embeds, attention_mask):
        outputs = super().forward(
            input_ids=None,
            inputs_embeds=self.mlp(inputs_embeds),
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=False,
            return_dict=True,
        )
        return self.lm_head(outputs.last_hidden_state), outputs.hidden_states


class BacPTLarge(RoFormerModel):
    def __init__(self, config):
        super().__init__(config)
        self.pooler = None
        self.embeddings.word_embeddings = None
        self.mlp = InputMLP(config)
        self.lm_head = ReconstructionHead(config)
        self.post_init()

    def forward(self, inputs_embeds, attention_mask):
        outputs = super().forward(
            input_ids=None,
            inputs_embeds=self.mlp(inputs_embeds),
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=False,
            return_dict=True,
        )
        return self.lm_head(outputs.last_hidden_state), outputs.hidden_states


def bacpt_config(variant):
    common = dict(
        hidden_size=480,
        max_position_embeddings=5000,
        intermediate_size=3072,
        type_vocab_size=1,
        hidden_dropout_prob=0.1,
        hidden_act="gelu",
        pca=True,
        pca_dim=480,
        tie_word_embeddings=False,
        attn_implementation="eager",
    )
    if variant == "small":
        return RobertaConfig(
            **common,
            num_hidden_layers=10,
            num_attention_heads=5,
            position_embedding_type="relative_key_query",
        )
    if variant == "large":
        return RoFormerConfig(**common, num_hidden_layers=19, num_attention_heads=10)
    raise ValueError(f"Unknown BacPT variant: {variant}")


def bacpt_model(variant):
    config = bacpt_config(variant)
    return BacPTSmall(config) if variant == "small" else BacPTLarge(config)
