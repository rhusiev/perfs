from typing import overload

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel

from ..peft import Peft
from .block import TransformerBlock
from .config import Config


class Transformer(nn.Module):
    """A simple GPT2-compatible decoder-only transformer model."""

    def __init__(self, config: Config) -> None:
        """Initialize the model.

        Args:
            config: The configuration for the model.
        """
        super().__init__()

        self.token_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embeddings = nn.Embedding(config.block_size, config.d_model)
        self.tower = nn.Sequential(
            *(TransformerBlock(config) for _ in range(config.n_layers))
        )
        self.ln_f = nn.LayerNorm(config.d_model)
        self.unembedding = nn.Linear(config.d_model, config.vocab_size, bias=False)

    @classmethod
    def from_pretrained(
        cls, model_name: str, peft: Peft | None = None
    ) -> "Transformer":
        """Load a pretrained model from Hugging Face's Transformers library.
        This closely follows Karpathy's minGPT loader, it was used as a reference.

        Args:
            model_name: The name of the model to load.
                Should be one of "gpt2".
            peft: The parameter efficient fine-tuning module.

        Returns:
            The pretrained model.
        """

        configs = {
            "gpt2": Config(
                n_layers=12,
                n_heads=12,
                d_model=768,
                vocab_size=50257,
                block_size=1024,
                peft=peft,
            )
        }

        if model_name not in configs:
            raise ValueError(f"Unknown model name: {model_name}")

        model = Transformer(configs[model_name])

        own_state_dictionary = model.state_dict()
        hf_state_dictionary = GPT2LMHeadModel.from_pretrained(model_name).state_dict()

        name_map = {
            "wte.weight": "token_embeddings.weight",
            "wpe.weight": "position_embeddings.weight",
            "mlp.c_fc.weight": "feedforward.upprojection.weight",
            "mlp.c_fc.bias": "feedforward.upprojection.bias",
            "mlp.c_proj.weight": "feedforward.downprojection.weight",
            "mlp.c_proj.bias": "feedforward.downprojection.bias",
            "attn.c_proj.weight": "attn.output.weight",
            "attn.c_proj.bias": "attn.output.bias",
            "lm_head.weight": "unembedding.weight",
            ".h.": ".tower.",
            "transformer.": "",
        }

        for k in hf_state_dictionary:
            if k.endswith("attn.masked_bias"):
                continue

            k_own = k
            for k_source, k_target in name_map.items():
                k_own = k_own.replace(k_source, k_target)

            with torch.no_grad():
                if "c_attn.weight" in k:
                    # This implementation does not have a single
                    # c_attn.weight, but rather q, k, and v weights

                    # We split up the weight matrix in 3 parts for q, k, v:
                    q, k_, v = hf_state_dictionary[k].t().chunk(3)
                    own_state_dictionary[
                        k_own.replace("c_attn.weight", "q.weight")
                    ].copy_(q)
                    own_state_dictionary[
                        k_own.replace("c_attn.weight", "k.weight")
                    ].copy_(k_)
                    own_state_dictionary[
                        k_own.replace("c_attn.weight", "v.weight")
                    ].copy_(v)
                elif "c_attn.bias" in k:
                    # Same for the bias
                    q, k_, v = hf_state_dictionary[k].chunk(3)
                    own_state_dictionary[k_own.replace("c_attn.bias", "q.bias")].copy_(
                        q
                    )
                    own_state_dictionary[k_own.replace("c_attn.bias", "k.bias")].copy_(
                        k_
                    )
                    own_state_dictionary[k_own.replace("c_attn.bias", "v.bias")].copy_(
                        v
                    )
                elif any(
                    k.endswith(w)
                    for w in {
                        "attn.c_attn.weight",
                        "attn.c_proj.weight",
                        "mlp.c_fc.weight",
                        "mlp.c_proj.weight",
                    }
                ):
                    # These are the weights that are trasposed in the
                    # original GPT-2 model, so we transpose them here
                    own_state_dictionary[k_own].copy_(hf_state_dictionary[k].t())
                else:
                    own_state_dictionary[k_own].copy_(hf_state_dictionary[k])

        return model

    @overload
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        ...

    @overload
    def forward(
        self, x: torch.Tensor, targets: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ...

    def forward(
        self,
        x: torch.Tensor,
        targets: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Perform the forward pass through the
        transformer model.

        Args:
            x: The input tensor, with shape (batch_size, seq_len).
            targets: The target tensor, with shape (batch_size, seq_len).
            attention_mask: The attention mask tensor, with shape (batch_size, seq_len).

        Returns:
            The logits, with shape (batch_size, seq_len, vocab_size).
            The loss, if targets are provided.
        """
        e = self.token_embeddings(x) + self.position_embeddings(
            torch.arange(x.size(1), device=x.device, dtype=torch.long)
            .to(x.device)
            .unsqueeze(0)
        )
        e = self.tower(e)
        e = self.ln_f(e)
        logits = self.unembedding(e)

        if attention_mask is not None:
            logits = logits.masked_fill(
                attention_mask.unsqueeze(-1) == 0, float("-inf")
            )

        loss = None
        if targets is not None:
            if attention_mask is not None:
                targets = targets.masked_fill(attention_mask == 0, -100)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100
            )

        return logits, loss

    def generate(
        self,
        prompt: list[int] | torch.Tensor,
        max_len: int,
        eos_token: int | None = None,
    ) -> torch.Tensor:
        """Generate a sequence of tokens from a prompt.

        Args:
            prompt: The prompt to generate from.
            max_len: The maximum length of the generated sequence.
            tokenizer: The tokenizer to use.
            eos_token: The token that marks the end of a sequence.

        Returns:
            The generated sequence of tokens.
        """
        inp = prompt.clone().detach().unsqueeze(0)
        self.eval()
        with torch.no_grad():
            for _ in range(max_len):
                logits = self.forward(inp)[0]
                last_level_logits = logits[0, -1, :]
                next_token = last_level_logits.argmax().item()
                if next_token == eos_token:
                    break
                inp = torch.cat(
                    [inp, torch.tensor([next_token]).unsqueeze(0).to(inp.device)], dim=1
                )

        return inp
