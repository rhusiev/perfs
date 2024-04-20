from dataclasses import dataclass
import os
from typing import Iterable, Iterator, overload

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers import GPT2LMHeadModel
from transformers.activations import NewGELUActivation


@dataclass
class Config:
    """The configuration for a Transformer model.

    Attributes:
        n_layers: The number of transformer layers.
        n_heads: The number of attention heads.
        d_model: The size of the hidden dimension.
        vocab_size: The size of the vocabulary.
        block_size: The size of the input block.
    """

    n_layers: int
    n_heads: int
    d_model: int
    vocab_size: int
    block_size: int

    @property
    def d_embedding(self) -> int:
        """The size of the key/query embedding dimension
        (d_k in the Attention is All You Need paper)."""
        return self.d_model // self.n_heads


class MultiHeadAttention(nn.Module):
    """A multi-head attention layer."""

    def __init__(self, config: Config) -> None:
        """Initialize the layer.

        Args:
            config: The configuration for the model.
        """
        super().__init__()

        # key, query, value projections for all heads, but as one concatenated tensor
        # We're keeping them conctenated just so that we can do LoRA finetunes
        # on them - otherwise we would split them up for readability
        self.q = nn.Linear(config.d_model, config.d_model)
        self.k = nn.Linear(config.d_model, config.d_model)
        self.v = nn.Linear(config.d_model, config.d_model)

        self.output = nn.Linear(config.d_model, config.d_model)

        self.config = config

    def _split_heads(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Split the key, query, and value tensors into multiple heads.

        Args:
            e: The key, query, or value tensor,
                with shape (batch_size, seq_len, d_model).
            seq_len: The length of the input sequence.

        Returns:
            The key, query, or value tensor with an additional head dimension,
            with shape (batch_size, n_heads, seq_len, d_embedding).
        """
        return x.view(
            -1, seq_len, self.config.n_heads, self.config.d_embedding
        ).transpose(1, 2)

    def _mask_out_future(self, att: torch.Tensor) -> torch.Tensor:
        """Mask out the future in the attention weights.
        This is the triangle thingy for the "causal LM" part.

        Args:
            att: The attention weights,
                with shape (batch_size, n_heads, seq_len, seq_len).

        Returns:
            The attention weights with the future masked out.
        """
        return att.masked_fill(
            torch.tril(torch.ones(att.size(-2), att.size(-1))).view(
                1, 1, att.size(-2), att.size(-1)
            )
            == 0,
            float("-inf"),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform the forward pass through the
        multi-head attention layer.

        Args:
            x: The input embedding, with shape (batch_size, seq_len, d_model).

        Returns:
            The output embedding, with shape (batch_size, seq_len, d_model).
        """
        seq_len = x.size(1)

        q = self._split_heads(self.q(x), seq_len)
        k = self._split_heads(self.k(x), seq_len)
        v = self._split_heads(self.v(x), seq_len)

        att = (q @ k.transpose(-2, -1)) * (self.config.d_embedding**-0.5)
        att = self._mask_out_future(att)
        att = F.softmax(att, dim=-1)

        y = (att @ v).transpose(1, 2).contiguous().view(*x.size())

        return self.output(y)


class FeedForward(nn.Module):
    """A feedforward layer for the transformer."""

    def __init__(self, config: Config) -> None:
        """Initialize the layer.

        Args:
            config: The configuration for the model.
        """
        super().__init__()

        self.upprojection = nn.Linear(config.d_model, 4 * config.d_model)
        self.downprojection = nn.Linear(4 * config.d_model, config.d_model)
        self.activation = NewGELUActivation()

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        """Perform the forward pass through the
        feedforward layer.

        Args:
            e: The input embedding, with shape (batch_size, seq_len, d_model).

        Returns:
            The output embedding, with shape (batch_size, seq_len, d_model).
        """
        return self.downprojection(self.activation(self.upprojection(e)))


class TransformerBlock(nn.Module):
    """A single simple transformer block with a multi-head attention"""

    def __init__(self, config: Config) -> None:
        """Initialize the block.

        Args:
            config: The configuration for the model.
        """
        super().__init__()

        # Keeping some of their names so that GPT2 weights can be loaded
        # more easily
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.ln_2 = nn.LayerNorm(config.d_model)

        self.attn = MultiHeadAttention(config)
        # AKA mlp, multi-layer perceptron:
        self.feedforward = FeedForward(config)

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        """Perform the forward pass through the
        transformer block.

        Args:
            e: The input embedding, with shape (batch_size, seq_len, d_model).

        Returns:
            The output embedding, with shape (batch_size, seq_len, d_model).
        """
        # I wish I could just normalize the input *after* the attention
        # and feedforward layers, to match the Attention is All You Need
        # paper, but the pretrained GPT-2 model does it before, so I'm
        # doing it here as well. I tried changing it around, but it become
        # way too complicated (cause remember - you skip the normalization)
        # even before I could get it working.

        # Note: we can't use += - PyTorch doesn't support it, saving gradients
        # breaks it. So we have to do it like this.
        e = e + self.attn(self.ln_1(e))
        e = e + self.feedforward(self.ln_2(e))
        return e


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
        self.tower = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.ln_f = nn.LayerNorm(config.d_model)
        self.unembedding = nn.Linear(config.d_model, config.vocab_size)

    @classmethod
    def from_pretrained(cls, model_name: str) -> "Transformer":
        """Load a pretrained model from Hugging Face's Transformers library.
        This closely follows Karpathy's minGPT loader, it was used as a reference.

        Args:
            model_name: The name of the model to load.
                Should be one of "gpt2".

        Returns:
            The pretrained model.
        """

        configs = {
            "gpt2": Config(
                n_layers=12, n_heads=12, d_model=768, vocab_size=50257, block_size=1024
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
                print(k)
                continue

            print(k)

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
        self, x: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ...

    def forward(
        self, x: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Perform the forward pass through the
        transformer model.

        Args:
            x: The input tensor, with shape (batch_size, seq_len).
            targets: The target tensor, with shape (batch_size, seq_len).

        Returns:
            The logits, with shape (batch_size, seq_len, vocab_size).
            The loss, if targets are provided.
        """
        e = self.token_embeddings(x) + self.position_embeddings(
            torch.arange(x.size(1), device=x.device, dtype=torch.long)
            .to(x.device)
            .unsqueeze(0)
        )
        for block in self.tower:
            e = block(e)
        e = self.ln_f(e)
        logits = self.unembedding(e)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    def generate(
        self,
        prompt: list[int] | torch.Tensor,
        max_len: int,
    ) -> torch.Tensor:
        """Generate a sequence of tokens from a prompt.

        Args:
            prompt: The prompt to generate from.
            max_len: The maximum length of the generated sequence.
            tokenizer: The tokenizer to use.

        Returns:
            The generated sequence of tokens.
        """
        inp = torch.tensor(prompt).unsqueeze(0)
        self.eval()
        with torch.no_grad():
            for _ in range(max_len):
                logits = self.forward(inp)[0]
                last_level_logits = logits[0, -1, :]
                next_token = last_level_logits.argmax().item()
                inp = torch.cat([inp, torch.tensor([next_token]).unsqueeze(0)], dim=1)

        return inp


class Trainer:
    """A simple trainer for the transformer model."""

    def __init__(self, model: Transformer, optimizer: torch.optim.Optimizer) -> None:
        """Initialize the trainer.

        Args:
            model: The model to train.
            optimizer: The optimizer to use.
        """
        self.model = model
        self.optimizer = optimizer

    def train(self, dataset: Iterable[torch.Tensor], epochs: int = 1) -> None:
        """Train the model on the given dataset.

        Args:
            dataset: The dataset to train on.
            epochs: The number of epochs to train for.
        """
        for _ in range(epochs):
            for batch in dataset:
                self.optimizer.zero_grad()
                _, loss = self.model(batch[:, :-1], batch[:, 1:])
                loss.backward()
                self.optimizer.step()


class RecursiveDirectoryListerDataset:
    """A dataset that lists all files in a directory recursively."""

    def __init__(self, root: str, tokenizer_name: str) -> None:
        """Initialize the dataset.

        Args:
            root: The root directory to list files from.
            tokenizer_name: The name of the tokenizer to use.
        """
        self.root = root
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def __iter__(self) -> Iterator[torch.Tensor]:
        """Iterate over the files in the dataset.

        Yields:
            The tokenized contents of each file.
        """
        for root, _, files in os.walk(self.root):
            for file in files:
                print(file)
                with open(os.path.join(root, file), "r") as f:
                    data = f.read()
                    if len(data) == 0:
                        continue
                    example = torch.tensor(
                        self.tokenizer.encode(data), dtype=torch.long
                    )

                    # We set 512 as the step for it to learn to start
                    # from the middle of a file as well
                    for i in range(0, len(example), 512):
                        yield example[i : i + 1024].unsqueeze(0)


if __name__ == "__main__":
    dataset = RecursiveDirectoryListerDataset("data", "gpt2")
    model = Transformer.from_pretrained("gpt2")
    optimizer = torch.optim.Adam(model.parameters())
    trainer = Trainer(model, optimizer)
    trainer.train(dataset, epochs=1)
