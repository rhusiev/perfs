import os
from typing import Iterator

from datasets import Dataset, load_dataset
import torch
from transformers import AutoTokenizer


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


class HFDatasetIterator:
    """An iterator that iterates over a Hugging Face dataset."""

    def __init__(self, dataset: str | Dataset, tokenizer_name: str) -> None:
        """Initialize the iterator.

        Args:
            dataset: The name of the dataset to use, or the dataset itself.
            tokenizer_name: The name of the tokenizer to use.
        """
        self.dataset = (
            load_dataset(dataset, "train") if isinstance(dataset, str) else dataset
        )
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Iterate over the dataset.

        Yields:
            The tokenized contents of each example.
        """
        batch = []
        for example in self.dataset["train"]["text"]:
            if len(example) == 0:
                continue
            example = torch.tensor(self.tokenizer.encode(example), dtype=torch.long)
            if example.size(0) > 1024:
                continue
            batch.append(example)
            if len(batch) == 8:
                max_len = max([example.size(0) for example in batch])
                batch = [
                    torch.cat(
                        [
                            example,
                            torch.zeros(max_len - example.size(0), dtype=torch.long),
                        ]
                    )
                    for example in batch
                ]
                attention_mask = torch.stack(
                    [
                        torch.cat(
                            [
                                torch.ones(example.size(0)),
                                torch.zeros(max_len - example.size(0) - 1),
                            ]
                            if max_len > example.size(0)
                            else [torch.ones(max_len - 1)]
                        )
                        for example in batch
                    ]
                ).to(torch.long)
                yield torch.stack(batch).to(torch.long), attention_mask
                batch = []
