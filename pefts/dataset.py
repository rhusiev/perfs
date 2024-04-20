import os
from typing import Iterator

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
