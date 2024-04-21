from datasets import load_dataset
import torch
from transformers import AutoTokenizer

from pefts.dataset import HFDatasetIterator
from pefts.layers.lora import LoRAPeft
from pefts.layers.loha import LoHaPeft
from pefts.layers.lokr import LoKrPeft
from pefts.model import Transformer
from pefts.trainer import Trainer

from pefts.inference import Inference

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    format_message = lambda x: f'{x["review"]}{tokenizer.eos_token}'
    dataset = (
        load_dataset("ajaykarthick/imdb-movie-reviews")["test"]
        .shuffle(0)
        .select(range(2000))
        .map(
            lambda x: {"text": format_message(x), "label": x["label"]},
        )
    )
    peft = LoKrPeft(36, 6)

    inference = Inference((peft, "movies_lokr.pt"), "gpt2")
    total = 0
    correct = 0
    for example in dataset:
        total += 1
        completion = inference(example["text"], 2)
        if "negative" if example["label"] else "positive" in completion:
            correct += 1
        print(
            f"\rAccuracy: {correct / total:.2%}  Done: {total}/{len(dataset)}",
            end="                        ",
        )
    print()
