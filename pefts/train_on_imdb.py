from datasets import load_dataset
import torch
from transformers import AutoTokenizer

from pefts.dataset import HFDatasetIterator
from pefts.layers.lora import LoRAPeft
from pefts.model import Transformer
from pefts.trainer import Trainer

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    format_message = (
        lambda x: f"{x["review"]}{tokenizer.eos_token}{"negative" if x["label"] else "positive"}{tokenizer.eos_token}"
    )
    dataset = HFDatasetIterator(
        {
            "train": load_dataset("ajaykarthick/imdb-movie-reviews")["train"].map(
                lambda x: {"text": format_message(x)}
            ),
        },
        "gpt2",
    )
    peft = LoRAPeft()
    model = Transformer.from_pretrained("gpt2", peft=peft)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Now we freeze everything except the LoRALinear layers
    for name, param in model.named_parameters():
        if "peft" not in name:
            param.requires_grad = False

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    trainer = Trainer(model, optimizer)
    trainer.train(dataset, epochs=1)
    torch.save(peft.state_dict(), "movies_lora.pt")
    # peft.load_state_dict(torch.load("tune.pt"))
