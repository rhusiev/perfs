from datasets import load_dataset
import torch

from pefts.dataset import HFDatasetIterator
from pefts.layers.lora import LoRAPeft
from pefts.model import Transformer
from pefts.trainer import Trainer

if __name__ == "__main__":
    from_map = {"human": "user", "gpt": "assistant"}
    format_message = (
        lambda x: f"{from_map.get(x['from'], x['from'])}\n{x['value']}<|endoftext|>"
    )
    dataset = HFDatasetIterator(
        load_dataset("totally-not-an-llm/sharegpt-hyperfiltered-3k", "train").map(
            lambda x: {"text": "".join(format_message(y) for y in x["conversations"])}
        ),
        "gpt2",
    )
    peft = LoRAPeft(lora_rank=4)
    model = Transformer.from_pretrained("gpt2", peft=peft)

    # Now we freeze everything except the LoRALinear layers
    for name, param in model.named_parameters():
        if "peft" not in name:
            param.requires_grad = False

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    trainer = Trainer(model, optimizer)
    trainer.train(dataset, epochs=1)
    torch.save(peft.state_dict(), "tune.pt")
    # peft.load_state_dict(torch.load("tune.pt"))
