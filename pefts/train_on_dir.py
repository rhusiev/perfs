import torch

from pefts.dataset import RecursiveDirectoryListerDataset
from pefts.layers.lora import LoRAPeft
from pefts.layers.lokr import LoKrPeft
from pefts.model import Transformer
from pefts.trainer import Trainer

if __name__ == "__main__":
    dataset = RecursiveDirectoryListerDataset("dataset", "gpt2")
    peft = LoRAPeft(lora_rank=4)
    # peft = LoKrPeft()
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
    torch.save(peft.state_dict(), "lora.pt")
