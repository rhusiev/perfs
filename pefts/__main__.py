import torch

from .dataset import RecursiveDirectoryListerDataset
from .model import Transformer
from .trainer import Trainer

if __name__ == "__main__":
    dataset = RecursiveDirectoryListerDataset("data", "gpt2")
    model = Transformer.from_pretrained("gpt2")
    optimizer = torch.optim.Adam(model.parameters())
    trainer = Trainer(model, optimizer)
    trainer.train(dataset, epochs=1)
