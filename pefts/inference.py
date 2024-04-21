import torch
from transformers import AutoTokenizer

from pefts.layers.lora import LoRAPeft
from pefts.layers.lokr import LoKrPeft
from pefts.peft import Peft
from pefts.model import Transformer


class Inference:
    """A class for inference with a pre-trained model."""

    def __init__(
        self, peft: tuple[Peft, str] | None = None, pretrained_name: str = "gpt2"
    ) -> None:
        """Load a pre-trained model with a given PEFT.

        Args:
            peft (tuple[Peft, str], optional): A tuple containing a PEFT
                and a file path to the PEFT. Defaults to None.
            pretrained_name (str, optional): The name of the pre-trained model.
            Defaults to "gpt2".
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if peft:
            model = Transformer.from_pretrained(pretrained_name, peft=peft[0])

            peft[0].load_state_dict(torch.load(peft[1]))
            self.peft = peft[0]
        else:
            model = Transformer.from_pretrained(pretrained_name)
            self.peft = None

        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_name)

        self.model.to(self.device)

    def __call__(self, prompt: str, max_tokens: int = 10) -> str:
        """Generate a completion for a given prompt.

        Args:
            prompt (str): The prompt to complete.
            max_tokens (int, optional): The maximum number of tokens to generate.

        Returns:
            str: The generated completion.
        """
        encoded_prompt = self.tokenizer.encode(prompt)
        encoded_prompt = torch.tensor(encoded_prompt).to(self.device)
        generated = self.model.generate(encoded_prompt, max_tokens)
        return self.tokenizer.decode(generated[0].tolist())


if __name__ == "__main__":
    pretrained_name = "gpt2"

    print("Original")
    inference = Inference(pretrained_name=pretrained_name)
    print(inference("Who was Jim Henson? Jim Henson was a"))

    print("LoRA")
    peft = LoRAPeft(lora_rank=4)
    file_path = "lora.pt"
    inference = Inference(peft=(peft, file_path), pretrained_name=pretrained_name)
    print(inference("Who was Jim Henson? Jim Henson was a"))

    print("LoKr")
    peft = LoKrPeft()
    file_path = "lokr.pt"
    inference = Inference(peft=(peft, file_path), pretrained_name=pretrained_name)
    print(inference("Who was Jim Henson? Jim Henson was a"))
