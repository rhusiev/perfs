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
        if encoded_prompt.size(0) > 500:
            encoded_prompt = encoded_prompt[-500:]
        generated = self.model.generate(encoded_prompt, max_tokens)
        return self.tokenizer.decode(generated[0].tolist()[len(encoded_prompt):])


if __name__ == "__main__":
    pretrained_name = "gpt2"

    print("LoRA")
    peft = LoRAPeft(lora_rank=6)
    file_path = "movies_lora.pt"
    inference = Inference(peft=(peft, file_path), pretrained_name=pretrained_name)
    print(inference("I liked this movie for the most part, but have to say had there been anyone else besides Bill Murray in the lead role it would not have been as good. He brings an energy to the role that steps this film up a notch than it would have been otherwise. I mainly enjoyed the pranks pulled on the one counselor and there are other humorous things in this movie too such as the hot dog eating contest. This movie would also set the stage for summer camp movies with the competition at the end. Nearly every camp movie has either this or the unruly or troubled kids plot, or a combination of both. This series also would take a rather strange shift in tone as this one and two are both family friendly movies while part three and four are more adult oriented, more like the old teen sex comedies of the time. It kind of did the opposite of the Police Academy movies that went from R to PG-13 to PG movies. This series goes from the opposite to R. Still this first one and only good one is worth some chuckles largely due to Bill Murray\nSentiment:", 20))

    # print("LoKr")
    # peft = LoKrPeft(36, 6)
    # file_path = "movies_lokr.pt"
    # inference = Inference(peft=(peft, file_path), pretrained_name=pretrained_name)
    # print(inference("", 2))
