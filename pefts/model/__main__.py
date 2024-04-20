from transformers import AutoTokenizer

from .transformer import Transformer

if __name__ == "__main__":
    model = Transformer.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    prompt = "Who was Jim Henson? Jim Henson was a"
    prompt = tokenizer.encode(prompt)
    generated = model.generate(prompt, 10)
    print(tokenizer.decode(generated[0].tolist()))
