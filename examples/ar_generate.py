#!/usr/bin/env python3
"""
Autoregressive generation with the context/AR tower (runs in stock mlx-lm).

    python examples/ar_generate.py --quant 4bit --prompt "The key idea behind Mamba is"

No custom code needed — the AR tower is a standard NemotronH model.
"""
import argparse
from mlx_lm import load, generate

REPO = "pipenetwork/Nemotron-3-Nano-30B-A3B-context-mlx-{quant}"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quant", default="4bit", choices=["4bit", "6bit", "8bit", "bf16"])
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--max-tokens", type=int, default=128)
    a = p.parse_args()

    model, tok = load(REPO.format(quant=a.quant))
    text = generate(model, tok, prompt=a.prompt, max_tokens=a.max_tokens, verbose=True)
    print(text)

if __name__ == "__main__":
    main()
