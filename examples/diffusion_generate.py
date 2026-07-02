#!/usr/bin/env python3
"""
Block-wise mask-diffusion generation with the full TwoTower model.

    # download once (see scripts/download.sh), then:
    python examples/diffusion_generate.py --model ./tt-4bit \
        --prompt "The capital of France is" --max-new-tokens 64

Requires the bundled nemotron_twotower_mlx.py (imported by run_twotower_mlx.load).
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mlx.core as mx
from run_twotower_mlx import load

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="local path to a downloaded TwoTower repo")
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--steps-per-block", type=int, default=16)
    p.add_argument("--mask-token-id", type=int, default=3)
    p.add_argument("--confidence-threshold", type=float, default=0.9)
    a = p.parse_args()

    model, tok = load(a.model)
    ids = mx.array([tok(a.prompt)["input_ids"]])
    out = model.generate_mask_diffusion(
        ids, max_new_tokens=a.max_new_tokens, block_size=a.block_size,
        steps_per_block=a.steps_per_block, mask_token_id=a.mask_token_id,
        confidence_threshold=a.confidence_threshold, eos_token_id=tok.eos_token_id,
        verbose=True,
    )
    print("\n" + tok.decode(out[0].tolist()))
    print(f"\n[{model.last_nfe} denoiser evaluations]")

if __name__ == "__main__":
    main()
