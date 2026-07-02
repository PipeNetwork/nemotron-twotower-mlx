#!/usr/bin/env python3
"""
Run the MLX port of nvidia/Nemotron-Labs-TwoTower-30B-A3B (block-wise AR diffusion LM).

Example:
  python run_twotower_mlx.py --model /path/to/weights \\
    --prompt "The capital of France is" --max-new-tokens 64 \\
    --block-size 16 --steps-per-block 16 --mask-token-id 3

`--model` must contain: the (MLX) safetensors + index, config.json, tokenizer files.
The custom modeling code (nemotron_twotower_mlx.py) must be importable.
"""
import argparse, glob, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.nemotron_h import ModelArgs
from transformers import AutoTokenizer
from nemotron_twotower_mlx import TwoTowerModel, quantization_predicate


def load(model_path):
    cfg = json.load(open(f"{model_path}/config.json"))
    args = ModelArgs.from_dict(cfg)
    model = TwoTowerModel(args)
    # Quantized packages record {"group_size","bits","scheme"} in config; recreate
    # the same quantized module structure BEFORE loading so scales/biases fit.
    q = cfg.get("quantization")
    if q:
        pred = quantization_predicate(q["bits"], q["group_size"]) \
            if q.get("scheme") == "mixed_v1" else None
        nn.quantize(model, group_size=q["group_size"], bits=q["bits"],
                    class_predicate=pred)
    weights = {}
    for shard in sorted(glob.glob(f"{model_path}/*.safetensors")):
        weights.update(mx.load(shard))
    weights = model.sanitize(weights)  # idempotent
    model.load_weights(list(weights.items()), strict=True)
    model.eval()
    mx.eval(model.parameters())
    tok = AutoTokenizer.from_pretrained(model_path)
    return model, tok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--steps-per-block", type=int, default=16)
    p.add_argument("--mask-token-id", type=int, default=3)
    p.add_argument("--confidence-threshold", type=float, default=0.9)
    p.add_argument("--verbose", action="store_true")
    a = p.parse_args()

    t0 = time.time()
    model, tok = load(a.model)
    print(f"loaded in {time.time()-t0:.0f}s", file=sys.stderr)

    ids = mx.array([tok(a.prompt)["input_ids"]])
    eos = tok.eos_token_id
    t0 = time.time()
    out = model.generate_mask_diffusion(
        ids, max_new_tokens=a.max_new_tokens, block_size=a.block_size,
        steps_per_block=a.steps_per_block, mask_token_id=a.mask_token_id,
        confidence_threshold=a.confidence_threshold, eos_token_id=eos,
        verbose=a.verbose,
    )
    dt = time.time() - t0
    gen = out[0].tolist()
    n_new = len(gen) - ids.shape[1]
    print(tok.decode(gen))
    print(f"\n[{n_new} tokens, {model.last_nfe} denoiser evals, {dt:.1f}s, "
          f"{n_new/dt:.1f} tok/s]", file=sys.stderr)


if __name__ == "__main__":
    main()
