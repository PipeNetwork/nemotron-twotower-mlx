# Nemotron TwoTower · MLX

Run NVIDIA's **Nemotron-Labs-TwoTower-30B-A3B** — a block-wise autoregressive **diffusion** language model — natively on Apple Silicon with [MLX](https://github.com/ml-explore/mlx).

<p>
<img alt="MLX" src="https://img.shields.io/badge/MLX-Apple%20Silicon-black">
<img alt="Model" src="https://img.shields.io/badge/model-Nemotron%20TwoTower%2030B--A3B-76B900">
<img alt="Arch" src="https://img.shields.io/badge/arch-Mamba2%20%2B%20Attn%20%2B%20MoE-blue">
<img alt="License" src="https://img.shields.io/badge/license-NVIDIA%20Open%20Model-lightgrey">
</p>

This repo bundles the MLX modeling code + copy-paste examples for two deliverables converted from the [source model](https://huggingface.co/nvidia/Nemotron-Labs-TwoTower-30B-A3B-Base-BF16):

| | Runs in | What it is |
|---|---|---|
| **AR / context tower** | stock `mlx-lm` | the frozen autoregressive backbone — an ordinary text model |
| **Full TwoTower diffusion** | this repo's code | the real two-tower **mask-diffusion** generator |

---

## ✨ Features

| | |
|---|---|
| 🍎 **Native Apple Silicon** | pure MLX — no CUDA, no PyTorch for inference |
| 🌊 **Real diffusion generation** | block-wise mask diffusion with confidence-based unmasking |
| 🧩 **Hybrid backbone** | 52 layers = 23 Mamba-2 · 6 attention · 23 MoE (128 experts, 6 active + 1 shared) |
| 🪶 **~3B active / token** | 30B total per tower, MoE-sparse |
| 📦 **4 / 6 / 8-bit + bf16** | mixed-precision quant tuned so diffusion stays coherent at 4-bit |
| ✅ **Verified** | token-for-token parity vs NVIDIA's CUDA reference (see [Validation](#-validation)) |

## 📐 Architecture

```
 prompt ──▶ ┌──────────────────┐   KV + Mamba states   ┌──────────────────┐
            │  CONTEXT TOWER   │ ────────────────────▶ │  DENOISER TOWER  │
            │  (frozen, AR)    │                       │  (diffusion)     │
            └──────────────────┘                       └────────┬─────────┘
                                                                │ adaLN(timestep)
   each block: start fully masked ──▶ denoise ×N ──▶ commit ────┘
   high-confidence tokens, remask the rest, then extend the context.
```

## 🗂️ Which model do I download?

Sizes and rough RAM needs (unified memory):

| Quant | AR / context tower | Full TwoTower diffusion |
|---|---|---|
| **4-bit** | [~17 GB](https://huggingface.co/pipenetwork/Nemotron-3-Nano-30B-A3B-context-mlx-4bit) · 32 GB Mac | [~34 GB](https://huggingface.co/pipenetwork/Nemotron-Labs-TwoTower-30B-A3B-mlx-4bit) · 48 GB Mac |
| **6-bit** | [~24 GB](https://huggingface.co/pipenetwork/Nemotron-3-Nano-30B-A3B-context-mlx-6bit) | [~48 GB](https://huggingface.co/pipenetwork/Nemotron-Labs-TwoTower-30B-A3B-mlx-6bit) · 64 GB Mac |
| **8-bit** | [~30 GB](https://huggingface.co/pipenetwork/Nemotron-3-Nano-30B-A3B-context-mlx-8bit) | [~63 GB](https://huggingface.co/pipenetwork/Nemotron-Labs-TwoTower-30B-A3B-mlx-8bit) · 96 GB Mac |
| **bf16** | [~57 GB](https://huggingface.co/pipenetwork/Nemotron-3-Nano-30B-A3B-context-mlx-bf16) | [~118 GB](https://huggingface.co/pipenetwork/Nemotron-Labs-TwoTower-30B-A3B-mlx-bf16) · 128 GB Mac |

Not sure? Start with **AR 4-bit** (smallest, runs anywhere) or **diffusion 4-bit** for the real two-tower behavior.

## 🛠️ Prerequisites

- macOS on Apple Silicon (M1/M2/M3/M4)
- Python 3.9+
- RAM per the table above

```bash
git clone https://github.com/PipeNetwork/nemotron-twotower-mlx.git
cd nemotron-twotower-mlx
pip install -r requirements.txt        # mlx, mlx-lm, transformers
pip install "huggingface_hub[hf_transfer]"   # faster downloads (optional)
```

## 🚀 Quick start

### A) Autoregressive (stock mlx-lm — no download step needed)

```bash
# one-liner
mlx_lm.generate --model pipenetwork/Nemotron-3-Nano-30B-A3B-context-mlx-4bit \
  --prompt "The key idea behind Mamba is" --max-tokens 128

# or via the example
python examples/ar_generate.py --quant 4bit --prompt "The capital of France is"
```

```python
from mlx_lm import load, generate
model, tok = load("pipenetwork/Nemotron-3-Nano-30B-A3B-context-mlx-4bit")
print(generate(model, tok, prompt="The capital of France is", max_tokens=128))
```

### B) Full TwoTower diffusion (uses this repo's code)

```bash
# 1. download a build
scripts/download.sh diff 4bit ./tt-4bit

# 2. generate by mask diffusion
python run_twotower_mlx.py --model ./tt-4bit \
  --prompt "The capital of France is" --max-new-tokens 64 \
  --block-size 16 --steps-per-block 16 --mask-token-id 3
```

```python
import sys; sys.path.insert(0, ".")
from run_twotower_mlx import load
import mlx.core as mx
model, tok = load("./tt-4bit")
ids = mx.array([tok("The capital of France is")["input_ids"]])
out = model.generate_mask_diffusion(ids, max_new_tokens=64, block_size=16,
        steps_per_block=16, mask_token_id=3, eos_token_id=tok.eos_token_id)
print(tok.decode(out[0].tolist()))
# -> "The capital of France is Paris, the capital of Germany is Berlin, ..."
```

## ⚙️ Configuration (diffusion)

| Flag | Default | Notes |
|---|---|---|
| `--max-new-tokens` | 64 | must be divisible by `--block-size` |
| `--block-size` | 16 | tokens denoised per block |
| `--steps-per-block` | 16 | denoising iterations per block |
| `--mask-token-id` | **3** | the model's mask token (training convention) |
| `--confidence-threshold` | 0.9 | commit tokens above this confidence |

## 🧊 Quantization (`mixed_v1`)

Diffusion compounds quantization error across denoising steps, so a naive uniform 4/6-bit produces degenerate output. The quantized builds use a mixed scheme: **timestep-conditioning MLPs stay bf16**, **embeddings & LM heads stay ≥8-bit**, and only the bulk (MoE experts, attention & Mamba projections) is quantized to the target bits. The loader reconstructs this automatically from `config.json` — nothing to configure.

## ✅ Validation

The MLX conversion was checked against NVIDIA's reference implementation running on an NVIDIA GB10 (CUDA). Greedy decoding matched **token-for-token — 120/120 tokens (100%), 5/5 top-1** across the test prompts (e.g. both produce *"George Washington. He was elected in 1789 and served two terms until 1797."*). The AR tower is the shared backbone the diffusion denoiser also uses.

## 📁 Project structure

```
nemotron-twotower-mlx/
├── README.md
├── requirements.txt
├── nemotron_twotower_mlx.py   # MLX TwoTower diffusion model
├── run_twotower_mlx.py        # diffusion CLI + load()
├── examples/
│   ├── ar_generate.py         # AR tower via stock mlx-lm
│   └── diffusion_generate.py  # full two-tower diffusion
└── scripts/
    └── download.sh            # fetch a chosen build
```

## 🐛 Troubleshooting

| Problem | Fix |
|---|---|
| Garbage / repetitive diffusion output | ensure `--mask-token-id 3`; use a 6/8-bit build if 4-bit looks weak |
| `max_new_tokens must be divisible by block_size` | pick e.g. 64 with block-size 16 |
| Out of memory | use a smaller quant, or the AR tower instead of diffusion |
| `ModuleNotFoundError: nemotron_twotower_mlx` | run from the repo root, or keep it next to the model folder |
| Slow first run | MLX compiles kernels on first use; subsequent runs are faster |

## 📚 Resources

- Source model: [nvidia/Nemotron-Labs-TwoTower-30B-A3B-Base-BF16](https://huggingface.co/nvidia/Nemotron-Labs-TwoTower-30B-A3B-Base-BF16)
- MLX models: [AR tower](https://huggingface.co/pipenetwork/Nemotron-3-Nano-30B-A3B-context-mlx-4bit) · [TwoTower diffusion](https://huggingface.co/pipenetwork/Nemotron-Labs-TwoTower-30B-A3B-mlx-4bit)
- [MLX](https://github.com/ml-explore/mlx) · [mlx-lm](https://github.com/ml-explore/mlx-lm)

## 📝 License

Model weights and code are governed by the [NVIDIA Open Model License](https://huggingface.co/nvidia/Nemotron-Labs-TwoTower-30B-A3B-Base-BF16) of the base model. The MLX porting code in this repo is provided as-is under the same terms.
