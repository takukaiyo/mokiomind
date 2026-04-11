# MokioMind

A from-scratch implementation of a modern Transformer-based causal language model, built for learning and experimentation.

MokioMind implements the core architecture of contemporary LLMs (Llama-style) with full training pipeline support — from pretraining to RLHF alignment.

## Architecture

```
Input Token IDs
       │
       ▼
┌─────────────┐
│  Embedding   │  vocab_size → hidden_size
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────┐
│         MokioMindBlock × 8          │
│                                     │
│  ┌───────────┐    ┌──────────────┐  │
│  │  RMSNorm   │───▶│  GQA Attention │  │
│  └───────────┘    │  + RoPE       │  │
│       │           │  + KV Cache   │  │
│       │           │  + Flash Attn │  │
│       │           └──────┬───────┘  │
│       └──── + residual ──┘          │
│                  │                  │
│  ┌───────────┐   │                  │
│  │  RMSNorm   │◀──┘                  │
│  └───────────┘                      │
│       │    ┌──────────────┐         │
│       └───▶│  SwiGLU FFN   │         │
│            └──────┬───────┘         │
│       ┌── + residual ──┘            │
│       │                             │
└───────┼─────────────────────────────┘
       │
       ▼
┌─────────────┐
│   RMSNorm    │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   lm_head    │  hidden_size → vocab_size (weight tying)
└──────┬──────┘
       │
       ▼
     Logits
```

## Key Components

| Component | Implementation | Description |
|-----------|---------------|-------------|
| Normalization | RMSNorm | Faster than LayerNorm, no mean subtraction |
| Position Encoding | RoPE + YaRN | Rotary embeddings with long-context extrapolation |
| Attention | GQA (8Q / 2KV) | Grouped-Query Attention, reduces KV cache memory |
| Acceleration | Flash Attention | PyTorch 2.0+ `scaled_dot_product_attention` |
| Feed-Forward | SwiGLU | Gated activation: `down(SiLU(gate(x)) * up(x))` |
| Weight Sharing | Embedding ↔ lm_head | Reduces parameters by tying input/output embeddings |

## Default Model Config

```
hidden_size:        512
num_attention_heads: 8
num_key_value_heads: 2
num_hidden_layers:   8
vocab_size:         6400
max_position:       32768
intermediate_size:  auto (hidden * 8/3, aligned to 64)
total parameters:   ~26M
```

## Training Pipeline

MokioMind supports four training paradigms covering the full LLM lifecycle:

| Stage | Dataset Class | What the model learns |
|-------|--------------|----------------------|
| Pretrain | `PretrainDataset` | Language modeling on raw text (next-token prediction) |
| SFT | `SFTDataset` | Follow instructions (loss only on assistant replies) |
| DPO | `DPODataset` | Preference alignment (chosen vs rejected responses) |
| RLAIF | `RLAIFDataset` | RL-based alignment (returns raw strings for online rollout) |

Training features:
- Cosine annealing LR schedule
- Gradient accumulation (simulate large batch on limited VRAM)
- Mixed precision training (AMP with GradScaler)
- Gradient clipping
- Checkpoint save/resume

## Project Structure

```
mokiomind/
├── model/
│   ├── model.py              # Model architecture (Config, RMSNorm, RoPE, Attention, FFN, CausalLM)
│   └── __init__.py
├── dataset/
│   ├── lm_dataset.py         # Dataset classes (Pretrain, SFT, DPO, RLAIF)
│   └── __init__.py
├── trainer/
│   ├── trainer_utils.py      # Training utilities (LR schedule, train loops, checkpoint)
│   ├── train_pretrain.py     # Pretraining entry point
│   └── __init__.py
├── model_study.py            # Annotated version of model.py for learning
├── pyproject.toml            # Project config & dependencies
└── README.md
```

## Quick Start

### Installation

```bash
# Clone
git clone https://github.com/takukaiyo/mokiomind.git
cd mokiomind

# Install dependencies (using uv)
uv sync

# Or using pip
pip install -e .
```

### Prepare Data

Pretraining data should be a JSONL file, one sample per line:

```json
{"text": "这是一段用于预训练的文本。模型会学习预测下一个 token。"}
{"text": "Another piece of training text. The model learns next-token prediction."}
```

### Pretrain

```bash
python -m trainer.train_pretrain \
    --data_path data/pretrain_data.jsonl \
    --tokenizer_path tokenizer \
    --epochs 3 \
    --batch_size 32 \
    --max_lr 5e-4
```

### Verify Model

```python
from model.model import MokioMindConfig, MokioMindForCausalLM
import torch

config = MokioMindConfig()
model = MokioMindForCausalLM(config)

x = torch.randint(0, config.vocab_size, (2, 32))
out = model(input_ids=x, labels=x)
print(f"logits: {out.logits.shape}")  # [2, 32, 6400]
print(f"loss: {out.loss.item():.4f}")
```

## Requirements

- Python >= 3.13
- PyTorch >= 2.9.0
- Transformers >= 4.57.1
- Datasets >= 4.8.4

## HuggingFace Compatibility

MokioMind inherits from `PreTrainedModel` and `GenerationMixin`, so it works with the HuggingFace ecosystem out of the box:

```python
# Save
model.save_pretrained("my_model")

# Load
model = MokioMindForCausalLM.from_pretrained("my_model")

# Generate (after training)
output_ids = model.generate(input_ids, max_new_tokens=50)
```

## License

MIT
