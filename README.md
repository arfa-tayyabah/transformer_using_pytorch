# Transformer from Scratch — PyTorch

<p align="center">
  <em>A clean, from-scratch PyTorch implementation of the Transformer architecture ("Attention Is All You Need")</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-1.13%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="License">
  <img src="https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square" alt="Status">
 
</p>

---

## Overview

This repository contains a **from-scratch implementation of the Transformer encoder-decoder architecture** in PyTorch, built directly from the original *"Attention Is All You Need"* paper (Vaswani et al., 2017). Every core component i.e. multi-head attention, positional encoding, layer normalization, residual connections, and the feed-forward network, is implemented manually using low-level `torch.nn` primitives, with no reliance on pre-built `nn.Transformer` modules.

The goal of this project is to demonstrate a deep, working understanding of transformer internals: how attention scores are computed, how residual/normalization sublayers are composed, and how encoder and decoder stacks fit together into a full sequence-to-sequence model.

## Why This Project

Building a transformer from the ground up (rather than importing one) demonstrates:

- Solid understanding of the **mathematics behind self-attention and scaled dot-product attention**
- Ability to translate a research paper into **clean, modular, production-style code**
- Familiarity with **PyTorch's `nn.Module` design patterns** (composition of submodules, parameter registration, buffers)
- Practical knowledge of sequence modeling concepts: masking, positional encoding, embedding scaling, and layer stacking

## Architecture

The implementation follows the standard encoder–decoder transformer design:

```
Input Sequence                         Target Sequence
      │                                       │
InputEmbedding + PositionalEncoding    InputEmbedding + PositionalEncoding
      │                                       │
      ▼                                       ▼
 ┌─────────────┐                      ┌─────────────────┐
 │  Encoder    │  ──── encoder ────▶  │     Decoder      │
 │  (N layers) │       output         │    (N layers)    │
 └─────────────┘                      └─────────────────┘
                                              │
                                       ProjectionLayer
                                              │
                                        Output Logits
```

Each **Encoder Block** consists of:
1. Multi-Head Self-Attention → Residual Connection + LayerNorm
2. Position-wise Feed-Forward Network → Residual Connection + LayerNorm

Each **Decoder Block** consists of:
1. Masked Multi-Head Self-Attention → Residual Connection + LayerNorm
2. Multi-Head Cross-Attention (attending to encoder output) → Residual Connection + LayerNorm
3. Position-wise Feed-Forward Network → Residual Connection + LayerNorm

## Implemented Components

| Module | Description |
|---|---|
| `InputEmbedding` | Token embeddings scaled by `√d_model` |
| `PositionalEncoding` | Fixed sinusoidal positional encodings (sin/cos) |
| `MultiHeadAttention` | Scaled dot-product attention split across multiple heads, with masking support |
| `FeedForwardBlock` | Two-layer position-wise feed-forward network with ReLU activation |
| `NormalizationLayer` | Custom layer normalization with learnable scale (`alpha`) and shift (`bias`) |
| `ResidualConnection` | Pre-norm residual wrapper (`x + dropout(sublayer(norm(x)))`) |
| `EncoderBlock` / `Encoder` | Stack of self-attention + feed-forward sublayers |
| `DecoderBlock` / `Decoder` | Stack of self-attention, cross-attention, and feed-forward sublayers |
| `ProjectionLayer` | Final linear layer mapping model dimensions to target vocabulary logits |
| `Transformer` | Full encoder–decoder model, with a `build_transformer()` factory method for easy instantiation |

## Getting Started

### Prerequisites

- Python 3.9+
- PyTorch 1.13+

### Installation

```bash
git clone https://github.com/arfa-tayyabah/transformer_using_pytorch.git
cd transformer_using_pytorch
pip install torch
```

### Usage

Build a transformer model using the provided factory method:

```python
from model import Transformer

model = Transformer.build_transformer(
    src_vocab=32000,     # source vocabulary size
    trgt_vocab=32000,    # target vocabulary size
    src_seq=350,         # source max sequence length
    trgt_seq=350,        # target max sequence length
    features=512,        # model (embedding) dimension
    N=6,                 # number of encoder/decoder layers
    h=8,                 # number of attention heads
    dropout=0.1,
    d_ff=2048            # feed-forward hidden dimension
)

# Encode a source sequence
encoder_output = model.encode(src_input_ids)

# Decode with target sequence and causal mask
decoder_output = model.decode(encoder_output, trgt_input_ids, trgt_mask)

# Project to vocabulary logits
logits = model.project(decoder_output)
```

Model weights are initialized using **Xavier uniform initialization** for all parameters with more than one dimension, consistent with common transformer training practices.

## Project Structure

```
transformer_using_pytorch/
│
├── model.py          # Full transformer implementation
└── README.md
```

## Roadmap

- [ ] Add training script with sample dataset (e.g., translation task)
- [ ] Add inference / greedy & beam search decoding
- [ ] Add unit tests for attention and masking logic
- [ ] Add configuration file (YAML/JSON) for hyperparameters
- [ ] Add TensorBoard / Weights & Biases logging

## References

- Vaswani, A. et al. (2017). [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762)

## License

This project is licensed under the [MIT License](LICENSE).

---

<p align="center">
  <sub>Built with PyTorch to demonstrate a hands-on understanding of transformer architectures.</sub>
</p>
