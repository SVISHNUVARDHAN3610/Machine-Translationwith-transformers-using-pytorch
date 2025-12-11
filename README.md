<div align="center">

# 🌐 Machine Translation with Transformers
### Sequence-to-Sequence Modeling using PyTorch & Self-Attention

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![NLP](https://img.shields.io/badge/Task-Machine%20Translation-green?style=for-the-badge)]()
[![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](LICENSE)

> **"Attention Is All You Need."**
> A complete PyTorch implementation of the Transformer architecture for high-quality English-to-German translation, achieving superior parallelization over RNNs.

[Project Overview](#overview) • [Architecture](#architecture) • [Key Features](#features) • [Installation](#installation) • [Results](#results)

</div>

---

## <a name="overview"></a>🧐 Project Overview

Traditional Sequence-to-Sequence models (RNNs, LSTMs, GRUs) process data sequentially, which prevents parallelization and struggles with long-range dependencies in text.

**This project implements the Transformer architecture**, which relies entirely on **Self-Attention mechanisms** to compute representations of its input and output without using sequence-aligned RNNs.
* **Goal:** Translate English sentences into German (or other target languages).
* **Core Tech:** PyTorch, Spacy (Tokenization), TorchText (Data Pipeline).
* **Impact:** Solves the vanishing gradient problem in long sentences and allows for massive parallel training speeds.

---

## <a name="architecture"></a>🏗️ Model Architecture

The model follows the standard Encoder-Decoder structure but replaces recurrence with Multi-Head Attention.

```mermaid
graph LR
    subgraph Encoder
    A[Input Embedding] --> B[Positional Encoding]
    B --> C[Multi-Head Attention]
    C --> D[Feed Forward]
    end
    
    subgraph Decoder
    E[Output Embedding] --> F[Positional Encoding]
    F --> G[Masked Multi-Head Attention]
    G --> H[Cross Attention]
    H --> I[Feed Forward]
    end
    
    D --> H
    I --> J[Linear & Softmax]
    J --> K[Predicted Word]
