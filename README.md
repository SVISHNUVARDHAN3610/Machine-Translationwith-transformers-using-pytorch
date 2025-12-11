<div align="center">

# 🌐 Machine Translation: Hindi ↔️ English
### Transformer Implementation using PyTorch & Self-Attention

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Languages](https://img.shields.io/badge/Lang-Hindi%20%7C%20English-orange?style=for-the-badge)]()
[![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](LICENSE)

> **"Attention Is All You Need."**
> A complete PyTorch scratch implementation of the Transformer architecture designed for bidirectional translation between Hindi and English.

[Project Overview](#overview) • [Architecture](#architecture) • [Key Features](#features) • [Installation](#installation) • [Results](#results)

</div>

---

## <a name="overview"></a>🧐 Project Overview

Machine translation between languages with vastly different grammatical structures, like Hindi and English, presents significant challenges for traditional RNN-based models due to long-range dependencies.

**This project implements the Transformer architecture**, which relies entirely on **Self-Attention mechanisms** to handle these dependencies efficiently.
* **Goal:** Bidirectional translation (Hindi to English and English to Hindi).
* **Core Tech:** PyTorch, TorchText, and specialized Hindi/English tokenizers.
* **Impact:** Overcomes the sequential processing limitations of LSTMs, allowing for faster parallel training and better capture of context in complex Hindi sentences.

---

## <a name="architecture"></a>🏗️ Model Architecture

We adhere strictly to the encoder-decoder structure proposed in the seminal paper "Attention Is All You Need" (Vaswani et al., 2017).

<div align="center">
  <img src="https://raw.githubusercontent.com/VishnuVardhan/Machine-Translation-Transformer/main/assets/transformer_architecture.png" alt="Transformer Architecture Icon" width="600">
  <br>
  <em>Figure 1: The standard Transformer network architecture. (Source: Vaswani et al., 2017)</em>
</div>

### Component Breakdown
1.  **Embeddings & Positional Encoding:** Since Transformers have no recurrence, we inject sinusoidal positional information so the model understands word order (crucial for Hindi syntax).
2.  **Encoder (Left):** A stack of `N` identical layers. Each layer contains a Multi-Head Self-Attention mechanism followed by a Position-wise Feed-Forward Network.
3.  **Decoder (Right):** Similar to the encoder but includes a third sub-layer that performs Multi-Head Attention over the encoder's output.
4.  **Masking:**
    * **Padding Mask:** Ensures the model ignores padding tokens in batches.
    * **Look-Ahead Mask:** Prevents the decoder from "cheating" by peeking at future target words during training.

---

## <a name="features"></a>🌟 Key Features

* **⚡ Parallelized Training:** Unlike LSTMs, the Transformer processes entire sentences simultaneously.
* **🧠 Multi-Head Attention:** Allows the model to jointly attend to information from different representation subspaces at different positions—essential for aligning Hindi words to their English counterparts across the sentence.
* **🗣️ Bilingual Tokenization:** Handles complex Hindi script (Devanagari) and English subword tokenization effectively.
* **📊 BLEU Score Evaluation:** Integrated metric calculation to assess translation quality against reference texts.

---

## <a name="installation"></a>🚀 Getting Started

### Prerequisites
* Python 3.8+
* PyTorch 1.12+
* **NLP Tools:** `spacy` (for English), `indic-nlp-library` or similar tools for Hindi tokenization.

### Installation

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/VishnuVardhan/Machine-Translation-Transformer.git](https://github.com/VishnuVardhan/Machine-Translation-Transformer.git)
    cd Machine-Translation-Transformer
    ```

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    # Download necessary language models (adjust based on your specific tokenizers)
    python -m spacy download en_core_web_sm
    ```

3.  **Train the Model**
    ```bash
    # Example command indicating direction (e.g., hi-en or en-hi)
    python train.py --source_lang hi --target_lang en --epochs 20 --batch_size 32
    ```

### Inference Example

Translate a custom Hindi sentence to English using the trained model:

```python
from model import Transformer
from utils import translate_sentence

# Load Hindi-to-English Model
model = Transformer(...)
model.load_state_dict(torch.load('weights/transformer_hi_en.pt'))

sentence_hi = "मशीन लर्निंग एक बहुत ही रोचक क्षेत्र है।"
# Expected English: "Machine learning is a very interesting field."

translation = translate_sentence(sentence_hi, model, device)

print(f"Original: {sentence_hi}")
print(f"Translated: {translation}")
