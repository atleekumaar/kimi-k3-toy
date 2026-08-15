# Kimi K3 Mini: Toy Architecture Implementation

A minimal, educational PyTorch implementation of the **Kimi K3** architecture ([Kimi Team, "Kimi K3: Open Frontier Intelligence"](https://www.k-a.in/KDA.html)), scaled down to run smoothly on standard CPUs or Google Colab.

---

## 🏛️ Architecture Overview & ASCII Diagram

The Kimi K3 architecture integrates advanced token mixing, specialized attention mechanisms, and sparse/dense mixture-of-experts (MoE) routing. Below is the structural layout of this toy implementation:

```text
Input Tokens (B, T, D)
       │
       ▼
┌──────────────────────────────────────────────┐
│  Embedding & Attention Residuals (AttnRes)   │
│  - Uses learned pseudo-queries (Eq. 8–9)     │
└──────────────────────┬───────────────────────┘
       │
       ▼  (3:1 Ratio Layout per Block)
┌──────────────────────────────────────────────┐
│  Sub-Block Sequence (KDA & Gated MLA)        │
│  ├── KDA Layer 1 (Causal Conv + Recurrence)  │
│  ├── KDA Layer 2 (Sequential Recurrent)      │
│  ├── KDA Layer 3 (Sequential Recurrent)      │
│  └── Gated MLA   (NoPE + Output Gate)        │
└──────────────────────┬───────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────┐
│  Stable LatentMoE Feed-Forward Network       │
│  ├── Shared Experts (Full-width SwiGLU)      │
│  └── Routed Experts (Compact SiTU-GLU)       │
└──────────────────────┬───────────────────────┘
       │
       ▼
Final RMSNorm & LM Head -> Next Token Logits
```

---

## 🚀 Key Architectural Features

This implementation includes deliberate simplifications to remain readable while following the core concepts of the paper:

* **Kimi Delta Attention (KDA):** Implemented via sequential recurrence (Eq. 1–6) combined with causal short convolutions.
* **Gated MLA:** Standard Multi-Head Attention featuring NoPE (No Positional Encoding) and an explicit output gate (§2.1.2).
* **Attention Residuals (AttnRes):** Full form implementation utilizing learned pseudo-queries to compute attention weights over history (§2.2).
* **Stable LatentMoE:** Shared and routed experts featuring SiTU-GLU and pre-up-projection RMSNorm (§2.3).

---

## 🛠️ Requirements & Installation

Make sure you have **PyTorch** installed in your Python environment:

```bash
pip install torch
```

---

## 💻 Usage

Run the Python script directly from your terminal or paste it into a Google Colab notebook cell:

```bash
python model.py
```

### What happens during execution:

1. **Model Building:** Initializes the `KimiK3Mini` model with toy-scale configuration parameters.
2. **Sanity Check:** Performs a forward and backward pass on synthetic text data to verify proper gradient flow and check for NaN values.
3. **Training Loop:** Trains the model for 300 steps on a periodic character sequence (`0123456789ABCDEF`) to demonstrate loss convergence.
4. **Generation:** Generates a sequence from the trained model to show that it successfully captures the underlying periodicity.

---

