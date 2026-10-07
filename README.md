# Seq2Seq — Sequence-to-Sequence Implementation

A from-scratch implementation of the **Sequence-to-Sequence (Seq2Seq) architecture** using **PyTorch**.

The project demonstrates how an encoder-decoder architecture can be used for sequence transformation tasks such as **machine translation, text generation, and sequence prediction**.

## 📌 Overview

A Seq2Seq model consists of two main components:

- **Encoder** — Processes the input sequence and converts it into a hidden representation.
- **Decoder** — Uses the encoder's representation to generate the target sequence one token at a time.

```text
Input Sequence
      │
      ▼
┌─────────────┐
│   Encoder   │
└─────────────┘
      │
      │ Context / Hidden State
      ▼
┌─────────────┐
│   Decoder   │
└─────────────┘
      │
      ▼
Output Sequence
```

## 🧠 Architecture

The implementation follows the standard encoder-decoder architecture.

### Encoder

The encoder:

1. Receives token IDs as input.
2. Converts tokens into embeddings.
3. Processes the embeddings using an RNN/LSTM/GRU.
4. Passes the final hidden state to the decoder.

```text
Token IDs
   ↓
Embedding
   ↓
RNN / LSTM / GRU
   ↓
Hidden State
```

### Decoder

The decoder generates the output sequence autoregressively.

At every timestep, it:

1. Takes the previous target token.
2. Converts it into an embedding.
3. Uses the previous hidden state.
4. Predicts the next token.

```text
Previous Token
      ↓
  Embedding
      ↓
RNN / LSTM / GRU
      ↓
Linear Layer
      ↓
Softmax
      ↓
Next Token
```

## 📂 Project Structure

```text
seq2seq/
│
├── encoder.py          # Encoder implementation
├── decoder.py          # Decoder implementation
├── seq2seq.py          # Seq2Seq model
├── train.py            # Training loop
├── inference.py        # Model inference
├── utils.py            # Utility functions
├── requirements.txt    # Dependencies
└── README.md
```

## ⚙️ Requirements

- Python 3.9+
- PyTorch
- NumPy

Install the dependencies:

```bash
pip install -r requirements.txt
```

## 🚀 Usage

Clone the repository:

```bash
git clone <your-repository-url>
cd <repository-name>
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python train.py
```

Run inference:

```bash
python inference.py
```

## 🔤 Tokenization

The model operates on **token IDs** rather than raw text.

For example:

```text
"I love AI"
```

can be converted into:

```text
[12, 45, 89]
```

These token IDs are then passed through an embedding layer:

```text
Token ID
   ↓
Embedding Layer
   ↓
Dense Vector
```

Special tokens such as the following can be used:

```text
<SOS>  Start of Sequence
<EOS>  End of Sequence
<PAD>  Padding
<UNK>  Unknown Token
```

## 🔄 Training

During training, the decoder can use **teacher forcing**.

Instead of always feeding the decoder's previous prediction back as the next input, the actual target token is provided.

```text
Target:      <
