# HW3 - Sentiment Analysis via Deep Neural Networks

# Exercise 1: DNN Models for Sequential Data

---

## Table of Contents
1. [Introduction](#introduction)
2. [Requirements](#requirements)
3. [Directory Structure](#directory-structure)
4. [Setup](#setup)
5. [Exercise 1.1: Recurrent Neural Network (RNN)](#exercise-1.1-recurrent-neural-network-rnn)
6. [Exercise 1.2: Long Short-Term Memory (LSTM)](#exercise-1.2-long-short-term-memory-lstm)
7. [Results](#results)
8. [References](#references)

---

### Introduction

This project implements various deep neural network models designed for processing sequential data as part of a sentiment analysis task using movie review data from the IMDb dataset. The models include:

1. Recurrent Neural Network (RNN)
2. Long Short-Term Memory (LSTM)
3. Multi-head Attention (Transformer)

The implementations are compared with equivalent modules from PyTorch to evaluate their performance and accuracy.

---

### Requirements

- Python 3.x
- Jupyter Notebook
- NumPy
- Matplotlib
- PyTorch

---

### Directory Structure

```
.
├── data
│   └── IMDb dataset files
├── notebooks
│   └── exercise_1.ipynb
├── src
│   ├── HW_YourAnswer_modules.py
│   ├── HW_YourAnswer_encoders.py
│   ├── utils.py
│   └── module_check.py
└── README.md
```

---

### Setup

1. **Clone the repository:**
   ```sh
   git clone <repository_url>
   cd <repository_directory>
   ```

2. **Run the Jupyter Notebook:**
   ```sh
   jupyter notebook
   ```

3. **Open and execute the `exercise_1.ipynb` notebook:**

---

### Exercise 1.1: Recurrent Neural Network (RNN)

#### 1.1.1 One-Layer RNN

The `OneLayerRNN` class computes the hidden state for each token in an input sequence. The implementation involves the use of `nn.Linear` modules for linear projections and bias addition.

**Steps:**
1. Implement `OneLayerRNN` in `HW_YourAnswer_modules.py`.
2. Return both the final hidden state and all hidden states for the input sequence.

**Comparison:**
The custom `OneLayerRNN` is compared against `torch.nn.RNN` using forward and backward passes, and the Mean Squared Error (MSE) is computed to validate the implementation.

#### 1.1.2 Multi-Layer RNN

The `MultiLayerRNN` class builds upon the `OneLayerRNN` to implement multi-layer RNN operations.

**Steps:**
1. Implement `MultiLayerRNN` in `HW_YourAnswer_modules.py`.
2. Use `nn.Sequential` to manage multiple layers effectively.

**Comparison:**
Forward and backward passes of the custom `MultiLayerRNN` are compared with `torch.nn.RNN` to ensure accuracy.

---

### Exercise 1.2: Long Short-Term Memory (LSTM)

#### 1.2.1 One-Layer LSTM

The `OneLayerLSTM` class computes hidden states similarly to RNN but includes additional gates for input, forget, cell, and output.

**Steps:**
1. Implement `OneLayerLSTM` in `HW_YourAnswer_modules.py`.
2. Perform linear projections in parallel and divide outputs appropriately.

**Comparison:**
The custom `OneLayerLSTM` is compared against `torch.nn.LSTM` using forward and backward passes, and MSE is computed to validate the implementation.

#### 1.2.2 Multi-Layer LSTM

The `MultiLayerLSTM` builds upon the `OneLayerLSTM` to implement multi-layer LSTM operations.

**Steps:**
1. Implement `MultiLayerLSTM` in `HW_YourAnswer_modules.py`.
2. Use `nn.Sequential` to manage multiple layers effectively.

**Comparison:**
Forward and backward passes of the custom `MultiLayerLSTM` are compared with `torch.nn.LSTM` to ensure accuracy.

---

### Results

The implementation results, including forward and backward MSE, are logged in the notebook for each model:

- **One-Layer RNN:**
  - Forward MSE for output layer: \(1.38455 \times 10^{-14}\)
  - Backward MSE: \(1.50610 \times 10^{-15}\)

- **Multi-Layer RNN:**
  - Forward MSE for output layer: \(9.45844 \times 10^{-15}\)
  - Backward MSE: \(5.75454 \times 10^{-15}\)

- **One-Layer LSTM:**
  - Forward MSE for output layer: \(1.19844 \times 10^{-15}\)
  - Backward MSE: \(6.45085 \times 10^{-16}\)

- **Multi-Layer LSTM:**
  - Forward MSE for output layer: \(1.13324 \times 10^{-14}\)
  - Backward MSE: \(5.75454 \times 10^{-15}\)

---

### References

1. [PyTorch RNN Documentation](https://pytorch.org/docs/stable/generated/torch.nn.RNN.html)
2. [PyTorch LSTM Documentation](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)

---

Please refer to the notebook `exercise_1.ipynb` for detailed implementation steps, code execution, and results visualization.


# Exercise 2: Multi-Head Attention in Transformers

## Overview

This repository contains the implementation for Exercise 2 of the Deep Learning course. The exercise focuses on implementing key components of the Transformer architecture, specifically the Multi-Head Attention mechanism and its integration within the Transformer Encoder. The main objectives of this exercise are to understand the intricacies of the attention mechanism and to implement it using PyTorch.

## Files and Structure

- `HW_YourAnswer_encoders.py`: Contains the implementation of the Transformer Encoder, including the positional encoding and the integration of multi-head attention.
- `HW_YourAnswer_modules.py`: Contains the implementation of the Multi-Head Attention mechanism.
- `utils.py`: Utility functions for data loading, training, and evaluation.
- `notebook.ipynb`: Jupyter Notebook to run and test the implementations.

## Key Components

### 1. Transformer Encoder

The Transformer Encoder is implemented in the `TransformerEncoder` class within `HW_YourAnswer_encoders.py`. The key components include:

- **Positional Encoding**: This component provides the model with information about the positions of elements in the input sequence, as Transformers process inputs in parallel and do not inherently capture sequential order.
  
  ```python
  def pos_enc(self, seq_len, d_model):
      pos = torch.arange(0, seq_len).unsqueeze(1)
      i = torch.arange(0, d_model//2).unsqueeze(0)
      angle_rates = 1 / torch.pow(10000, (2 * (i // 2)) / torch.tensor(d_model, dtype=torch.float32))
      angle_rads = pos * angle_rates
      pos_encoding = torch.zeros(seq_len, d_model)
      pos_encoding[:, 0::2] = torch.sin(angle_rads)
      pos_encoding[:, 1::2] = torch.cos(angle_rads)
      return pos_encoding
  ```

- **Multi-Head Attention Integration**: The multi-head attention mechanism is integrated within the encoder to allow the model to focus on different parts of the input sequence simultaneously.

### 2. Multi-Head Attention

The Multi-Head Attention mechanism is implemented in the `MultiheadAttention` class within `HW_YourAnswer_modules.py`. The key steps include:

- **Linear Projections**: Query, Key, and Value vectors are linearly projected.
  
  ```python
  self.q_proj = nn.Linear(d_model, d_model)
  self.k_proj = nn.Linear(d_model, d_model)
  self.v_proj = nn.Linear(d_model, d_model)
  ```

- **Scaled Dot-Product Attention**: Computes the attention weights and applies them to the Value vectors.
  
  ```python
  def scaled_dot_product_attention(self, Q, K, V, mask=None):
      matmul_qk = torch.matmul(Q, K.transpose(-2, -1))
      dk = K.size()[-1]
      scaled_attention_logits = matmul_qk / math.sqrt(dk)
      if mask is not None:
          scaled_attention_logits += (mask * -1e9)
      attention_weights = torch.nn.functional.softmax(scaled_attention_logits, dim=-1)
      output = torch.matmul(attention_weights, V)
      return output, attention_weights
  ```

- **Concatenation and Final Linear Layer**: The outputs from the multiple heads are concatenated and passed through a final linear layer.

### 3. Training and Evaluation

The training script is implemented in the `notebook.ipynb`. It includes:

- **Data Loading**: Using the IMDb dataset for training and evaluation.
- **Training Loop**: Training the Transformer Encoder on the dataset and evaluating its performance.

  ```python
  loaders = imdb.make_small_loaders(pad_first=False)
  embedding_dim, hidden_dim = 256, 256
  transformer = TransformerEncoder(num_tokens=num_tokens, embedding_dim=embedding_dim, hidden_dim=hidden_dim, num_layers=1).to(DEVICE)
  train(transformer, loaders, epochs=50, lr=1e-4, use_mask=True, small_loader=True)
  ```

## How to Run

1. **Install Dependencies**: Ensure you have PyTorch and other required libraries installed.

    ```bash
    pip install torch torchvision
    ```

2. **Run the Jupyter Notebook**: Open `notebook.ipynb` and run the cells to train and evaluate the model.

3. **Check Results**: The notebook includes code to visualize the positional encodings and the performance metrics of the trained model.

## References

- Vaswani, A., et al. "Attention is All you Need." Advances in Neural Information Processing Systems, 2017.
- PyTorch Documentation: [MultiheadAttention](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)

---

This readme file provides a comprehensive guide to understanding and running the implementation for Exercise 2. For detailed instructions and code, refer to the files mentioned above and the Jupyter Notebook.
