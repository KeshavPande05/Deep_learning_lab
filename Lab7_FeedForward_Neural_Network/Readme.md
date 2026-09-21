# Lab 7: Feedforward Network using CPU and GPU

This project implements an end-to-end Feedforward Neural Network classifier using PyTorch. The focus of this lab is to move from basic syntax to building, training, and tuning a deep learning model while managing hardware acceleration (CPU vs. GPU).

## 🎯 Learning Goals
The goal of this lab is to apply PyTorch knowledge to build a complete deep learning pipeline. Key outcomes include:
1. Designing and building a **Deep Neural Network (DNN)** classifier.
2. Implementing the full training loop (Forward pass $\rightarrow$ Loss Calculation $\rightarrow$ Backward pass $\rightarrow$ Optimizer step).
3. Managing hardware resources by moving models and tensors between **CPU** and **GPU (CUDA)**.
4. Understanding training dynamics, including **batch sizes**, **learning rates**, and **stopping criteria** (preventing overfitting).

## 🏗️ Model Architecture
The implemented model is a Feedforward Neural Network (Multi-Layer Perceptron) consisting of:
- **Input Layer:** Adapted to the dimensions of the input dataset.
- **Hidden Layers:** Multiple fully connected (`nn.Linear`) layers with non-linear activation functions (e.g., ReLU).
- **Output Layer:** A final linear layer mapping to the number of target classes.

## ⚙️ Training Details
- **Loss Function:** (e.g., CrossEntropyLoss for classification).
- **Optimizer:** (e.g., SGD or Adam).
- **Hardware:** The code includes logic to automatically detect and use a GPU if available:
  ```python
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model.to(device)
