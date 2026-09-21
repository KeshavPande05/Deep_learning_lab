# Lab 6: Getting Started with PyTorch

This repository contains the introductory exercises for PyTorch, focusing on the fundamental data structures and tools required to build deep learning models. This lab serves as the transition from "from-scratch" implementations to using a professional industry-standard framework.

## 🎯 Learning Goals
The primary objective of this lab is to gain proficiency with the PyTorch ecosystem. By the end of this lab, I have learned how to:
1. Install and configure PyTorch for CPU-based computation.
2. Manipulate **Tensors**, the core data structure of PyTorch.
3. Handle data efficiently using **Datasets** and **DataLoaders**.
4. Use **Transforms** for data preprocessing.
5. Understand the mechanics of **Autograd** for automatic differentiation.
6. Implement model building, optimization, and the process of saving/loading trained models.

## 🛠️ Key Concepts Covered
- **Tensors:** Understanding n-dimensional arrays and their operations.
- **Autograd:** The engine that powers neural network training by calculating gradients.
- **Data Pipeline:** 
    - `Dataset`: How to store and access data samples.
    - `DataLoader`: How to wrap datasets into iterable batches.
    - `Transforms`: Applying transformations to data (e.g., normalization).
- **Model Lifecycle:** Building a class using `nn.Module`, defining an optimizer, and persisting the model using `torch.save()` and `torch.load()`.

## 🚀 Setup & Installation
To run the scripts in this lab, ensure you have PyTorch installed.

**Installation command (CPU version):**
```bash
pip install torch torchvision
