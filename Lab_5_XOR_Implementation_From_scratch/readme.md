# Lab 5: XOR Implementation from Scratch

## Overview
This repository contains the implementation of a 2-layer Neural Network designed to solve the XOR (Exclusive OR) logic problem. The primary constraint of this lab is to build the model **from scratch** without using any deep learning libraries (such as TensorFlow, PyTorch, or Keras).

The goal is to understand the mathematical foundations of neural networks, specifically the forward propagation and backpropagation algorithms.

## Learning Goals
By completing this lab, you will:
1.  **Build a working neural network model** capable of implementing the XOR operation.
2.  **Understand the training process**, specifically how a neural network approximates non-linear functions through weight adjustments.
3.  **Implement core algorithms** including the Forward Pass and Backward Pass (Backpropagation) manually.

## The Problem: XOR
The XOR problem is a classic example of a problem that is not linearly separable. A single-layer perceptron cannot solve it; a hidden layer is required to map the inputs to the correct outputs.

**Truth Table:**

| Input 1 (x1) | Input 2 (x2) | Output (y) |
| :---: | :---: | :---: |
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

