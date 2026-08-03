# Lab 02 – Forward Pass in Neural Networks

## 🎯 Learning Goals

In this lab, we will study the **forward pass** mechanism in a neural network. The forward pass is the process through which input data flows through the network to generate predictions.

By the end of this lab, you should be able to:

- Understand how input data flows through a neural network.
- Learn how neurons transform input data using weights, biases, and activation functions.
- Understand matrix and vector representations used in neural networks.
- Become familiar with matrix multiplication and dot product operations.
- Understand how activation functions introduce non-linearity into neural networks.
- Implement the forward pass of a neural network from scratch using **NumPy**.

---

## 📝 Exercises

### Exercise 1: Single-Layer Neural Network

Consider the following neural network:

- **W** is a weight matrix.
- **x** is an input vector.
- **z** is the weighted sum obtained using the dot product.
- **a** is the activated output after applying the ReLU function.
- **ŷ** is the final prediction (scalar).

Initialize:

- `x` randomly
- `W` randomly
- Bias values randomly

Compute the following:

```
z = W · x + b
a = ReLU(z)
ŷ = a
```

Print all intermediate values.

---

### Exercise 2: Multi-Layer Neural Network

Implement the forward pass for the following deep neural network:

Input Layer → Hidden Layer 1 → Hidden Layer 2 → Output Layer

For every layer:

1. Initialize weights randomly.
2. Initialize biases randomly.
3. Compute

```
z = W · x + b
a = ReLU(z)
```

4. Use the activation of the previous layer as the input to the next layer.

Print the following for every layer:

- Input
- Weights
- Biases
- Weighted Sum (z)
- Activation (a)

Finally print the predicted output **ŷ**.

---

### Exercise 3: Vectorized Forward Pass

Implement the forward pass using **vectorized operations only**.

Given:

- `W` → Weight matrix
- `x` → Input vector
- `z` → Weighted sum vector
- `a` → Activation vector

Requirements:

- Use **NumPy matrix operations**
- Do **not** use any loops (`for` or `while`)
- Compute the entire forward pass using vectorized expressions.

---

## 📌 Expected Learning Outcomes

After completing this lab, you should be able to:

- Understand the complete forward propagation process.
- Perform matrix multiplication for neural networks.
- Implement single-layer and multi-layer neural networks.
- Apply activation functions after every layer.
- Differentiate between vectors, matrices, and tensors.
- Write efficient vectorized implementations without loops.

---

## 🛠️ Technologies Used

- Python 3
- NumPy
- Matplotlib *(optional for visualization)*

---

## 📚 Concepts Covered

- Forward Propagation
- Artificial Neural Networks (ANN)
- Matrix Multiplication
- Dot Product
- Weight Initialization
- Bias Initialization
- ReLU Activation Function
- Hidden Layers
- Output Layer
- Vectorization using NumPy

---

## 📁 Files

```
Lab02_Forward_Pass/
│
├── forward_pass.py
├── images/
│   ├── single_layer_network.png
│   └── multi_layer_network.png
│
└── README.md
```

---

## 📖 Submission

- Implement the forward pass for both neural network architectures.
- Print the intermediate computations (`z`, `a`, and `ŷ`) for each layer.
- Implement the vectorized version without using loops.
- Upload the completed lab to your GitHub repository.

---

## 👨‍💻 Author

**Keshav Pande**

M.Sc. Big Data Biology  
Institute of Bioinformatics and Applied Biotechnology (IBAB)
