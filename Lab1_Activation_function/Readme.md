# Lab 01 – Activation Functions

## 🎯 Learning Goals

In this lab, we explore different activation functions used to introduce **non-linearity** in neural networks. Activation functions enable deep neural networks to learn complex relationships from data by transforming the output of neurons.

By the end of this lab, you will be able to:

- Understand the purpose of activation functions in deep learning.
- Learn the mathematical formulation of common activation functions.
- Compare the advantages and disadvantages of different activation functions.
- Implement activation functions and their derivatives from scratch using Python.
- Visualize activation functions and analyze their behavior.
- Understand how activation functions affect gradient flow during backpropagation.

---

## 📝 Exercises

### Task 1: Generate Input Values

Generate **100 equally spaced values** between **-10 and 10** and store them in a variable named `z`.

---

### Task 2: Implement Activation Functions

Implement the following activation functions **from scratch**.

> **Note:** Do not use built-in activation functions from any deep learning library. You may use **NumPy** for numerical computations and **Matplotlib** for plotting.

Implement both the function and its derivative for:

- Sigmoid
- Tanh
- ReLU (Rectified Linear Unit)
- Leaky ReLU
- Softmax *(Derivative implementation only; visualization is not required.)*

---

### Task 3: Visualization

Using `z` as the input:

- Plot each activation function.
- Plot its derivative on a separate graph.
- Label the axes and provide appropriate titles.

---

### Task 4: Observations

For each activation function, answer the following questions based on the generated plots.

1. What are the minimum and maximum output values?
2. Is the function output zero-centered?
3. What happens to the gradient when the input values become very small or very large?
4. What are the advantages and disadvantages of this activation function?

Finally, explain:

- What is the relationship between the **Sigmoid** and **Tanh** activation functions?

---

## 📌 Expected Learning Outcomes

After completing this lab, you should be able to:

- Implement common activation functions without using deep learning libraries.
- Understand the mathematical intuition behind each activation function.
- Compare activation functions based on their output range and gradient behavior.
- Identify issues such as vanishing gradients and dying ReLU.
- Choose appropriate activation functions for different deep learning tasks.

---

## 🛠️ Technologies Used

- Python 3
- NumPy
- Matplotlib

---

## 📚 Concepts Covered

- Non-linearity in Neural Networks
- Activation Functions
- Sigmoid Function
- Hyperbolic Tangent (Tanh)
- Rectified Linear Unit (ReLU)
- Leaky ReLU
- Softmax Function
- Derivatives of Activation Functions
- Vanishing Gradient Problem
- Zero-Centered Activations

---

## 📁 Files

```
Lab01_Activation_Functions/
│
├── activation_functions.py
├── plot(combined)
│   ├── sigmoid.png
│   ├── tanh.png
│   ├── relu.png
│   └── leaky_relu.png
│
└── README.md
```

---

## 📖 Submission

- Implement all activation functions from scratch.
- Generate plots for each activation function and its derivative.
- Include observations in the source code as comments or in a separate markdown section.
- Upload the completed lab to your GitHub repository.

---

## 👨‍💻 Author

**Keshav Pande**

M.Sc. Big Data Biology  
Institute of Bioinformatics and Applied Biotechnology (IBAB)
