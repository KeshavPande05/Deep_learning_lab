# Implement the following functions in Python from scratch.
# Do not use any library functions.
# You are allowed to use numpy and matplotlib.
# Generate 100 equally spaced values between -10 and 10.
# Call this list as  z.
# Implement the following functions and its derivative.
# Use class notes to find the expression for these functions.
# Use z as input and plot both the function outputs
# and its derivative outputs.  Upload your code into Github and share it with me.
# Sigmoid
# Tanh
# ReLU (Rectified Linear Unit)
# Leaky ReLU
# Softmax (no need for visualization)

import numpy as np

import matplotlib.pyplot as plt



import numpy as mp
z = np.linspace(-10, 10,100)
print(z)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# tanh
def tanh(x):
    e_pos = np.exp(x)
    e_neg = np.exp(-x)

    return (e_pos - e_neg) / (e_neg + e_pos)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

def relu(x):
    y = np.zeros_like(x)

    for i in range(len(x)):
        if x[i] < 0:
            y[i] = 0
        else:
            y[i] = x[i]

    return y

def relu_derivative(x):
    y = np.zeros_like(x)

    for i in range(len(x)):
        if x[i] < 0:
            y[i] = 0
        else:
            y[i] = 1

    return y

def leaky_relu(x, alpha=0.01):
    y = np.zeros_like(x)

    for i in range(len(x)):
        if x[i] < 0:
            y[i] = alpha * x[i]
        else:
            y[i] = x[i]

    return y

def leaky_relu_derivative(x, alpha=0.01):
    y = np.zeros_like(x)

    for i in range(len(x)):
        if x[i] < 0:
            y[i] = alpha
        else:
            y[i] = 1

    return y

def softmax(x):
    exp_x = np.exp(x - np.max(x))      # Numerical stability
    return exp_x / np.sum(exp_x)

functions = [
    ("Sigmoid", sigmoid, sigmoid_derivative),
    ("Tanh", tanh, tanh_derivative),
    ("ReLU", relu, relu_derivative),
    ("Leaky ReLU", leaky_relu, leaky_relu_derivative),
]

plt.figure(figsize=(12, 10))

for i, (name, func, deriv) in enumerate(functions):

    plt.subplot(2, 2, i + 1)
    plt.plot(z, func(z), label=name, linewidth=2)
    plt.plot(z, deriv(z), label="Derivative", linestyle="--", linewidth=2)
    plt.title(name)
    plt.xlabel("z")
    plt.ylabel("Output")
    plt.grid(True)
    plt.legend()

plt.tight_layout()
plt.show()


softmax_output = softmax(z)

print("Softmax Output:")
print(softmax_output)
print("\nSum of Softmax =", np.sum(softmax_output))



