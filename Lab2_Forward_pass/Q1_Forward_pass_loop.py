# Consider the following two networks. W is a matrix, x is a vector, z is a vector, and a is a vector.
# y^ is a scalar and a final prediction.
# Initialize x, w randomly, z is a dot product of x and w, a is ReLU(z).
# Initialize X and W randomly. Every neuron has a bias term.

import numpy as np
import math


# --- ACTIVATION FUNCTIONS ---
def relu(val):
    return max(0.0, val)


def sigmoid(val):
    return 1.0 / (1.0 + math.exp(-val))


# --- LAYER 1 (Hidden Layer 1) ---
a = int(input("Enter the number of neurons Initially (Input size): "))
b = int(input("Enter the number of Neurons in Next layer (Layer 1 size): "))

w = np.random.random((a, b))
b1 = np.random.random(b).tolist()  # One bias per neuron in Layer 1
print("Weights W:\n", w)
print("Biases B1:\n", b1)

x = np.random.random(a).tolist()
print("Input Vector x:\n", x)

y = []
for j in range(b):  # Loop through neurons in Layer 1
    z = 0
    for i in range(len(x)):  # Dot product calculation
        z += x[i] * w[i][j]

    # Add bias and apply ReLU activation
    activated_z = relu(z + b1[j])
    y.append(activated_z)
print("Layer 1 Output (y):\n", y)

# --- LAYER 2 (Hidden Layer 2) ---
z_len = len(y)
k = int(input("Enter the number of neurons in next layer (Layer 2 size): "))

e = np.random.random((z_len, k))
b2 = np.random.random(k).tolist()  # One bias per neuron in Layer 2
print("Weights E:\n", e)
print("Biases B2:\n", b2)

u = []
for j in range(k):  # Loop through neurons in Layer 2
    z = 0
    for i in range(len(y)):  # Dot product calculation
        z += y[i] * e[i][j]

    # Add bias and apply ReLU activation
    activated_u = relu(z + b2[j])
    u.append(activated_u)
print("Layer 2 Output (u):\n", u)

# --- LAYER 3 (Final Output Layer / y^) ---
u_len = len(u)
m = int(input("Enter the number of neurons in final layer (Layer 3 size, enter 1 for scalar y^): "))

g = np.random.random((u_len, m))
b3 = np.random.random(m).tolist()  # One bias per neuron in Layer 3
print("Weights G:\n", g)
print("Biases B3:\n", b3)

final_out = []
for j in range(m):  # Loop through neurons in Layer 3
    z = 0
    for i in range(len(u)):  # Dot product calculation
        z += u[i] * g[i][j]

    # Add bias and apply Sigmoid activation for the final prediction
    y_hat = sigmoid(z + b3[j])
    final_out.append(y_hat)
print("Layer 3 Output (final_out / y^):\n", final_out)
