import numpy as np


def relu(x):
    return np.maximum(0, x)



# Exercise 1
# Single Layer Neural Network


# Input vector (4 features)
x = np.random.randn(4, 1)

# Weight matrix (1 neuron)
W = np.random.randn(1, 4)

# Bias
b = np.random.randn(1, 1)

print("\nInput x:")
print(x)

print("\nWeights W:")
print(W)

print("\nBias b:")
print(b)

# Forward Pass
z = np.dot(W, x) + b
a = relu(z)

y_hat = a

print("\nWeighted Sum (z):")
print(z)

print("\nActivation (a):")
print(a)

print("\nPrediction (ŷ):")
print(y_hat)



# Exercise 2
# Multi-Layer Neural Network
# Architecture:
# Input(4) -> Hidden1(3) -> Hidden2(2) -> Output(1)

print("\n\n")
print("=" * 60)
print("EXERCISE 2: MULTI-LAYER FORWARD PASS")
print("=" * 60)

# Input Layer
x = np.random.randn(4, 1)

# Hidden Layer 1
W1 = np.random.randn(3, 4)
b1 = np.random.randn(3, 1)

z1 = np.dot(W1, x) + b1
a1 = relu(z1)

# Hidden Layer 2
W2 = np.random.randn(2, 3)
b2 = np.random.randn(2, 1)

z2 = np.dot(W2, a1) + b2
a2 = relu(z2)

# Output Layer
W3 = np.random.randn(1, 2)
b3 = np.random.randn(1, 1)

z3 = np.dot(W3, a2) + b3
a3 = relu(z3)

y_hat = a3

print("\nInput")
print(x)

print("\n--------------- Hidden Layer 1 ---------------")
print("Weights:")
print(W1)

print("\nBias:")
print(b1)

print("\nWeighted Sum (z1):")
print(z1)

print("\nActivation (a1):")
print(a1)

print("\n--------------- Hidden Layer 2 ---------------")
print("Weights:")
print(W2)

print("\nBias:")
print(b2)

print("\nWeighted Sum (z2):")
print(z2)

print("\nActivation (a2):")
print(a2)

print("\n--------------- Output Layer ---------------")
print("Weights:")
print(W3)

print("\nBias:")
print(b3)

print("\nWeighted Sum (z3):")
print(z3)

print("\nActivation (a3):")
print(a3)

print("\nFinal Prediction (ŷ):")
print(y_hat)


# Exercise 3
# Fully Vectorized Implementation (No Loops)


print("\n\n")
print("=" * 60)
print("EXERCISE 3: VECTORIZED FORWARD PASS")
print("=" * 60)

# Batch of 5 samples
X = np.random.randn(4, 5)

# Weight Matrix
W = np.random.randn(3, 4)

# Bias Vector
b = np.random.randn(3, 1)

print("\nInput Matrix X:")
print(X)

print("\nWeight Matrix W:")
print(W)

print("\nBias Vector b:")
print(b)

# Vectorized Forward Pass
Z = np.dot(W, X) + b
A = relu(Z)

print("\nWeighted Sum Matrix (Z):")
print(Z)

print("\nActivation Matrix (A):")
print(A)