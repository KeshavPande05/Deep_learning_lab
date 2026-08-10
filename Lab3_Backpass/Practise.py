import numpy as np

-
def sigmoid(x):
    x_clipped = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x_clipped))

def sigmoid_derivative(s):
    # s is already sigmoid(x)
    return s * (1.0 - s)

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=float)

y = np.array([
    [0],
    [1],
    [1],
    [0]
], dtype=float)

input_dim = 2
hidden_dim = 12
output_dim = 1

W1 = np.random.uniform(-1, 1, (input_dim, hidden_dim))
b1 = np.zeros((1, hidden_dim))

W2 = np.random.uniform(-1, 1, (hidden_dim, output_dim))
b2 = np.zeros((1, output_dim))

learning_rate = 0.5
epochs = 100000
convergence_threshold = 0.00001

m = X.shape[0]

# -------------------------------------------------------------
# Training Loop
# -------------------------------------------------------------
for epoch in range(epochs):

    # ---------- FORWARD PASS ----------
    hidden_in = X @ W1 + b1
    hidden_out = sigmoid(hidden_in)

    output_in = hidden_out @ W2 + b2
    predicted_output = sigmoid(output_in)

    # ---------- LOSS COMPUTATION ----------
    # Mean Squared Error: MSE = (1 / 2m) * sum((pred - y)^2)
    loss = np.mean(0.5 * (predicted_output - y) ** 2)

    # Convergence Check
    if loss < convergence_threshold:
        print(f"Converged successfully at epoch {epoch} with loss: {loss:.6f}")
        break

    # ---------- BACKPROPAGATION ----------
    # Derivative of MSE loss w.r.t prediction
    error_output = predicted_output - y

    # Output layer delta
    d_output = error_output * sigmoid_derivative(predicted_output)

    # Hidden layer error & delta
    error_hidden = d_output @ W2.T
    d_hidden = error_hidden * sigmoid_derivative(hidden_out)

    # ---------- GRADIENTS ----------
    dW2 = (hidden_out.T @ d_output) / m
    db2 = np.sum(d_output, axis=0, keepdims=True) / m

    dW1 = (X.T @ d_hidden) / m
    db1 = np.sum(d_hidden, axis=0, keepdims=True) / m

    # ---------- WEIGHT UPDATE ----------
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    if epoch % 5000 == 0:
        print(f"Epoch {epoch:6d} | Loss: {loss:.6f}")

else:
    print(f"Failed to converge within {epochs} epochs. Final Loss: {loss:.6f}")

# -------------------------------------------------------------
# Final Forward Pass & Output
# -------------------------------------------------------------
hidden_out = sigmoid(X @ W1 + b1)
predicted_output = sigmoid(hidden_out @ W2 + b2)

print("\nPredictions after training:")
for i in range(len(X)):
    print(
        f"Input: {X[i].astype(int)} "
        f"-> Predicted: {predicted_output[i, 0]:.4f} "
        f"(Target: {int(y[i, 0])})"
    )