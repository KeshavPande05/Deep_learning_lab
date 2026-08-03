import numpy as np

# ---------------- Input ----------------
x = [1, 2, 3, 4]

# -------- Hidden Layer 1 ---------------
w = [
    [0.1, 0.2, 0.3],
    [0.4, 0.5, 0.6],
    [0.7, 0.8, 0.9],
    [1.0, 1.1, 1.2]
]

y = []

for j in range(len(w[0])):
    z = 0
    for i in range(len(x)):
        z += x[i] * w[i][j]
    y.append(z)

print("Hidden Layer 1:", y)

# -------- Hidden Layer 2 ---------------
w2 = [
    [0.1, 0.2],
    [0.4, 0.5],
    [0.7, 0.8]
]

y2 = []

for j in range(len(w2[0])):
    z = 0
    for i in range(len(y)):
        z += y[i] * w2[i][j]
    y2.append(z)

print("Hidden Layer 2:", y2)

# -------- Output Layer -----------------
w3 = [[0.1, 0.2]]

y_final = np.dot(w3, y2)

print("Output:", y_final)