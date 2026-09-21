import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

# Device
device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)
print("Using:", device)

# Data
transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True)
])

data_path = "/home/ibab/Deep_learning_lab/data"

train_data = datasets.FashionMNIST(
    root=data_path,
    train=True,
    download=False,
    transform=transform
)

test_data = datasets.FashionMNIST(
    root=data_path,
    train=False,
    download=False,
    transform=transform
)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)


# Model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.net(self.flatten(x))


model = NeuralNetwork().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


# Training
for epoch in range(10):
    model.train()
    total_loss = 0

    for X, y in train_loader:
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(
        f"Epoch {epoch + 1}/10 - "
        f"Loss: {total_loss / len(train_loader):.4f}"
    )


# Testing
model.eval()
correct = 0

with torch.no_grad():
    for X, y in test_loader:
        X, y = X.to(device), y.to(device)

        pred = model(X)
        correct += (pred.argmax(1) == y).sum().item()

accuracy = 100 * correct / len(test_data)

print(f"Test Accuracy: {accuracy:.2f}%")
# ============================================================
# EXTRA CHECKS
# ============================================================

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

# Check dataset
print("\nDATASET CHECK")
print("Training samples:", len(train_data))
print("Test samples:", len(test_data))

X, y = next(iter(test_loader))
print("Image batch shape:", X.shape)
print("Label batch shape:", y.shape)
print("Image range:", X.min().item(), "to", X.max().item())
print("Labels:", y[:10].tolist())


# Check predictions
model.eval()

correct = 0
class_correct = [0] * 10
class_total = [0] * 10

with torch.no_grad():
    for X, y in test_loader:
        X, y = X.to(device), y.to(device)

        pred = model(X)
        predicted = pred.argmax(1)

        correct += (predicted == y).sum().item()

        for i in range(len(y)):
            label = y[i].item()
            class_total[label] += 1

            if predicted[i] == y[i]:
                class_correct[label] += 1


accuracy = 100 * correct / len(test_data)

print("\nACCURACY CHECK")
print(f"Overall accuracy: {accuracy:.2f}%")

for i in range(10):
    class_accuracy = 100 * class_correct[i] / class_total[i]
    print(
        f"{classes[i]:15s}: "
        f"{class_accuracy:6.2f}% "
        f"({class_correct[i]}/{class_total[i]})"
    )


# Sample predictions
print("\nSAMPLE PREDICTIONS")

images, labels = next(iter(test_loader))

images = images.to(device)

with torch.no_grad():
    predictions = model(images).argmax(1)

for i in range(10):
    print(
        f"{i+1}. "
        f"Predicted: {classes[predictions[i].item()]:15s} | "
        f"Actual: {classes[labels[i].item()]}"
    )