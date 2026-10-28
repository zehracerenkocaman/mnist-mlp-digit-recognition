"""
MNIST + MLP (Deney 1 - Temel Model)
Çalıştırma: python train.py
"""
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# ---------------- Hiperparametreler ----------------
CONFIG = {
    "hidden_sizes": [256, 128],   # en az 2 hidden layer şartı
    "activation": "ReLU",
    "optimizer": "Adam",
    "learning_rate": 1e-3,
    "batch_size": 128,
    "epochs": 15,
    "val_size": 10_000,           # 60.000 -> 50.000 train + 10.000 validation
    "seed": 42,
}
MNIST_MEAN, MNIST_STD = 0.1307, 0.3081

RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

torch.manual_seed(CONFIG["seed"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Cihaz:", device)

# ---------------- Veri ----------------
transform = transforms.Compose([
    transforms.ToTensor(),                       
    transforms.Normalize(MNIST_MEAN, MNIST_STD),  # normalizasyon
])

train_full = datasets.MNIST("data", train=True, download=True, transform=transform)
test_set = datasets.MNIST("data", train=False, download=True, transform=transform)

n_val = CONFIG["val_size"]
train_set, val_set = random_split(
    train_full,
    [len(train_full) - n_val, n_val],
    generator=torch.Generator().manual_seed(CONFIG["seed"]),
)
print(f"Train: {len(train_set)} | Validation: {len(val_set)} | Test: {len(test_set)}")

bs = CONFIG["batch_size"]
train_loader = DataLoader(train_set, batch_size=bs, shuffle=True)
val_loader = DataLoader(val_set, batch_size=1000)
test_loader = DataLoader(test_set, batch_size=1000)


# ---------------- Model ----------------
def build_mlp(hidden_sizes):
    layers, in_features = [nn.Flatten()], 28 * 28
    for h in hidden_sizes:
        layers += [nn.Linear(in_features, h), nn.ReLU()]
        in_features = h
    layers.append(nn.Linear(in_features, 10))  # softmax, CrossEntropyLoss içinde
    return nn.Sequential(*layers)


model = build_mlp(CONFIG["hidden_sizes"]).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
print(model)


# ---------------- Yardımcı fonksiyonlar ----------------
def evaluate(loader):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            total_loss += criterion(out, y).item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            n += x.size(0)
    return total_loss / n, correct / n


# ---------------- Eğitim ----------------
history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
start = time.time()

for epoch in range(1, CONFIG["epochs"] + 1):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

    tr_loss, tr_acc = evaluate(train_loader)
    va_loss, va_acc = evaluate(val_loader)
    history["train_loss"].append(tr_loss)
    history["val_loss"].append(va_loss)
    history["train_acc"].append(tr_acc)
    history["val_acc"].append(va_acc)
    print(f"Epoch {epoch:2d}/{CONFIG['epochs']} | "
          f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
          f"val loss {va_loss:.4f} acc {va_acc:.4f}")

train_time = time.time() - start

# ---------------- Test ----------------
test_loss, test_acc = evaluate(test_loader)

print("\n===== SONUÇLAR =====")
print(f"Training loss       : {history['train_loss'][-1]:.4f}")
print(f"Validation loss     : {history['val_loss'][-1]:.4f}")
print(f"Training accuracy   : {history['train_acc'][-1]:.4f}")
print(f"Validation accuracy : {history['val_acc'][-1]:.4f}")
print(f"Test accuracy       : {test_acc:.4f}")
print(f"Eğitim süresi       : {train_time:.1f} sn")

# ---------------- Kaydetme ----------------
torch.save(model.state_dict(), RESULTS / "mlp_model.pt")

summary = {
    "hyperparameters": CONFIG,
    "final_train_loss": history["train_loss"][-1],
    "final_val_loss": history["val_loss"][-1],
    "final_train_acc": history["train_acc"][-1],
    "final_val_acc": history["val_acc"][-1],
    "test_loss": test_loss,
    "test_acc": test_acc,
    "train_time_sec": train_time,
}
with open(RESULTS / "deney1_sonuc.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

# ---------------- Grafikler ----------------
epochs = range(1, CONFIG["epochs"] + 1)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(epochs, history["train_loss"], label="Train")
ax[0].plot(epochs, history["val_loss"], label="Validation")
ax[0].set_title("Loss")
ax[0].set_xlabel("Epoch")
ax[0].legend()
ax[1].plot(epochs, history["train_acc"], label="Train")
ax[1].plot(epochs, history["val_acc"], label="Validation")
ax[1].set_title("Accuracy")
ax[1].set_xlabel("Epoch")
ax[1].legend()
plt.tight_layout()
plt.savefig(RESULTS / "deney1_egitim_grafikleri.png", dpi=150)
print("Grafik ve model results/ klasörüne kaydedildi.")
