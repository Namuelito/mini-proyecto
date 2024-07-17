import torch
from dataset import herramientas
from torch.utils.data import DataLoader
from red import mynn
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Cargar el dataset
dataset_path = os.path.dirname(__file__) + "/dataset"
Herramientas = herramientas(dataset_path)

# Dividir el dataset en entrenamiento y validación
train_size = int(0.8 * len(Herramientas))
valid_size = len(Herramientas) - train_size
train_dataset, valid_dataset = torch.utils.data.random_split(Herramientas, [train_size, valid_size])

# DataLoaders para entrenamiento y validación
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=128, shuffle=False)

# Definir el modelo, criterio y optimizador
model = mynn().cuda()
criterio = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Inicializar variables para seguimiento de pérdidas
max_epochs, best_valid_loss = 200, np.inf
running_loss = np.zeros(shape=(max_epochs, 2))

def train_one_epoch(epoch):
    model.train()
    train_loss = 0.0
    for image, label in train_dataloader:
        image = image.cuda()
        label = label.cuda()
        optimizer.zero_grad()

        with torch.autocast(device_type="cuda"): 
            output = model(image)
            loss = criterio(output, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_dataloader)
    return train_loss

def validate_one_epoch(epoch):
    model.eval()
    valid_loss = 0.0
    with torch.no_grad():
        for image, label in valid_dataloader:
            image = image.cuda()
            label = label.cuda()
            output = model(image)
            loss = criterio(output, label)
            valid_loss += loss.item()
    
    valid_loss /= len(valid_dataloader)
    return valid_loss

# Entrenamiento y validación del modelo
for epoch in tqdm(range(max_epochs)):
    train_loss = train_one_epoch(epoch)
    valid_loss = validate_one_epoch(epoch)
    running_loss[epoch] = [train_loss, valid_loss]

    print(f"Epoch {epoch+1}/{max_epochs} - Training Loss: {train_loss:.4f}, Validation Loss: {valid_loss:.4f}")

    # Guardar los pesos del modelo si el modelo mejora en validación
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.cpu().state_dict(), "pesos2.pt")
        model = model.cuda()

    # Graficar las pérdidas
    fig, ax = plt.subplots(figsize=(7, 4), tight_layout=True)
    ax.plot(running_loss[:, 0], label='Entrenamiento')
    ax.plot(running_loss[:, 1], label='Validación')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    plt.savefig('lossvsepoch.png')
    plt.show()