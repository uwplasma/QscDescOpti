import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from Data_preprocess import load_and_preprocess_data
from model import FullyConnectedModel, init_weights
from loss_function import CustomQscDescLoss

# Hyperparameters
input_dim = 10
output_dim = 10
hidden_dim = 256
batch_size = 32
n_epochs = 100
lr = 2e-4
patience = 15  
model_path = "best_model.pth"  

train_loader, val_loader, test_loader = load_and_preprocess_data('pass_desc.csv')

model = FullyConnectedModel(input_dim, output_dim)
# model.apply(init_weights)
optimizer = optim.Adam(model.parameters(), lr=lr)

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

model.to(device)
criterion = CustomQscDescLoss()

# Initialize lists to store loss values
train_losses = []
val_losses = []


best_val_loss = float('inf')
epochs_no_improve = 0

for epoch in range(n_epochs):
    model.train()
    train_loss = 0
    
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        X_updated = model(X_batch)

        loss = criterion(X_updated, y_batch, device)

        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    
    
    model.eval()
    val_loss = 0
    with torch.no_grad():  
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            X_updated = model(X_batch)
            loss = criterion(X_updated, y_batch, device)
            val_loss += loss.item()
    
    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    
    print(f'Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0  
        
        torch.save(model.state_dict(), model_path)
        print(f"Model saved with Val Loss: {val_loss:.4f}")
    else:
        epochs_no_improve += 1
    
    if epochs_no_improve == patience:
        print(f"Early stopping at epoch {epoch+1}")
        break

# Plot train and validation loss
plt.figure(figsize=(10,6))
plt.plot(train_losses, label="Train Loss", color='blue')
plt.plot(val_losses, label="Validation Loss", color='orange')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Train and Validation Loss over Epochs")
plt.grid(True)
plt.show()
