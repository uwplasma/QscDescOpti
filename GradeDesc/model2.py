import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load data
data = pd.read_csv("updated_data.csv")

# deal with y, normlize y based on min and max value.
# data['iota_diff'] = np.log1p(data['iota_diff']-data['iota_diff'].min())
data['iota_diff'] = np.log1p(data['iota_diff'].abs())
data['beta_diff'] = data['beta_diff'].abs()


y = data[['iota_diff', 'L_grad_B_min_diff', 'beta_diff']].values
y_means = y.mean(axis=0)
y_stds = y.std(axis=0)
y = (y - y_means) / y_stds

# normalize x based on their range on table provided. 
data['rc1_norm'] = (data['rc1'] + 1) / 2
data['rc2_norm'] = (data['rc2'] + 1) / 2
data['rc3_norm'] = (data['rc3'] + 1) / 2

data['zs1_norm'] = (data['zs1'] + 1) / 2
data['zs2_norm'] = (data['zs2'] + 1) / 2
data['zs3_norm'] = (data['zs3'] + 1) / 2
data['etabar_norm'] = (data['etabar'] +3) / 6
data['B2c_norm'] = (data['B2c'] + 3) / 6
data['nfp_norm'] = data['nfp'] / 10
data['p2_norm'] = (data['p2'] + 4e6) / 4e6


X = data[['rc1_norm', 'rc2_norm', 'rc3_norm', 'zs1_norm', 'zs2_norm', 'zs3_norm', 'etabar_norm', 'B2c_norm', 'nfp_norm', 'p2_norm']].values
# Convert data to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Split data
X_train, X_temp, y_train, y_temp = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# DataLoader setup
batch_size = 64
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # If in_channels != out_channels, use a 1x1 convolution to match dimensions
        self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += identity  # Add the skip connection
        out = torch.relu(out)
        return out

class ResNetRegressor(nn.Module):
    def __init__(self, output_size = 3):
        super(ResNetRegressor, self).__init__()
        # Initial Conv layer
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        
        # Stacking more residual blocks for increased depth
        self.layer1 = ResidualBlock(16, 32)
        self.layer2 = ResidualBlock(32, 64)
        self.layer3 = ResidualBlock(64, 128)
        self.layer4 = ResidualBlock(128, 128)
        self.layer5 = ResidualBlock(128, 128)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layers for regression
        self.fc1 = nn.Linear(128 , 128)  
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, output_size)  # Output layer for regression

    def forward(self, x):
        # Initial conv layer
        x = x.unsqueeze(1) 
        x = torch.relu(self.bn1(self.conv1(x)))  # Shape: [batch_size, 16, 10]

        # Residual blocks
        x = self.layer1(x)  # Shape: [batch_size, 32, 10]
        x = self.layer2(x)  # Shape: [batch_size, 64, 10]
        x = self.layer3(x)  # Shape: [batch_size, 128, 10]
        x = self.layer4(x)  # Shape: [batch_size, 128, 10]
        x = self.layer5(x)  # Shape: [batch_size, 128, 10]
        
        # Flatten for fully connected layers
        x = self.global_avg_pool(x).view(x.size(0), -1)  # Shape becomes [batch_size, 128 * 10]
        
        # Fully connected layers
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)  
        
        return x

# Initialize model with custom weight initialization
model = ResNetRegressor().to(device)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training settings and early stopping
num_epochs = 30
patience = 30
best_val_loss = np.inf
epochs_no_improve = 0
train_epoch_losses, val_epoch_losses = [], []
train_batch_losses, val_batch_losses = [], []

for epoch in range(num_epochs):
    model.train()
    running_train_loss = 0.0
    batch_train_losses = []

    # Training loop with progress bar
    with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as tepoch:
        for X_batch, y_batch in tepoch:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions,y_batch)
            loss.backward()
            optimizer.step()

            # Track batch loss
            batch_train_losses.append(loss.item())
            train_batch_losses.append(loss.item())  # Store globally for entire training

            # iota_error = criterion(predictions[:,0],y_batch[:,0])
            # L_grad_B_min_error = criterion(predictions[:,1],y_batch[:,1])
            # beta_error = criterion(predictions[:,2],y_batch[:,2])

            # Update progress bar with the latest loss
            tepoch.set_postfix(loss=loss.item())

    # Compute and record the average training loss for the epoch
    train_epoch_loss = np.mean(batch_train_losses)
    train_epoch_losses.append(train_epoch_loss)
    
    

    # Validation step
    model.eval()
    batch_val_losses = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            predictions = model(X_batch)
            val_loss = criterion(predictions,y_batch)
            batch_val_losses.append(val_loss.item())
            val_batch_losses.append(val_loss.item())  

    # Compute and record the average validation loss for the epoch
    val_epoch_loss = np.mean(batch_val_losses)
    val_epoch_losses.append(val_epoch_loss)

    # Early stopping check
    if val_epoch_loss < best_val_loss:
        best_val_loss = val_epoch_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), "best_model2.pth")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("Early stopping triggered")
            break
    iota_error = criterion(predictions[:,0],y_batch[:,0])
    L_grad_B_min_error = criterion(predictions[:,1],y_batch[:,1])
    beta_error = criterion(predictions[:,2],y_batch[:,2])
    print(iota_error,L_grad_B_min_error,beta_error)
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_epoch_loss:.4f}, Val Loss: {val_epoch_loss:.4f}")


plt.figure(figsize=(14, 6))

# Plot Training Loss
plt.subplot(1, 2, 1)
plt.plot(train_batch_losses, label="Training Loss (per batch)", color='blue')
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.title("Training Loss per Batch")
plt.legend()

# Plot Validation Loss
plt.subplot(1, 2, 2)
plt.plot(val_batch_losses, label="Validation Loss (per batch)", color='orange')
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.title("Validation Loss per Batch")
plt.legend()

plt.suptitle("Batch-wise Training and Validation Loss")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to fit the main title
plt.savefig("model2_loss_batch.png")
plt.show()


plt.figure(figsize=(10, 5))
plt.plot(train_epoch_losses, label="Training Loss (per epoch)")
plt.plot(val_epoch_losses, label="Validation Loss (per epoch)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Epoch-wise Training and Validation Loss")
plt.savefig("model2_loss_epoch.png")
plt.show()