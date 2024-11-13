import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load and normalize data
# deal with y, normlize y based on min and max value.
data = pd.read_csv("dataset.csv")

# iota_min = data['iota_diff'].min()
# deal with y, normlize y based on min and max value.
# data['iota_diff'] = np.log1p(data['iota_diff']-data['iota_diff'].min())
# data['iota_diff'] = np.log1p(data['iota_diff'].abs())
# data['beta_diff'] = data['beta_diff'].abs()

y = data[['iota', 'min_L_grad_B']].values
y_means = y.mean(axis=0) # [0.0015223 0.0357987]
print(y_means)
y_stds = y.std(axis=0) # [8.68295316 0.10227327]
print(y_stds)
y = (y - y_means) / y_stds

# normalize x based on their range on table provided. 
data['rc1_norm'] = (data['rc1'] + 1) / 2
data['rc2_norm'] = np.where(
    data['rc1'] == 0,
    0,
    (data['rc2']/data['rc1']+1.)/2.
)
data['rc3_norm'] = np.where(
    data['rc2'] == 0,
    0,
    (data['rc3']/data['rc2']+1.)/2.
)
data['zs1_norm'] = (data['zs1'] + 1) / 2
data['zs2_norm'] = np.where(
    data['zs1'] == 0,
    0,
    (data['zs2']/data['zs1']+1.)/2.
)
data['zs3_norm'] = np.where(
    data['zs2'] == 0,
    0,
    (data['zs3']/data['zs2']+1.)/2.
)
data['etabar_norm'] = (data['etabar'] +3) / 6
data['B2c_norm'] = (data['B2c'] + 3) / 6
data['nfp_norm'] = data['nfp'] / 10
data['p2_norm'] = (data['p2'] + 4e6) / 4e6


X = data[['rc1_norm', 'rc2_norm', 'rc3_norm', 'zs1_norm', 'zs2_norm', 'zs3_norm', 'etabar_norm', 'nfp_norm']].values
X_tensor = torch.tensor(X, dtype=torch.float64)
y_tensor = torch.tensor(y, dtype=torch.float64)

X_train, X_temp, y_train, y_temp = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# DataLoader for test set
batch_size = 1024
test_dataset = TensorDataset(X_test, y_test)
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
    def __init__(self,output_size=2):
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
        self.fc1 = nn.Linear(128, 128) 
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, output_size)  # Output layer for regression

    def forward(self, x):
        # Initial conv layer
        x = x.unsqueeze(1)  # Add channel dimension: [batch_size, 1, 10]
        x = torch.relu(self.bn1(self.conv1(x)))  # Shape: [batch_size, 16, 10]

        # Residual blocks
        x = self.layer1(x)  # Shape: [batch_size, 32, 10]
        x = self.layer2(x)  # Shape: [batch_size, 64, 10]
        x = self.layer3(x)  # Shape: [batch_size, 128, 10]
        x = self.layer4(x)  # Shape: [batch_size, 128, 10]
        x = self.layer5(x)  # Shape: [batch_size, 128, 10]
        
        # Flatten for fully connected layers
        x = self.global_avg_pool(x).view(x.size(0), -1) # Shape becomes [batch_size, 128 * 10]
        
        # Fully connected layers
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)  
        
        return x


# Initialize the model and load the saved weights
model = ResNetRegressor().double().to(device)
model.load_state_dict(torch.load("best_model2.pth", weights_only=True))
model.eval()

# Loss function for evaluation
criterion = nn.MSELoss()

# Evaluate on the test set
test_loss = 0.0
actual_diffs = []
predicted_diffs = []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        # Make predictions
        predictions = model(X_batch)
        
        # Calculate loss
        loss = criterion(predictions, y_batch)
        test_loss += loss.item()
        
        actual_diffs.append(y_batch.cpu().numpy())
        predicted_diffs.append(predictions.cpu().numpy())

# Calculate average test loss
avg_test_loss = test_loss / len(test_loader)
print(f"Test Loss: {avg_test_loss:.4f}")

actual_diffs = np.concatenate(actual_diffs, axis=0)
predicted_diffs = np.concatenate(predicted_diffs, axis=0)

# Inverse transform the diffs back to their original scales
actual_diffs = actual_diffs * y_stds + y_means
predicted_diffs = predicted_diffs * y_stds + y_means

# # Apply expm1 on iota_diff to reverse the log1p transformation
# actual_diffs[:, 0] = np.expm1(actual_diffs[:, 0]) # + iota_min  # For iota_diff (index 0)
# predicted_diffs[:, 0] = np.expm1(predicted_diffs[:, 0]) # + iota_min # For iota_diff predictions (index 0)

diff_names = ['iota', 'min_L_grad_B']
for i in range(2):
    plt.figure(figsize=(10, 5))
    
    sns.kdeplot(actual_diffs[:, i], label=f'Actual {diff_names[i]}', color='blue', lw=2)
    sns.kdeplot(predicted_diffs[:, i], label=f'Predicted {diff_names[i]}', color='orange', lw=2)
    
    plt.xlabel(f"{diff_names[i]}")
    plt.ylabel("Density")
    plt.title(f"Density Distribution of Actual vs. Predicted {diff_names[i]}")
    plt.legend()
    plt.savefig(f"{diff_names[i]}_distribution_comparison.png")
    print(f"{diff_names[i]} density distribution plot saved to '{diff_names[i]}_distribution_comparison.png'.")
    plt.show()


# Define tolerances for each diff type
iota_tolerance = 3  # Larger tolerance for iota_diff
min_L_B_tolerance = 0.05 # Standard tolerance for the other diffs

# Initialize counters for each difference type
accurate_iota_count = 0.0
accurate_L_grad_B_min_count = 0.0
total_count = len(actual_diffs)

# Calculate accuracy for each diff with their respective tolerances
for actual, predicted in zip(actual_diffs, predicted_diffs):
    iota_accuracy = abs(predicted[0] - actual[0]) <= iota_tolerance 
    L_grad_B_min_accuracy = abs(predicted[1] - actual[1]) <= min_L_B_tolerance
    
    # Update counts for each diff type
    accurate_iota_count += iota_accuracy
    accurate_L_grad_B_min_count += L_grad_B_min_accuracy

# Calculate accuracy percentages
iota_accuracy_percentage = (accurate_iota_count / total_count) * 100
L_grad_B_min_accuracy_percentage = (accurate_L_grad_B_min_count / total_count) * 100

# Print accuracy results
print(f"Accuracy with specified tolerances:")
print(f"  iota_diff accuracy (tolerance = {iota_tolerance}): {iota_accuracy_percentage:.2f}%")
print(f"  L_grad_B_min_diff accuracy (tolerance = {min_L_B_tolerance}): {L_grad_B_min_accuracy_percentage:.2f}%")
