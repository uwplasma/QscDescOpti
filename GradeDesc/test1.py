import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


# Define the Residual Block and ResNetRegressor (unchanged)
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = torch.relu(out)
        return out

class ResNetRegressor(nn.Module):
    def __init__(self, output_size=2):
        super(ResNetRegressor, self).__init__()
        self.output_size = output_size
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.layer1 = ResidualBlock(16, 32)
        self.layer2 = ResidualBlock(32, 64)
        self.layer3 = ResidualBlock(64, 128)
        self.layer4 = ResidualBlock(128, 128)
        self.layer5 = ResidualBlock(128, 128)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, output_size)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.global_avg_pool(x).view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)

        if self.output_size == 8:
            x = torch.sigmoid(x)
        return x

model1 = ResNetRegressor(output_size=8)
model2 = ResNetRegressor(output_size=2)  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model1.load_state_dict(torch.load("best_model1.pth", weights_only=True))
model1.to(device)
model1.eval()

model2.load_state_dict(torch.load("best_model2.pth", weights_only=True))
model2.to(device)
model2.eval()

# Load data
data = pd.read_csv("updated_data.csv")

# deal with y, normlize y based on min and max value.
data['iota_diff'] = np.log1p(data['iota_diff'].abs())
data['beta_diff'] = data['beta_diff'].abs()

y = data[['iota_diff', 'L_grad_B_min_diff', 'beta_diff']].values
y_means = y.mean(axis=0)
y_stds = y.std(axis=0)
y = (y - y_means) / y_stds

# normalize x based on their range on table provided. 
# data['rc1_norm'] = (data['rc1'] + 1) / 2
# data['rc2_norm'] = np.where(
#     data['rc1'] == 0,
#     0,
#     (data['rc2']/data['rc1']+1.)/2.
# )
# data['rc3_norm'] = np.where(
#     data['rc2'] == 0,
#     0,
#     (data['rc3']/data['rc2']+1.)/2.
# )
# data['zs1_norm'] = (data['zs1'] + 1) / 2
# data['zs2_norm'] = np.where(
#     data['zs1'] == 0,
#     0,
#     (data['zs2']/data['zs1']+1.)/2.
# )
# data['zs3_norm'] = np.where(
#     data['zs2'] == 0,
#     0,
#     (data['zs3']/data['zs2']+1.)/2.
# )
# data['etabar_norm'] = (data['etabar'] +3) / 6
# data['B2c_norm'] = (data['B2c'] + 3) / 6
# data['nfp_norm'] = data['nfp'] / 10
# data['p2_norm'] = (data['p2'] + 4e6) / 4e6

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


X = data[['rc1_norm', 'rc2_norm', 'rc3_norm', 'zs1_norm', 'zs2_norm', 'zs3_norm', 'etabar_norm', 'nfp_norm']].values

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

batch_size = 64
test_dataset = TensorDataset(X_tensor, y_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
# Track differences
actual_diffs = []
predicted_diffs = []
optimized_diffs = []

# Evaluate model1 and model2
with torch.no_grad():
    for X_batch, y_batch in tqdm(test_loader, desc="Evaluating Models"):
        X_batch = X_batch.to(device)
        actual_diffs.extend(y_batch.cpu().numpy())

        # Predicted differences for original data
        predicted_diff = model2(X_batch).cpu().numpy()
        predicted_diff = predicted_diff * y_stds + y_means
        predicted_diff[:, 0] = np.expm1(predicted_diff[:, 0])  # Apply expm1 to iota_diff
        predicted_diffs.extend(predicted_diff)

        # Optimized differences
        optimized_output = model1(X_batch)
        optimized_diff = model2(optimized_output).cpu().numpy()
        # Inverse normalization for optimized diffs
        optimized_diff = optimized_diff * y_stds + y_means
        optimized_diff[:, 0] = np.expm1(optimized_diff[:, 0])  # Apply expm1 to iota_diff
        optimized_diffs.extend(optimized_diff)


actual_diffs = np.array(actual_diffs) * y_stds + y_means  # Un-normalize actual diffs
actual_diffs[:, 0] = np.expm1(actual_diffs[:, 0])  # Apply expm1 to iota_diff

predicted_diffs = np.array(predicted_diffs)
optimized_diffs = np.array(optimized_diffs)

# Plot distributions
diff_names = ['iota_diff', 'L_grad_B_min_diff', 'beta_diff']
plt.figure(figsize=(18, 5))
for i, diff_name in enumerate(diff_names):
    plt.subplot(1, 3, i+1)
    plt.hist(actual_diffs[:, i], bins=50, alpha=0.5, label=f'Actual {diff_name}', color='blue', edgecolor='black', density=True)
    plt.hist(predicted_diffs[:, i], bins=50, alpha=0.5, label=f'Predicted {diff_name}', color='green', edgecolor='black', density=True)
    plt.hist(optimized_diffs[:, i], bins=50, alpha=0.5, label=f'Optimized {diff_name}', color='orange', edgecolor='black', density=True)
    plt.xlabel(diff_name)
    plt.ylabel("Density")
    plt.title(f"{diff_name} Distribution Comparison")
    plt.legend()

plt.suptitle("Distribution of Actual, Predicted, and Optimized Differences")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("distribution_comparison.png")
plt.show()