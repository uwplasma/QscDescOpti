import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from data_processing import load_and_process_data
from model import FullyConnectedModel
from qsc_functions import compute_warm_start_loss, compute_qsc_outputs

X_raw, y_raw = load_and_process_data('computed_desc.csv')
X_train, X_temp, y_train, y_temp = train_test_split(X_raw, y_raw, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)

model = FullyConnectedModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

num_epochs = 50
num_warm_start_epochs = 10

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        predicted_params = model(X_batch)

        if epoch < num_warm_start_epochs:
            loss = compute_warm_start_loss(predicted_params, X_batch)
        else:
            predicted_params = compute_qsc_outputs(predicted_params)
            loss = criterion(predicted_params, y_batch)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_val_batch, y_val_batch in val_loader:
            predicted_params = model(X_val_batch)
            if epoch < num_warm_start_epochs:
                val_loss += compute_warm_start_loss(predicted_params, X_val_batch).item()
            else:
                predicted_params = compute_qsc_outputs(predicted_params)
                val_loss += criterion(predicted_params, y_val_batch).item()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss:.4f}, Val Loss: {val_loss:.4f}")
