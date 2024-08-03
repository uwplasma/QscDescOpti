import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from model import Generator, Discriminator

# read data

original_data = pd.read_csv('train_data.csv')
target_data = pd.read_csv('good_data.csv')

# get the values of the data
X = original_data.values
Y = target_data.values

# standardize the data
scaler_X = StandardScaler()
scaler_Y = StandardScaler()
X = scaler_X.fit_transform(X)
Y = scaler_Y.fit_transform(Y)

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.15, random_state=42)

# convert the data to tensor
X_train = torch.tensor(X_train, dtype=torch.float32)
Y_train = torch.tensor(Y_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
Y_val = torch.tensor(Y_val, dtype=torch.float32)


# create data loader
batch_size = 64
train_dataset = TensorDataset(X_train, Y_train)
val_dataset = TensorDataset(X_val, Y_val)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

input_dim = 20
output_dim = 20
batch_size = 32
n_epochs = 2000
patience = 50

"""
L2 or MSE provides a smooth gradient, reducing the chances of the gradients exploding or vanishing, 
ensuring a smooth training process. 
L1 or MAE is less sensitive to outliers, allowing the reconstructed data to retain distinct features. 
Since the original bad configurations are each different, using L1 helps preserve these distinct features.
"""
adversarial_loss = nn.MSELoss()  # GAN loss
cycle_loss = nn.L1Loss()  # Cycle-consistency loss

G_A2B = Generator(input_dim, output_dim)

G_B2A = Generator(output_dim, input_dim)

D_A = Discriminator(input_dim)

D_B = Discriminator(output_dim)

device = torch.device('mps')

G_A2B.to(device)
G_B2A.to(device)
D_A.to(device)
D_B.to(device)

# Define the loss function and optimizers
# By putting the parameters of both generators together in a single optimizer, their updates are synchronized.
optimizer_G = optim.Adam(list(G_A2B.parameters()) + list(G_B2A.parameters()), lr=0.0002, betas=(0.5, 0.999))
optimizer_D_A = optim.Adam(D_A.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D_B = optim.Adam(D_B.parameters(), lr=0.0002, betas=(0.5, 0.999))

train_loss_records = {
    'loss_GAN_A2B': [],
    'loss_GAN_B2A': [],
    'loss_cycle_A': [],
    'loss_cycle_B': [],
    'loss_id_A': [],
    'loss_id_B': [],
    'loss_D_A': [],
    'loss_D_B': [],
     'loss_G': []
}

val_loss_records = {
    'loss_GAN_A2B': [],
    'loss_GAN_B2A': [],
    'loss_cycle_A': [],
    'loss_cycle_B': [],
    'loss_id_A': [],
    'loss_id_B': [],
    'loss_D_A': [],
    'loss_D_B': [],
    'loss_G': []
}

# To save the best model
best_loss_G = float('inf')
best_model_path = ''
early_stop_counter = 0

for epoch in range(n_epochs):
    G_A2B.train()
    G_B2A.train()
    D_A.train()
    D_B.train()

    train_loss_GAN_A2B = 0
    train_loss_GAN_B2A = 0
    train_loss_cycle_A = 0
    train_loss_cycle_B = 0
    train_loss_id_A = 0
    train_loss_id_B = 0
    train_loss_D_A = 0
    train_loss_D_B = 0
    train_loss_G = 0
    num_batches = 0

    for real_A, real_B in train_loader:
        real_A, real_B = real_A.to(device), real_B.to(device)
        current_batch_size = real_A.size(0)
        
        # make the labels
        valid = torch.ones((real_A.size(0), 1), requires_grad=False).to(device)
        fake = torch.zeros((real_A.size(0), 1), requires_grad=False).to(device)
        
        # Train the generators
        optimizer_G.zero_grad()

        # Identity loss
        '''
        The identity loss ensures that when an image of domain A (or B) 
        is fed to the generator for the same domain (G_B2A for A or G_A2B for B), 
        the output should be the same as the input.
        '''
        loss_id_A = cycle_loss(G_B2A(real_A), real_A)
        loss_id_B = cycle_loss(G_A2B(real_B), real_B)

        # GAN loss
        '''
        The GAN loss is used to make the generated configurations (fake_B and fake_A) indistinguishable 
        from good configurations by the discriminators (D_B and D_A).
        '''
        fake_B = G_A2B(real_A)
        loss_GAN_A2B = adversarial_loss(D_B(fake_B), valid[:fake_B.size(0)])
        fake_A = G_B2A(real_B)
        loss_GAN_B2A = adversarial_loss(D_A(fake_A), valid[:fake_A.size(0)])

        # Cycle loss
        '''
        The cycle loss enforces that if translating a configuration to the other domain 
        and then back to the original domain, we should get back the original configuration
        (i.e., real_A -> fake_B -> recov_A should be similar to real_A).
        '''
        recov_A = G_B2A(fake_B)
        loss_cycle_A = cycle_loss(recov_A, real_A)
        recov_B = G_A2B(fake_A)
        loss_cycle_B = cycle_loss(recov_B, real_B)

        # Total loss
        '''
        Balances the three types of losses to ensure that the generators 
        not only produce realistic images 
        but also maintain the identity and cycle consistency of the images.

        '''
        loss_G = (2*loss_GAN_A2B + loss_GAN_B2A) + (loss_cycle_A + loss_cycle_B) + (loss_id_A + loss_id_B)
        # loss_G = (loss_GAN_A2B + loss_GAN_B2A) + 5*(loss_cycle_A + loss_cycle_B) + 5*(loss_id_A + loss_id_B)


        loss_G.backward()
        optimizer_G.step()

        train_loss_GAN_A2B += loss_GAN_A2B.item()
        train_loss_GAN_B2A += loss_GAN_B2A.item()
        train_loss_cycle_A += loss_cycle_A.item()
        train_loss_cycle_B += loss_cycle_B.item()
        train_loss_id_A += loss_id_A.item()
        train_loss_id_B += loss_id_B.item()
        train_loss_G += loss_G.item()
        num_batches += 1
        
        # Train the discriminators
        optimizer_D_A.zero_grad()
        # Real loss
        '''
        The real loss measures how well the discriminator can identify real images as real.
        '''
        loss_real = adversarial_loss(D_A(real_A), valid[:real_A.size(0)])
        # Fake loss
        '''
        The fake loss measures how well the discriminator can identify fake images as fake.
        '''
        loss_fake = adversarial_loss(D_A(fake_A.detach()), fake[:fake_A.size(0)])
        # Total loss
        loss_D_A = (loss_real + loss_fake) / 2

        loss_D_A.backward()
        optimizer_D_A.step()
        train_loss_D_A += loss_D_A.item()

        optimizer_D_B.zero_grad()

        # Real loss
        loss_real = adversarial_loss(D_B(real_B), valid[:real_B.size(0)])
        # Fake loss
        loss_fake = adversarial_loss(D_B(fake_B.detach()), fake[:fake_B.size(0)])
        # Total loss
        loss_D_B = (loss_real + loss_fake) / 2

        loss_D_B.backward()
        optimizer_D_B.step()

        train_loss_D_B += loss_D_B.item()
    
    train_loss_records['loss_GAN_A2B'].append(train_loss_GAN_A2B / num_batches)
    train_loss_records['loss_GAN_B2A'].append(train_loss_GAN_B2A / num_batches)
    train_loss_records['loss_cycle_A'].append(train_loss_cycle_A / num_batches)
    train_loss_records['loss_cycle_B'].append(train_loss_cycle_B / num_batches)
    train_loss_records['loss_id_A'].append(train_loss_id_A / num_batches)
    train_loss_records['loss_id_B'].append(train_loss_id_B / num_batches)
    train_loss_records['loss_D_A'].append(train_loss_D_A / num_batches)
    train_loss_records['loss_D_B'].append(train_loss_D_B / num_batches)
    train_loss_records['loss_G'].append(train_loss_G / num_batches)

    # if (epoch+1) % 1000 == 0 or loss_GAN_A2B.cpu().item() < best_loss_GAN_A2B:
    #     if (epoch+1) % 1000 == 0:
    # #         torch.save(G_A2B.state_dict(), f'G_A2B_checkpoint_{epoch+1}.pth')
    # #     if loss_GAN_A2B.cpu().item() < best_loss_GAN_A2B:
    # #         best_loss_GAN_A2B = loss_GAN_A2B.cpu().item()
    # #         best_model_path = f'G_A2B_best_model.pth'
    # #         torch.save(G_A2B.state_dict(), best_model_path)

    G_A2B.eval()
    G_B2A.eval()
    D_A.eval()
    D_B.eval()

    with torch.no_grad():
        val_loss_GAN_A2B, val_loss_GAN_B2A, val_loss_cycle_A, val_loss_cycle_B, val_loss_id_A, val_loss_id_B, val_loss_D_A, val_loss_D_B, val_loss_G = 0, 0, 0, 0, 0, 0, 0, 0,0
        val_batches = 0

        for real_A, real_B in val_loader:
            real_A, real_B = real_A.to(device), real_B.to(device)
            valid = torch.ones((real_A.size(0), 1), requires_grad=False).to(device)

            fake_B = G_A2B(real_A)
            fake_A = G_B2A(real_B)
            val_loss_GAN_A2B += adversarial_loss(D_B(fake_B), valid[:fake_B.size(0)]).cpu().item()
            val_loss_GAN_B2A += adversarial_loss(D_A(fake_A), valid[:fake_A.size(0)]).cpu().item()
            recov_A = G_B2A(fake_B)
            val_loss_cycle_A += cycle_loss(recov_A, real_A).cpu().item()
            recov_B = G_A2B(fake_A)
            val_loss_cycle_B += cycle_loss(recov_B, real_B).cpu().item()
            val_loss_id_A += cycle_loss(G_B2A(real_A), real_A).cpu().item()
            val_loss_id_B += cycle_loss(G_A2B(real_B), real_B).cpu().item()

            val_loss_real_A = adversarial_loss(D_A(real_A), valid[:real_A.size(0)]).cpu().item()
            val_loss_fake_A = adversarial_loss(D_A(fake_A.detach()), fake[:fake_A.size(0)]).cpu().item()
            val_loss_D_A += (val_loss_real_A + val_loss_fake_A) / 2

            val_loss_real_B = adversarial_loss(D_B(real_B), valid[:real_B.size(0)]).cpu().item()
            val_loss_fake_B = adversarial_loss(D_B(fake_B.detach()), fake[:fake_B.size(0)]).cpu().item()
            val_loss_D_B += (val_loss_real_B + val_loss_fake_B) / 2

            val_loss_G += (2 * val_loss_GAN_A2B + val_loss_GAN_B2A) + (val_loss_cycle_A + val_loss_cycle_B) + (val_loss_id_A + val_loss_id_B)


            val_batches += 1

        
        val_loss_records['loss_GAN_A2B'].append(val_loss_GAN_A2B / val_batches)
        val_loss_records['loss_GAN_B2A'].append(val_loss_GAN_B2A / val_batches)
        val_loss_records['loss_cycle_A'].append(val_loss_cycle_A / val_batches)
        val_loss_records['loss_cycle_B'].append(val_loss_cycle_B / val_batches)
        val_loss_records['loss_id_A'].append(val_loss_id_A / val_batches)
        val_loss_records['loss_id_B'].append(val_loss_id_B / val_batches)
        val_loss_records['loss_D_A'].append(val_loss_D_A / val_batches)
        val_loss_records['loss_D_B'].append(val_loss_D_B / val_batches)
        val_loss_records['loss_G'].append(val_loss_G / val_batches)

        print(f'Epoch {epoch+1}/{n_epochs}, Validation Loss: '
              f'GAN_A2B: {val_loss_GAN_A2B / val_batches}, '
              f'GAN_B2A: {val_loss_GAN_B2A / val_batches}, '
              f'Cycle_A: {val_loss_cycle_A / val_batches}, '
              f'Cycle_B: {val_loss_cycle_B / val_batches}, '
              f'ID_A: {val_loss_id_A / val_batches}, '
              f'ID_B: {val_loss_id_B / val_batches}, '
              f'D_A: {val_loss_D_A / val_batches}, '
              f'D_B: {val_loss_D_B / val_batches}',
              f'G: {val_loss_G / val_batches}')
        
        if val_loss_G / val_batches < best_loss_G:
            best_loss_G = val_loss_G / val_batches
            best_model_path = 'G_A2B_best_model.pth'
            torch.save(G_A2B.state_dict(), best_model_path)
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            print(f"Early stopping triggered, stopping training at epoch {epoch+1}")
            break

# Plot loss graphs
epochs_range = range(1, len(train_loss_records['loss_G']) + 1)

plt.figure(figsize=(10, 8))
plt.subplot(3, 1, 1)
for key, values in train_loss_records.items():
    if key != 'loss_G':
        plt.plot(epochs_range, values, label=f'Train {key}')
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.legend()

plt.subplot(3, 1, 2)
for key, values in val_loss_records.items():
    if key != 'loss_G':
        plt.plot(epochs_range, values, label=f'Val {key}')
plt.xlabel('Epochs')
plt.ylabel('Validation Loss')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(epochs_range, train_loss_records['loss_G'], label='Train Loss G')
plt.plot(epochs_range, val_loss_records['loss_G'], label='Val Loss G')
plt.xlabel('Epochs')
plt.ylabel('Generator Loss')
plt.legend()

plt.show()