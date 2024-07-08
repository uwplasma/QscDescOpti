import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import Generator, Discriminator

# read data
'''Reading the original and target data from CSV files. 
Only the first 116 rows are read from the original dataset 
to maintain consistency because the dataset contains only 116 good stellarators.'''

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

# split the data into training set, validation set and test set
# Splitting the data into training (70%), validation (15%), and test (15%) sets.
X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.3, random_state=42)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

# convert the data to tensor
X_train = torch.tensor(X_train, dtype=torch.float32)
Y_train = torch.tensor(Y_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
Y_val = torch.tensor(Y_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
Y_test = torch.tensor(Y_test, dtype=torch.float32)

# create data loader
batch_size = 64
train_dataset = TensorDataset(X_train, Y_train)
val_dataset = TensorDataset(X_val, Y_val)
test_dataset = TensorDataset(X_test, Y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

# Define the generator and discriminator models
input_dim = 20
output_dim = 20

# Transform samples from domain X to samples from domain Y.
G_X2Y = Generator(input_dim, output_dim)

# Transform samples from domain Y to samples from domain X.
G_Y2X = Generator(input_dim, output_dim)

# Discriminate between samples from domain X and generated samples from domain Y.
D_X = Discriminator(input_dim)

# Discriminate between samples from domain Y and generated samples from domain X.
D_Y = Discriminator(input_dim)

# Apple uses the MPS backend for training on the M1 chip
# Linux and Windows users can remove the following line
device = torch.device('mps')

G_X2Y.to(device)
G_Y2X.to(device)
D_X.to(device)
D_Y.to(device)

# Define the loss function and optimizers
criterion = nn.MSELoss()
optimizer_G_X2Y = optim.RMSprop(G_X2Y.parameters(), lr=0.0002)
optimizer_G_Y2X = optim.RMSprop(G_Y2X.parameters(), lr=0.0002)
optimizer_D_X = optim.RMSprop(D_X.parameters(), lr=0.0002)
optimizer_D_Y = optim.RMSprop(D_Y.parameters(), lr=0.0002)

# use learning rate scheduler, which reduces the learning rate by a factor of 0.5 every 200 epochs
scheduler_G_X2Y = optim.lr_scheduler.StepLR(optimizer_G_X2Y, step_size=100, gamma=0.5)
scheduler_G_Y2X = optim.lr_scheduler.StepLR(optimizer_G_Y2X, step_size=100, gamma=0.5)
scheduler_D_X = optim.lr_scheduler.StepLR(optimizer_D_X, step_size=100, gamma=0.5)
scheduler_D_Y = optim.lr_scheduler.StepLR(optimizer_D_Y, step_size=100, gamma=0.5)

num_epochs = 3000
save_interval = 1000

for epoch in range(num_epochs):
    G_X2Y.train()
    G_Y2X.train()
    D_X.train()
    D_Y.train()
    
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        current_batch_size = batch_x.size(0)

        # train the discriminator (every other step),since discriminator learns faster than the generator
        if epoch % 5 == 0:
            optimizer_D_X.zero_grad()

            '''
            These two lines create smoothed labels for the real (0.9-1.0) and fake (0.0-0.1) data 
            to improve GAN training stability by preventing the discriminator from becoming overconfident.
            '''
            real_labels_smooth = torch.FloatTensor(current_batch_size, 1).uniform_(0.9, 1.0).to(device)
            fake_labels_smooth = torch.FloatTensor(current_batch_size, 1).uniform_(0.0, 0.1).to(device)

            d_x_real = D_X(batch_x)
            d_loss_real_x = criterion(d_x_real, real_labels_smooth)
            
            y_fake = G_X2Y(batch_x)
            d_x_fake = D_X(y_fake.detach())
            d_loss_fake_x = criterion(d_x_fake, fake_labels_smooth)
            
            d_loss_x = (d_loss_real_x + d_loss_fake_x) / 2
            d_loss_x.backward()
            optimizer_D_X.step()

            optimizer_D_Y.zero_grad()
            
            d_y_real = D_Y(batch_y)
            d_loss_real_y = criterion(d_y_real, real_labels_smooth)
            
            x_fake = G_Y2X(batch_y)
            d_y_fake = D_Y(x_fake.detach())
            d_loss_fake_y = criterion(d_y_fake, fake_labels_smooth)
            
            d_loss_y = (d_loss_real_y + d_loss_fake_y) / 2
            d_loss_y.backward()
            optimizer_D_Y.step()
           
           # train the generator
        '''
        Generate fake samples 'y_fake' from 'batch_x' using the generator G_X2Y 
        Calculate cycle consistency loss: torch.mean(torch.abs(batch_x - G_Y2X(y_fake))).
        to ensure that the generator can reconstruct the original input.
        '''
        optimizer_G_X2Y.zero_grad()
        y_fake = G_X2Y(batch_x)
        d_y_fake = D_Y(y_fake)
        real_labels_smooth = torch.ones(current_batch_size, 1).to(device) 
        g_loss_x2y = criterion(d_y_fake, real_labels_smooth) + torch.mean(torch.abs(batch_x - G_Y2X(y_fake)))
        g_loss_x2y.backward()
        optimizer_G_X2Y.step()

        '''
        Generate fake samples 'x_fake' from 'batch_y' using the generator G_Y2X
        Calculate cycle consistency loss: torch.mean(torch.abs(batch_y - G_X2Y(x_fake)))
        to ensure that the generator can reconstruct the original input.
        '''
        optimizer_G_Y2X.zero_grad()
        x_fake = G_Y2X(batch_y)
        d_x_fake = D_X(x_fake)
        g_loss_y2x = criterion(d_x_fake, real_labels_smooth) + torch.mean(torch.abs(batch_y - G_X2Y(x_fake)))
        g_loss_y2x.backward()
        optimizer_G_Y2X.step()
    
    # update the learning rate
    scheduler_G_X2Y.step()
    scheduler_G_Y2X.step()
    scheduler_D_X.step()
    scheduler_D_Y.step()
    
    # evaluate the model
    G_X2Y.eval()
    G_Y2X.eval()
    D_X.eval()
    D_Y.eval()

    with torch.no_grad():
        val_d_loss_x = 0
        val_d_loss_y = 0
        val_g_loss_x2y = 0
        val_g_loss_y2x = 0
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            current_batch_size = batch_x.size(0)

            real_labels_smooth = torch.FloatTensor(current_batch_size, 1).uniform_(0.9, 1.0).to(device)
            fake_labels_smooth = torch.FloatTensor(current_batch_size, 1).uniform_(0.0, 0.1).to(device)

            d_x_real = D_X(batch_x)
            d_loss_real_x = criterion(d_x_real, real_labels_smooth)
            
            y_fake = G_X2Y(batch_x)
            d_x_fake = D_X(y_fake)
            d_loss_fake_x = criterion(d_x_fake, fake_labels_smooth)
            val_d_loss_x += (d_loss_real_x + d_loss_fake_x) / 2
            
            d_y_real = D_Y(batch_y)
            d_loss_real_y = criterion(d_y_real, real_labels_smooth)
            
            x_fake = G_Y2X(batch_y)
            d_y_fake = D_Y(x_fake)
            d_loss_fake_y = criterion(d_y_fake, fake_labels_smooth)
            val_d_loss_y += (d_loss_real_y + d_loss_fake_y) / 2

            real_labels_smooth = torch.ones(current_batch_size, 1).to(device)
            d_y_fake = D_Y(y_fake)
            val_g_loss_x2y += criterion(d_y_fake, real_labels_smooth) + torch.mean(torch.abs(batch_x - G_Y2X(y_fake)))
            
            d_x_fake = D_X(x_fake)
            val_g_loss_y2x += criterion(d_x_fake, real_labels_smooth) + torch.mean(torch.abs(batch_y - G_X2Y(x_fake)))
        
        val_d_loss_x /= len(val_loader)
        val_d_loss_y /= len(val_loader)
        val_g_loss_x2y /= len(val_loader)
        val_g_loss_y2x /= len(val_loader)
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], d_loss_x: {d_loss_x.item():.4f}, d_loss_y: {d_loss_y.item():.4f}, g_loss_x2y: {g_loss_x2y.item():.4f}, g_loss_y2x: {g_loss_y2x.item():.4f}')
        print(f'Validation: d_loss_x: {val_d_loss_x:.4f}, d_loss_y: {val_d_loss_y:.4f}, g_loss_x2y: {val_g_loss_x2y:.4f}, g_loss_y2x: {val_g_loss_y2x:.4f}')
    
    if (epoch + 1) % save_interval == 0:
        torch.save(G_X2Y.state_dict(), f'G_X2Y_epoch_{epoch+1}.pth')
        torch.save(G_Y2X.state_dict(), f'G_Y2X_epoch_{epoch+1}.pth')
        torch.save(D_X.state_dict(), f'D_X_epoch_{epoch+1}.pth')
        torch.save(D_Y.state_dict(), f'D_Y_epoch_{epoch+1}.pth')