import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from model import Generator


original_data = pd.read_csv('test_data.csv')
target_data = pd.read_csv('good_data.csv')

X = original_data.values

scaler_X = StandardScaler()
scaler_Y = StandardScaler()
Y= scaler_Y.fit_transform(target_data.values)

G_A2B = Generator(20, 20)

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

G_A2B.to(device)
G_A2B.load_state_dict(torch.load('G_A2B_Base_55ac.pth'))
G_A2B.eval()

def process_chunk(chunk, scaler, scaler_Y, G_A2B, device):
    data = chunk.values
    data = scaler.fit_transform(data)
    data = torch.tensor(data, dtype=torch.float32).to(device)

    with torch.no_grad():
        output = G_A2B(data)
        output = output.cpu().numpy()
        output = scaler_Y.inverse_transform(output)

    generated_data = pd.DataFrame(output, columns=target_data.columns)
    return generated_data

chunk_size = 64
num_chunks = len(original_data) // chunk_size + (1 if len(original_data) % chunk_size != 0 else 0)

for i in range(num_chunks):
    start_row = i * chunk_size
    end_row = (i + 1) * chunk_size
    chunk = original_data.iloc[start_row:end_row]
    
    generated_chunk = process_chunk(chunk, scaler_X, scaler_Y, G_A2B, device)
    
    if i == 0:
        generated_chunk.to_csv('generated_data.csv', index=False, mode='w')  
    else:
        generated_chunk.to_csv('generated_data.csv', index=False, mode='a', header=False)  

generated_samples = pd.read_csv('generated_data.csv')