import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
from qsc import Qsc
from model import Generator
from sklearn.preprocessing import StandardScaler

# Define the generator models
input_dim = 20
output_dim = 20

G_X2Y = Generator(input_dim, output_dim)

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

G_X2Y.to(device)

# Load the trained model parameters
G_X2Y.load_state_dict(torch.load('G_X2Y_epoch_3000.pth'))

G_X2Y.eval()

# Load the test data
original_data = pd.read_csv('test_data.csv')
good_data = pd.read_csv('good_data.csv')

X = original_data.values

scaler_X = StandardScaler()
scaler_Y = StandardScaler()
X = scaler_X.fit_transform(X)
Y= scaler_Y.fit_transform(good_data.values)


X_test = torch.tensor(X, dtype=torch.float32).to(device)

# Generate samples
with torch.no_grad():
    test_y_fake = G_X2Y(X_test).cpu().numpy()

test_y_fake = scaler_Y.inverse_transform(test_y_fake)
generated_df = pd.DataFrame(test_y_fake, columns=original_data.columns)
generated_df.to_csv('generated_samples.csv', index=False)


generated_samples = pd.read_csv('generated_samples.csv')
transformed_valid = pd.DataFrame()

for index, row in generated_samples.iterrows():
    try:
        label = [row['rc1'], row['rc2'], row['rc3'], row['zs1'], row['zs2'], row['zs3'], row['nfp'], row['etabar'], row['B2c'], row['p2']]
        
        rc_values = label[:3]
        zs_values = label[3:6]
        nfp_value = label[6]
        etabar_value = label[7]
        B2c_value = label[8]
        p2_value = label[9]
    
        stel = Qsc(
            rc=[1.] + rc_values,
            zs=[0.] + zs_values,
            nfp=nfp_value,
            etabar=etabar_value,
            B2c=B2c_value,
            p2=p2_value,
            order='r2'
        )
        
        stel.plot_boundary(r=0.01)
        transformed_valid = transformed_valid.append(row)
    
    except ValueError as e:
        print(f"Row {index} raised ValueError: {e}")
    
    except Exception as e:
        print(f"Row {index} raised an unexpected error: {e}")

accuracy = len(transformed_valid) / len(generated_samples) * 100
print(f"Generated Samples useable output percent: {accuracy:.2f}%")