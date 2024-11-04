import pandas as pd
import numpy as np
from scipy.constants import mu_0

def convert_to_numpy_array(column):
    return column.apply(lambda x: np.array(x.replace('[', '').replace(']', '').split(), dtype=float))

def load_and_process_data(file_path):
    data = pd.read_csv(file_path)

    data['B_variance_computed'] = convert_to_numpy_array(data['B_variance_computed'])
    data['L_grad_B_min_computed'] = convert_to_numpy_array(data['L_grad_B_min_computed'])
    data['beta_computed'] = convert_to_numpy_array(data['beta_computed'])

    data['L_grad_B_min_computed'] = data['L_grad_B_min_computed'].apply(np.min)

    data['beta_computed'] = data.apply(
        lambda row: np.mean(row['beta_computed'] * mu_0 * 2 / (row['B_variance_computed'] ** 2)), axis=1
    )

    X_raw = data[['rc1', 'rc2', 'rc3', 'zs1', 'zs2', 'zs3', 'nfp', 'etabar', 'B2c', 'p2']].values
    y_raw = data[['iota_computed', 'L_grad_B_min_computed', 'beta_computed']].values

    return X_raw, y_raw
