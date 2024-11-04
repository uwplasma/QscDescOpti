import pandas as pd
import os
from qsc import Qsc
from desc.equilibrium import Equilibrium
from desc.grid import LinearGrid
import numpy as np

# Directory containing the CSV files
directory = "./"  # Assuming the script is in the same directory

def extract_numeric_part(filename):
    numeric_part = ""
    for char in filename:
        if char.isdigit():
            numeric_part += char
    return numeric_part

# Loop through all CSV files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        # Load the first 10 columns of the CSV file into a DataFrame
        df = pd.read_csv(os.path.join(directory, filename), usecols=range(10))

        # DataFrame to store rows with computed results
        computed_columns = ['iota_computed', 'B_variance_computed', 'L_grad_B_min_computed', 'beta_computed']
        succ_rows = pd.DataFrame(columns=df.columns.tolist() + computed_columns)

        for index, row in df.iterrows():
            try:
                # Extract the parameters for QSC
                rc_values = [row['rc1'], row['rc2'], row['rc3']]
                zs_values = [row['zs1'], row['zs2'], row['zs3']]
                nfp_value = int(row['nfp'])
                etabar_value = row['etabar']
                B2c_value = row['B2c']
                p2_value = row['p2']

                # Create Qsc object without solving DESC (as the data is already verified)
                stel = Qsc(
                    rc=[1.] + rc_values,
                    zs=[0.] + zs_values,
                    nfp=nfp_value,
                    etabar=etabar_value,
                    B2c=B2c_value,
                    p2=p2_value,
                    order='r2'
                )

                # Generate DESC equilibrium
                eq_fit = Equilibrium.from_near_axis(L=6,M=6,N=6, na_eq = stel,r = 0.05)
                eq = eq_fit.copy()
                eq.solve(verbose=2,ftol=1e-3,maxiter=100)
                # Create the grid and compute metrics (iota, |B|, L_grad(B), p)
                grid = LinearGrid(rho=0.0,axis=True, N=3*eq.N, NFP=eq.NFP)
                data_axis = eq.compute(["iota","|B|","L_grad(B)","p"],grid=grid)

                # Extract the computed metrics
                iota_value = data_axis["iota"][0]
                B_variance = data_axis["|B|"]
                L_grad_B_min = data_axis["L_grad(B)"]
                beta_value = data_axis["p"]

                # Append the computed values to the DataFrame
                row_data = row.tolist() + [iota_value, B_variance, L_grad_B_min, beta_value]
                succ_rows.loc[len(succ_rows)] = row_data

            except Exception as e:
                print(f"Row {index} raised an unexpected error: {e}")

        # Extract numeric part from the filename
        numeric_part = extract_numeric_part(filename)

        # Prepare the output filename using the numeric part
        output_filename = f"computed_output_{numeric_part}.csv"

        # Save the DataFrame with computed results to CSV
        succ_rows.to_csv(os.path.join(directory, output_filename), index=False)
