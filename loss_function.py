import torch
import math
from qsc import Qsc
from desc.equilibrium import Equilibrium
from desc.grid import LinearGrid
from scipy.constants import mu_0
import numpy as np

def compute_loss(X_updated):
    total_loss = 0
    loss_fn = torch.nn.MSELoss()
    
    for i in range(X_updated.shape[0]):
        try:
            rc_values = X_updated[i, :3].tolist()
            zs_values = X_updated[i, 3:6].tolist()
            nfp, etabar, B2c, p2 = X_updated[i, 6:].tolist()
            nfp = max(1, math.ceil(nfp))

            # Create a Qsc object
            qsc = Qsc(rc=[1.0] + rc_values, zs=[0.0] + zs_values, nfp=nfp, etabar=etabar, B2c=B2c, p2=p2, order="r2")

            # Create a DESC object
            eq_fit = Equilibrium.from_near_axis(L=6, M=6, N=6, na_eq=qsc, r=0.05)
            eq = eq_fit.copy()
            eq.solve(verbose=2, ftol=1e-3, maxiter=100)

            grid = LinearGrid(rho=0.0, axis=True, N=3 * eq.N, NFP=eq.NFP)
            data_axis = eq.compute(["iota", "|B|", "L_grad(B)", "p"], grid=grid)

            # calculate the difference, which is Z_updated - Y_updated
            iota_difference = float(data_axis["iota"][0] - qsc.iota)
            min_L_grad_B_diff = float(np.min(data_axis["L_grad(B)"]) - np.min(qsc.L_grad_B))

            beta_on_axis_DESC = np.mean(data_axis["p"] * mu_0 * 2 / data_axis["|B|"]**2)
            beta_on_axis_NAE = np.mean(qsc.p2 * mu_0 * 2 / qsc.B0**2)
            beta_on_axis_diff = float(beta_on_axis_DESC - beta_on_axis_NAE)

            total_loss += loss_fn(torch.tensor([iota_difference, min_L_grad_B_diff, beta_on_axis_diff]), torch.tensor([0.0, 0.0, 0.0]))
        except ValueError as e:
            total_loss += torch.tensor(1e6)

    return (total_loss / X_updated.shape[0]).clone().detach().requires_grad_(True)
