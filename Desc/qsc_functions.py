import torch
from qsc import Qsc
from scipy.constants import mu_0

def compute_warm_start_loss(predicted_params, target):
    loss_fn = torch.nn.MSELoss()
    total_loss = 0
    for i in range(len(predicted_params)):
        loss = loss_fn(predicted_params[i], target[i])
        total_loss += loss
    return total_loss / len(predicted_params)

def compute_qsc_outputs(predicted_params):
    qsc_outputs = []
    for i in range(len(predicted_params)):
        rc_values = predicted_params[i, :3].tolist()
        zs_values = predicted_params[i, 3:6].tolist()
        nfp, etabar, B2c, p2 = predicted_params[i, 6:].tolist()

        stel = Qsc(rc=[1.0] + rc_values, zs=[0.0] + zs_values, nfp=nfp, etabar=etabar, B2c=B2c, p2=p2, order="r2")

        iota = torch.tensor(stel.iota, dtype=torch.float32, requires_grad=True)
        min_L_grad_B = torch.min(torch.tensor(stel.L_grad_B, dtype=torch.float32, requires_grad=True))
        B0_tensor = torch.tensor(stel.B0, dtype=torch.float32, requires_grad=True)
        p2_tensor = torch.tensor(stel.p2, dtype=torch.float32, requires_grad=True)
        beta_on_axis_NAE = torch.mean(p2_tensor * mu_0 * 2 / B0_tensor**2)

        qsc_output = torch.tensor([iota, min_L_grad_B, beta_on_axis_NAE], dtype=torch.float32)
        qsc_outputs.append(qsc_output)

    return torch.stack(qsc_outputs)
