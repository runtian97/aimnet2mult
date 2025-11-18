import torch
from openbabel import pybel
from aimnet2mult.models.jit_wrapper import load_jit_model

# Load the TorchScript model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = "/Users/nickgao/Desktop/pythonProject/aimnet2mult/examples/run/model_fid1.jpt"

model = load_jit_model(model_path)
mol_path = '/Users/nickgao/Desktop/pythonProject/local_code_template/test_sdfs/CAMVES_a.sdf'
# Example input batch

mol = next(pybel.readfile('sdf', mol_path))

coord = torch.as_tensor([a.coords for a in mol.atoms]).unsqueeze(0).to(device)
coord.requires_grad_(True)
numbers = torch.as_tensor([a.atomicnum for a in mol.atoms]).unsqueeze(0).to(device)
charge = torch.as_tensor([mol.charge]).to(device)
mult = torch.as_tensor([mol.spin]).to(device)

_in = {"coord": coord, "numbers": numbers, "charge": charge, "mult": mult}

# Predict forces and Hessian
_out = model(_in, compute_hessian=True, hessian_mode="autograd")

print("Forces shape :", _out["forces"])
print("Hessian:", _out["hessian"])
print(_out.keys())
