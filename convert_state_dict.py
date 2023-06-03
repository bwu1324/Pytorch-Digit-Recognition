# Converts PyTorch State Dictionary to TorchScript Format

import torch
from model import DigitRecognition

# Location of State Dictionary
MODEL_PATH = "training_checkpoints/2023-06-03_10-00-58/epoch-100.pth"
# Where to save TorchScript File
SAVE_LOCATION = "models/1_big_linear.pt"

model = DigitRecognition(0.0)
model.load_state_dict(torch.load(MODEL_PATH))
torch.jit.script(model).save(SAVE_LOCATION)
