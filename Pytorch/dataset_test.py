from torch.utils.data import Dataset
from PIL import Image
import torch

if torch.cuda.is_available():
    print("CUDA is available")

print("PyTorch version:", torch.__version__)