import torch
import os
import sys

print("-" * 40)
print(f"Python Executable: {sys.executable}")
print(f"Python Version: {sys.version.split()[0]}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}") # EXPECT False
print(f"Device used by default: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
print("-" * 40)
