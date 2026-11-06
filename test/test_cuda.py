import torch
print("CUDA доступна:", torch.cuda.is_available())
print("Версия PyTorch:", torch.__version__)
print("Версия CUDA в PyTorch:", torch.version.cuda)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))