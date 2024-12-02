import torch
print("GPU Available:", torch.cuda.is_available())
print("Current GPU:", torch.cuda.get_device_name(0))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)