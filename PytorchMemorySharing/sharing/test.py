import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 设置随机数种子，以确保每次生成的 Tensor 内容一致
torch.manual_seed(42)

# 定义一系列 Tensor 大小
tensor_sizes = [(5000, 5000), (10000, 10000), (15000, 15000),(20000,20000),(25000,25000),(30000,30000),(35000,35000)]

streamPooling=[]
  #eventPooling=[]
  # 4 Init.
for i in range(0,4):
    stream = torch.cuda.Stream(priority=50)
    #event = torch.cuda.Event()
    streamPooling.append(stream)
for i in range(0,4):
    with torch.cuda.stream(streamPooling[i]):
        tensor = torch.empty((5000,5000), device=device)

for i in range(0,4):
    with torch.cuda.stream(streamPooling[i]):
        for size in tensor_sizes:
            tensor = torch.empty(size, device=device)
            print(f"Tensor size: {tensor.size()}")
            del tensor
            print("Tensor 已释放\n")

