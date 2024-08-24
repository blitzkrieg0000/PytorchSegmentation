import torch



data = torch.tensor([[255, 0, 0], [192, 0, 0], [0, 255, 0],[0, 255, 0], [0, 0, 255], [0, 0, 255]], dtype=torch.uint8)

# print(data.unique(dim=0))

print(data.shape)

result = data.sum(dim=-1)

print(result)