import torch

data = torch.Tensor(
    [
        [[1, 0],
        [0, 1]],

        [[0, 1],
         [0, 0]],

        [[1, 0],
         [0, 1]]
    ]
)


values = torch.Tensor([1.0, 0.0, 1.0])


# yeni = values.unsqueeze(0).transpose(0, 1)
# print(yeni.shape)

# print(values[:, None, None].shape)
# # Her pikselin target_values ile eşleşip eşleşmediğini kontrol etmek için maske oluşturuyoruz
# mask = torch.all(data == values[:, None, None], dim=0)

# # İndeksleri buluyoruz
# indices = torch.nonzero(mask)

# print(indices)

data = data.permute(1, 2, 0)

# Broadcast ile karşılaştırma yapma
# values tensörünü (1, 1, C) boyutlarına genişletiyoruz
# ve data tensörü ile karşılaştırıyoruz
mask = torch.all(data == values, dim=-1)
print(mask)