import cv2
from matplotlib import pyplot as plt
import torch
from torchvision.transforms import v2 as tranformsv2
from Tool.Util import GetUniqueRGBPixels, RGBMaskToColorMap


img = cv2.imread("data/data/masks/00000.png")
image = torch.from_numpy(img)


class MaskTransforms():
    def __init__(self, color_maps:dict[int, torch.Tensor]=[], channel_first=False):
        self.ColorMaps = color_maps
        self.ChannelFirst = channel_first

    def __call__(self, image):
        maps = GetUniqueRGBPixels(image, self.ChannelFirst)
        maps.update(self.ColorMaps)
        newMask = RGBMaskToColorMap(image, maps, self.ChannelFirst)
        return newMask


image = image.permute(2, 0, 1)
print("Önceki boyut:", image.shape)

MASK_TRANSFORMS = tranformsv2.Compose([
    tranformsv2.ToImage(),
    tranformsv2.Resize((256, 256), antialias=True),
])

newMask = MASK_TRANSFORMS(image)

print("Sonraki Boyut:", newMask.shape)

# plt.figure(figsize=(10, 10))
# plt.imshow(newMask.numpy())
# plt.show()



