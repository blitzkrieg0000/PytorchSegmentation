from matplotlib import pyplot as plt
import torch
from torchvision.transforms import v2 as transformsv2
from Tool.Util import GetUniqueRGBPixels, RGBMaskToColorMap
import torchvision

# img = cv2.imread("data/data/masks/00000.png")
# image = torch.from_numpy(img)

image = torchvision.io.read_image("data/data/masks/00001.png", torchvision.io.ImageReadMode.RGB)


class MaskTransforms():
    def __init__(self, color_maps:dict[int, torch.Tensor]=[], channel_first=False):
        self.ColorMaps = color_maps
        self.ChannelFirst = channel_first

    def __call__(self, image):
        maps = GetUniqueRGBPixels(image, self.ChannelFirst)
        maps.update(self.ColorMaps)
        newMask = RGBMaskToColorMap(image, maps, self.ChannelFirst)
        return newMask
    

class MaskTransforms():
    def __init__(self, color_maps:dict[int, torch.Tensor]=[], channel_first=False):
        self.ColorMaps = color_maps
        self.ChannelFirst = channel_first

    def __call__(self, image):
        maps = GetUniqueRGBPixels(image, self.ChannelFirst)
        maps.update(self.ColorMaps)
        newMask = RGBMaskToColorMap(image, maps, self.ChannelFirst)
        return newMask
    
TRANSFORMS = transformsv2.Compose([
    transformsv2.ToImage(),
    transformsv2.Resize((256, 256), antialias=False, interpolation=torchvision.transforms.InterpolationMode.NEAREST)
])

MASK_TRANSFORMS = transformsv2.Compose([
    TRANSFORMS,
    MaskTransforms(channel_first=True),

])


newMask = MASK_TRANSFORMS(image)

plt.figure(figsize=(10, 10))
plt.imshow(newMask.numpy())
plt.show()



