import os

import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2 as transformsv2

from Tool.Util import GetUniqueRGBPixels, RGBMaskToColorMap

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") 


image_dir = "./data/data/images"
mask_dir = "./data/data/masks"


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
    # transformsv2.ToImage(),
    transformsv2.Resize((256, 256), antialias=False, interpolation=InterpolationMode.NEAREST) # Maskenin bozulmaması için NEAREST
])

MASK_TRANSFORMS = transformsv2.Compose([
    TRANSFORMS,
    MaskTransforms(channel_first=True),

])


class CustomSegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transforms=None, mask_transforms=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.dataset = []
        self.transforms = transforms
        self.mask_transforms = mask_transforms
        self.unique_values = set()
        self.img_files = []
        self.mask_files = []
        self.ReadMetadata()


    def ReadMetadata(self):
        # Klasörleri Oku
        self.img_files = np.array(os.listdir(self.img_dir))
        self.mask_files = np.array(os.listdir(self.mask_dir))
        
        for img_file in self.img_files :
            img_name = "".join(img_file.split(".")[:-1])                # "image" dosyasının adını çıkar. Uzantısı hariç
            conditions = list(map(lambda x: img_name in x, self.mask_files))
            if any(conditions):
                mask_file = self.mask_files[conditions]
                if not mask_file:
                    continue

                mask_path = os.path.join(self.mask_dir, mask_file[0])
                image_path = os.path.join(self.img_dir, img_file)
                self.dataset += [[image_path, mask_path]]
               

    def ReadImage(self, path) -> torch.Tensor:
        return torchvision.io.read_image(path, torchvision.io.ImageReadMode.RGB) # C x H x W
    
        # return Image.open(path).convert("RGB")


    def ApplyTransforms(self, image, mask):
        if self.transforms:
            image = self.transforms(image)

        if self.mask_transforms:
            mask = self.mask_transforms(mask) 

        return image, mask


    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx):
        img_path, mask_path = self.dataset[idx]
        image = self.ReadImage(img_path)
        mask = self.ReadImage(mask_path)
        print(mask_path)
        image, mask = self.ApplyTransforms(image, mask)
        return image, mask
    


# Load
DATASET = CustomSegmentationDataset(
    img_dir=image_dir,
    mask_dir=mask_dir,
    transforms=TRANSFORMS,
    mask_transforms=MASK_TRANSFORMS
)

data = "data/data/masks/00500.png"
data = torchvision.io.read_image(data, torchvision.io.ImageReadMode.RGB)


result = MASK_TRANSFORMS(data)
print(result.unique())


# TRAIN_LOADER = DataLoader(DATASET, batch_size=4, shuffle=False)


# for (images, masks) in DATASET:
#     print(masks.unique())



    # plt.figure(figsize=(10, 10))
    # plt.imshow(masks.numpy())
    # plt.show()

