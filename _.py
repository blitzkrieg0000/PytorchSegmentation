import os

import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2 as transformsv2

from Tool.Util import GetUniqueRGBPixels, RGBMaskToColorMap, RemoveAntialiasing

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") 


image_dir = "./data/data/images"
mask_dir = "./data/data/masks"


class MaskTransforms():
    def __init__(self, color_maps:dict[int, torch.Tensor]={}, override_colors=False, channel_first=False, rm_aa_thres=15, remove_aa=True):
        self.ColorMaps = color_maps
        self.ChannelFirst = channel_first
        self.RemoveAntialiasingThreshold = rm_aa_thres
        self.RemoveAntialiasing = remove_aa
        self.OverrideColors = override_colors

    def __call__(self, image: torch.Tensor):
        if self.ChannelFirst:
            image = image.permute(1, 2, 0)

        if self.RemoveAntialiasing:
            image = RemoveAntialiasing(image, target_colors=self.ColorMaps, threshold=15)

        if self.OverrideColors:
            maps = self.ColorMaps
        else:
            maps = GetUniqueRGBPixels(image)
            maps.update(self.ColorMaps)

        newMask = RGBMaskToColorMap(image, maps)
        return newMask
    


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
        image, mask = self.ApplyTransforms(image, mask)
        return image, mask
    


COLOR_MAPS = {
    0: [0, 0, 0],
    1: [120, 0, 0]
}

TRANSFORMS = transformsv2.Compose([
    transformsv2.ToImage(),
    transformsv2.Resize((256, 256), antialias=False, interpolation=InterpolationMode.NEAREST) # Maskenin bozulmaması için NEAREST
])

MASK_TRANSFORMS = transformsv2.Compose([
    TRANSFORMS,
    MaskTransforms(
        channel_first=True,
        remove_aa=True,
        rm_aa_thres=15,
        override_colors=True,
        color_maps=COLOR_MAPS
    )
])

# Load
DATASET = CustomSegmentationDataset(
    img_dir=image_dir,
    mask_dir=mask_dir,
    transforms=TRANSFORMS,
    mask_transforms=MASK_TRANSFORMS
)


TRAIN_LOADER = DataLoader(DATASET, batch_size=8, shuffle=True)


for image, mask in TRAIN_LOADER:
    print(mask.unique())




# data = "data/data/masks/00500.png"
# data = torchvision.io.read_image(data, torchvision.io.ImageReadMode.RGB)
# rdata = RemoveAntialiasing(data, targetColors, threshold=15, channel_first=True)
# maps = GetUniqueRGBPixels(rdata, channel_first=False)
# print(maps)
# # result = MASK_TRANSFORMS(data)
# # print(result.unique())

# fig, ax = plt.subplots(1, 2)
# ax[0].imshow(data.permute(1, 2, 0).numpy())  # data.permute(1, 2, 0).numpy()
# ax[1].imshow(rdata.numpy())  # data.permute(1, 2, 0).numpy()
# plt.show()

