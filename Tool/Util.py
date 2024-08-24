from matplotlib import pyplot as plt
import numpy as np
import torch




def RemoveAntialiasing(mask, target_colors:dict[int, torch.Tensor]={}, threshold=15, channel_first=False):
    if isinstance(mask, np.ndarray):
        mask = torch.tensor(mask)

    if channel_first:
        mask = mask.permute(1, 2, 0)

    cleanMask = torch.zeros_like(mask)

    for color in target_colors.values():
        diff = torch.abs(mask - torch.tensor(color, dtype=mask.dtype))
        diff_sum = diff.sum(dim=-1)
        cleanMask[diff_sum < threshold] = torch.tensor(color, dtype=mask.dtype)

    if channel_first:
        cleanMask = cleanMask.permute(2, 0, 1)

    return cleanMask


def GetUniqueRGBPixels(image:torch.Tensor, channel_first:bool=False):
    """
        Çok katmanlı resimdeki (RGB gibi) benzersiz renklerin bulunması için kullanılır.

        args:
            image: H x W X C şeklinde tensor
            channel_first: C x H x W şeklinde tensor verilecekse belirtir
            
        return:
            dict[int, torch.Tensor]
    """

    if channel_first: # To Channel Last: C x H x W => H x W x C
        image = image.permute(1, 2, 0)
        

    reshaped = image.view(-1, image.shape[-1])
    print(reshaped.stride())
    uniques = reshaped.unique(dim=0)
    maps = {index: val for index, val in enumerate(uniques)}
    return maps


def RGBMaskToColorMap(image:torch.Tensor, map:dict[int, torch.Tensor], channel_first:bool=False):
    """
        Verilen eşleştirme haritasına göre verilen görseli colormap'e dönüştürür.
        
        args:
            image: H x W X C şeklinde tensor
            map: dict[int, torch.Tensor] şeklinde bir renk eşleştirme haritası
            channel_first: C x H x W şeklinde tensor verilecekse belirtir
            
        return:
            torch.Tensor H x W şeklinde bir tensor
    """
    if channel_first:
        image = image.permute(1, 2, 0)

    newMask = torch.zeros((image.shape[0], image.shape[1]), dtype=torch.int64)
    for key, value in map.items():
        mask = torch.all(image == torch.tensor(value), dim=-1)
        newMask[mask] = int(key)

    return newMask
