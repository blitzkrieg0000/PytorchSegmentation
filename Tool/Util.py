from matplotlib import pyplot as plt
import torch


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
        mask = torch.all(image == value, dim=-1)
        newMask[mask] = int(key)

    return newMask
