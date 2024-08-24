import os

from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import Accuracy, ConfusionMatrix, F1Score, Precision, Recall
import torchvision
from torchvision.transforms import v2 as transformsv2, InterpolationMode
from torch.utils.data import Dataset
from PIL import Image
from torch.utils.data import DataLoader, random_split
from Tool.Util import GetUniqueRGBPixels, RGBMaskToColorMap, RemoveAntialiasing

# =================================================================================================================== #
#! PARAMS
# =================================================================================================================== #
# DATASET
image_dir = "./data/data/images"
mask_dir = "./data/data/masks"
CLASS_NAMES = ["background", "disease"]
COLOR_MAPS = {
    0: [0, 0, 0],   # Arkaplan Pikselleri
    1: [120, 0, 0]  # Hastalıklı Piksel
}
NUM_CLASSES = len(COLOR_MAPS)

# MODEL
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
EPOCH = 2



# =================================================================================================================== #
#! TOOL
# =================================================================================================================== #
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
    

TRANSFORMS = transformsv2.Compose([
    transformsv2.ToImage(),
    transformsv2.Resize((256, 256), antialias=False, interpolation=InterpolationMode.NEAREST) # Yeniden Boyutlandırırken Ara Pikseller Doldurulunca Maskenin bozulmaması için NEAREST yöntemi kullanıldı.
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


# =================================================================================================================== #
#! Load Dataset
# =================================================================================================================== #
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
    

    def number_of_classes(self):
       for file in self.mask_files:
            mask_path = os.path.join(mask_dir, file)
            mask = Image.open(mask_path).convert("L")
            self.unique_values.update(np.unique(mask))
       return len(self.unique_values)



# Load
DATASET = CustomSegmentationDataset(
    img_dir=image_dir,
    mask_dir=mask_dir,
    transforms=TRANSFORMS,
    mask_transforms=MASK_TRANSFORMS
)


# Split
val_ratio = 0.1
test_ratio = 0.2
train_ratio = 1-val_ratio-test_ratio

train_dataset, validation_dataset, test_dataset = random_split(DATASET, [train_ratio, val_ratio, test_ratio])

TRAIN_LOADER = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
VAL_LOADER = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False)
TEST_LOADER = DataLoader(test_dataset, batch_size=1, shuffle=False)

#* DATALOADER TEST
# for (images, masks) in TRAIN_LOADER:  # Görüntü ve maske yollarını al
#     images, masks = images.to(DEVICE, dtype=torch.float32), masks.to(DEVICE, dtype=torch.int64)
#     print(masks.unique())


# =================================================================================================================== #
#! Create Model
# =================================================================================================================== #
class SimpleSegmentation(nn.Module):
    def __init__(self, num_classes):
        super(SimpleSegmentation, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, num_classes, kernel_size=2, stride=2)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    





# =================================================================================================================== #
#! Compile Model
# =================================================================================================================== #
model = SimpleSegmentation(num_classes=NUM_CLASSES)
model = model.to(DEVICE)
model.train()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)



# =================================================================================================================== #
#! TRAIN
# =================================================================================================================== #
images: torch.Tensor
masks: torch.Tensor
outputs: torch.Tensor
for epoch in range(EPOCH):
    running_loss = 0.0
    
    # Train
    model.train()
    for (images, masks) in TRAIN_LOADER:  # Görüntü ve maske yollarını al
        images, masks = images.to(DEVICE, dtype=torch.float32), masks.to(DEVICE, dtype=torch.int64)

        # plt.figure(figsize=(10, 10))
        # plt.imshow(masks[0].cpu().numpy())
        # plt.show()

        # Hafızadaki Gradyantlerı sıfırla
        optimizer.zero_grad()

        # Forward
        outputs = model(images)

        # Loss
        # Loss motodu ile tahmin edilen görüntü (PREDICTION) ile görüntü (LABEL) kıyaslar.
        # Pytorchta Loss metotları segmentasyon için her tahmin edilen 4D tensorü (N x C x H x W) ile 3D maske tensorü (N x H x W) karşılaştırılır.
        # Her sınıf için ayrı bir "C" boyutu vardır. İki sınıf varsa, iki adet "C" boyutu vardır. Yani her piksel için ayrı sınıf olasılıkları hesaplanır.
        # Örnekte Cross Entropy Loss metodu, kullanılmıştır. Bu metot (N x C x H x W)' a önce softmax uygular. Bu yüzden modelin sonunda softmax uygulanmasına gerek yoktur.
        # PREDICTION(float)(BATCH x CLASSES x H x W)  <- X ->  LABEL(int64(long)) (BATCH x H x W)
        loss = criterion(outputs, masks)   

        # Backward
        loss.backward()

        # Optimize
        optimizer.step()

        # prediction = outputs.argmax(dim=1)
        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCH}], Loss: {running_loss/len(TRAIN_LOADER):.4f}")

    # Validation
    model.eval()
    predictions = []
    predictionGroundTruths = []
    with torch.no_grad():
        for images, masks in VAL_LOADER:
            images, masks = images.to(DEVICE, dtype=torch.float32), masks.to(DEVICE, dtype=torch.int64)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            predictions += [preds]
            predictionGroundTruths += [masks]



# TEST
predictions = []
predictionGroundTruths = []
with torch.no_grad():
    for images, masks in TEST_LOADER:
        images, masks = images.to(DEVICE, dtype=torch.float32), masks.to(DEVICE, dtype=torch.int64)
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        predictions += [preds]
        predictionGroundTruths += [masks]



y_pred = torch.cat(predictions).cpu()
y_gt = torch.cat(predictionGroundTruths).cpu()
confmat = ConfusionMatrix(task="multiclass", num_classes=NUM_CLASSES, normalize="true")
confmat = confmat.to("cpu")
confmat(y_pred, y_gt)
cm = confmat.compute()
print(cm)

import seaborn as sb
ax = plt.subplot()
sb.heatmap(cm.numpy(), ax=ax, annot=True, fmt=".4f", cmap="seismic", xticklabels=range(NUM_CLASSES), yticklabels=range(NUM_CLASSES))
ax.set_xlabel("Predicted Labels")
ax.set_ylabel("True Labels")
ax.set_title("Normalized Confusion Matrix")
plt.show()

exit()
accuracy = Accuracy(task="multiclass",num_classes=NUM_CLASSES)
acc = accuracy(y_pred, y_true.argmax(dim=1))

precision = Precision(task="multiclass",num_classes=NUM_CLASSES, average=None)
prec = precision(y_pred, y_true.argmax(dim=1))

recall = Recall(task="multiclass",num_classes=NUM_CLASSES, average=None)
rec = recall(y_pred,y_true.argmax(dim=1))

f1score = F1Score(task="multiclass",num_classes=NUM_CLASSES, average=None)
f1 = f1score(y_pred, y_true.argmax(dim=1))

for i in range(NUM_CLASSES):
   print(f"Class {i} - Precision: {prec[i].item():.4f}")
   print(f"Class {i} - Recall: {rec[i].item():.4f}")
   print(f"Class {i} - F1 Score: {f1[i].item():.4f}")

print(f"Overall Accuracy: {acc:.4f}")    








    






