import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader,Dataset,random_split
from torchmetrics import ConfusionMatrix,Accuracy,Recall,Precision,F1Score
import torchvision.transforms as transforms
import torchvision.models as models
import os 
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

image_dir = "./data/data/images"
mask_dir = "./data/data/masks"

class CustomSegmentationDataset():
    def __init__(self, img_dir,mask_dir, transform = None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.dataset = []
        self.transform = transform
        self.img_file = os.listdir(img_dir)
        self.mask_file = os.listdir(mask_dir)
        self.unique_values = set()
        for img in self.img_file :
           image = os.path.join(img_dir,img)
           img_splt = img.split(".")
           for msk in self.mask_file:
               msk_splt = msk.split(".")
               if(msk_splt[0] == img_splt[0]):
                  mask = os.path.join(mask_dir,msk)
                  self.dataset.append([image,mask])
                  break
               else:
                   continue

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_name, mask_name = self.dataset[idx]
        
        image = Image.open(img_name).convert('RGB')
        mask = Image.open(mask_name).convert('RGB') 
        if self.transform:
            print("girdi")
            image = self.transform(image)
            mask = self.transform(mask) 
        return image, mask
    
    def number_of_classes(self):
       for file in self.mask_file:
            mask_path = os.path.join(mask_dir, file)
            mask = Image.open(mask_path).convert("L")
            self.unique_values.update(np.unique(mask))
       return len(self.unique_values)
               
transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor()
]) 
       
datasets = CustomSegmentationDataset(img_dir=image_dir,mask_dir=mask_dir,transform=transform)
dataset = datasets.dataset
train_data_size = int(len(dataset) * 0.8)
test_data_size = int(len(dataset) - train_data_size)

train_dataset, test_dataset = random_split(dataset, [train_data_size, test_data_size])

batch_size = 32
train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False)

x_train_list = []
y_train_list = []

for image_paths, mask_paths in train_loader:
    for img_path, mask_path in zip(image_paths, mask_paths):
        image = Image.open(img_path).convert('RGB')
        tensor_image = transform(image)
        x_train_list.append(tensor_image)

        mask = Image.open(mask_path).convert('RGB')
        tensor_mask = transform(mask)
        y_train_list.append(tensor_mask)

x_test_list = []
y_test_list = []

for image_paths, mask_paths in test_loader:
    for img_path, mask_path in zip(image_paths, mask_paths):
        image = Image.open(img_path).convert('RGB')
        tensor_image = transform(image)
        x_test_list.append(tensor_image)

        mask = Image.open(mask_path).convert('RGB')
        tensor_mask = transform(mask)
        y_test_list.append(tensor_mask)

x_train = torch.stack(x_train_list)    
y_train = torch.stack(y_train_list)
x_test = torch.stack(x_test_list)
y_test = torch.stack(y_train_list) 

num_classes = datasets.number_of_classes()
print (num_classes)

class SimplerSegmentation(nn.Module):
    def __init__(self, num_classes):
        super(SimplerSegmentation, self).__init__()
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
    

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
model = SimplerSegmentation(num_classes=num_classes)
model = model.to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for image_paths, mask_paths in train_loader:  # Görüntü ve maske yollarını al
        images = []
        masks = []
        for img_path, mask_path in zip(image_paths, mask_paths): # Her bir batch'deki yollar üzerinde yinele
            image = Image.open(img_path).convert('RGB')
            tensor_image = transform(image).to(DEVICE)
            images.append(tensor_image)
            
            mask = Image.open(mask_path).convert('RGB')
            tensor_mask = transform(mask).to(DEVICE)
            masks.append(tensor_mask)
        
        # Görüntüleri ve maskeleri tensörlere yığınla
        images = torch.stack(images)
        masks = torch.stack(masks)
        
        optimizer.zero_grad()
        # print(images.shape) # Görüntü tensörünün şeklini kontrol et
        outputs = model(images)  # Şimdi görüntü tensörlerini modele geçir
        loss = criterion(outputs, masks.argmax(dim=1))
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

#model.eval()
#
#all_preds = []
#with torch.no_grad():
#    for image_paths, mask_paths in train_loader: 
#        images = []
#        masks = []
#        for img_path, mask_path in zip(image_paths, mask_paths): 
#            image = Image.open(img_path).convert('RGB')
#            tensor_image = transform(image).to(DEVICE)
#            images.append(tensor_image)
#            
#            mask = Image.open(mask_path).convert('RGB')
#            tensor_mask = transform(mask).to(DEVICE)
#            masks.append(tensor_mask)
#        
#        
#        images = torch.stack(images)
#        masks = torch.stack(masks)
#        outputs = model(images)
#        _, preds = torch.max(outputs, 1)
#        all_preds.append(preds)
#        
#y_pred = torch.cat(all_preds).cpu()
#y_true = y_test.cpu()
#
#confmat = ConfusionMatrix(task="multiclass",num_classes=num_classes)
#confmat = confmat.to('cpu')
#confmat(y_pred,y_true.argmax(dim=1))
#cm = confmat.compute()
#print(cm)
#
#accuracy = Accuracy(task="multiclass",num_classes=num_classes)
#acc = accuracy(y_pred, y_true.argmax(dim=1))
#
#precision = Precision(task="multiclass",num_classes=num_classes, average=None)
#prec = precision(y_pred, y_true.argmax(dim=1))
#
#recall = Recall(task="multiclass",num_classes=num_classes, average=None)
#rec = recall(y_pred,y_true.argmax(dim=1))
#
#f1score = F1Score(task="multiclass",num_classes=num_classes, average=None)
#f1 = f1score(y_pred, y_true.argmax(dim=1))
#
#for i in range(num_classes):
#    print(f'Class {i} - Precision: {prec[i].item():.4f}')
#    print(f'Class {i} - Recall: {rec[i].item():.4f}')
#    print(f'Class {i} - F1 Score: {f1[i].item():.4f}')
#
#print(f'Overall Accuracy: {acc:.4f}')    
#
#






    






