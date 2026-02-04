import torch
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch import nn, optim

DATA_DIR = r"D:\LabelySAM\BrandX\dataset\brandx_ready"  # padded crops
import os
os.makedirs("../models", exist_ok=True)

train_tfms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomPerspective(distortion_scale=0.5),
    transforms.RandomRotation(25),
    transforms.ColorJitter(0.4,0.4,0.4),
    transforms.ToTensor(),              # <-- convert here
    transforms.RandomErasing(p=0.5),    # tensor-only transform
])


val_tfms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

train_ds = datasets.ImageFolder(DATA_DIR+"/train", train_tfms)
val_ds   = datasets.ImageFolder(DATA_DIR+"/val", val_tfms)

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader   = torch.utils.data.DataLoader(val_ds, batch_size=32)

model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
model.classifier[1] = nn.Linear(1280, len(train_ds.classes))

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=3e-4)

best_acc = 0

for epoch in range(20):
    model.train()
    for imgs,labels in train_loader:
        imgs,labels = imgs.to(device),labels.to(device)
        out = model(imgs)
        loss = criterion(out,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # validation
    model.eval()
    correct,total=0,0
    with torch.no_grad():
        for imgs,labels in val_loader:
            imgs,labels = imgs.to(device),labels.to(device)
            preds = model(imgs).argmax(1)
            correct += (preds==labels).sum().item()
            total += labels.size(0)

    acc = correct/total
    print(f"Epoch {epoch} Val Acc: {acc:.3f}")

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "../models/brandx_efficientnet.pth")
        print("Saved best model.")
