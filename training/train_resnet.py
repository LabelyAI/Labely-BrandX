import os
import torch
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch import nn, optim

DATA_DIR = r"D:\LabelySAM\BrandX\dataset\brandx_ready"
CHECKPOINT_PATH = r"D:\LabelySAM\BrandX\models\brandx_checkpoint.pth"
BEST_MODEL_PATH = r"D:\LabelySAM\BrandX\models\brandx_resnet18.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- AUGMENTATION ----------------
transform_train = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.RandomResizedCrop(224, scale=(0.6,1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.4,0.4,0.4),
    transforms.RandomGrayscale(p=0.1),
    transforms.GaussianBlur(3),
    transforms.ToTensor()
])

transform_val = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

train_data = datasets.ImageFolder(DATA_DIR+"/train", transform_train)
val_data   = datasets.ImageFolder(DATA_DIR+"/val", transform_val)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
val_loader   = torch.utils.data.DataLoader(val_data, batch_size=32)

# ---------------- MODEL ----------------
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, len(train_data.classes))
model.to(device)

# ---------------- FREEZE BACKBONE (Phase 1) ----------------
for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.3)

# ---------------- RESUME SUPPORT ----------------
start_epoch = 0
best_acc = 0

if os.path.exists(CHECKPOINT_PATH):
    print("Resuming model weights...")
    ckpt = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(ckpt['model_state'])
    start_epoch = ckpt['epoch'] + 1


# ---------------- TRAINING ----------------
EPOCHS = 20

for epoch in range(start_epoch, EPOCHS):

    # 🔁 Unfreeze backbone after 5 epochs
    if epoch == 5:
        print("Unfreezing backbone...")
        for param in model.layer4.parameters():
            param.requires_grad = True
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    model.train()
    total_loss = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        out = model(imgs)
        loss = criterion(out, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    scheduler.step()

    print(f"Epoch {epoch} Train Loss: {total_loss/len(train_loader):.4f}")

    # -------- VALIDATION --------
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total
    print(f"Epoch {epoch} Val Accuracy: {val_acc:.4f}")

    # -------- SAVE CHECKPOINT --------
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optim_state': optimizer.state_dict()
    }, CHECKPOINT_PATH)

    # -------- SAVE BEST MODEL --------
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print("🔥 New best model saved.")

print("Training complete.")
