from torchvision import datasets
import json

train_dir = r"D:\LabelySAM\BrandX\dataset\brandx_ready\train"

dataset = datasets.ImageFolder(train_dir)

classes = dataset.classes  # list of folder names (brands)

print("Classes:", classes)

with open(r"D:\LabelySAM\BrandX\classes.json", "w") as f:
    json.dump(classes, f)

print("Saved to classes.json")
