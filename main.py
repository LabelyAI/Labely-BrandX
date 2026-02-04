import torch
from torchvision import models, transforms
from PIL import Image
import json

# ---------- Paths ----------
MODEL_PATH = r"D:\LabelySAM\BrandX\models\brandx_resnet18.pth"
CLASS_PATH = r"D:\LabelySAM\BrandX\models\classes.json"
IMAGE_PATH = "adidas.jpg"

# ---------- Load classes ----------
with open(CLASS_PATH) as f:
    classes = json.load(f)

NUM_CLASSES = len(classes)

# ---------- Model ----------
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# ---------- Image ----------
img = Image.open(IMAGE_PATH)

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

inp = transform(img).unsqueeze(0).to(device)

# ---------- Predict ----------
with torch.no_grad():
    logits = model(inp)
    probs = torch.softmax(logits, dim=1)
    conf, pred = torch.max(probs, dim=1)

label = classes[pred.item()]
confidence = conf.item()

print(f"Predicted brand: {label}")
print(f"Confidence: {confidence:.3f}")
