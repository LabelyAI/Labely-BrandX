import cv2
import torch
import json
import os
import numpy as np
from ultralytics import YOLO
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image

# ================= PATHS =================
PERSON_MODEL = r"D:\LabelySAM\BrandX\models\yolov8n.pt"
LOGO_MODEL   = r"D:\LabelySAM\BrandX\models\logos_yolo.pt"
EFF_MODEL    = r"D:\LabelySAM\BrandX\models\brandx_efficientnet.pth"
CLASSES_JSON = r"D:\LabelySAM\BrandX\models\classes.json"
IMAGE_PATH   = "adidas.jpg"

device = "cuda" if torch.cuda.is_available() else "cpu"

DEBUG_DIR = "debug_outputs"
os.makedirs(DEBUG_DIR, exist_ok=True)

def save_debug(name, img):
    path = os.path.join(DEBUG_DIR, name)
    cv2.imwrite(path, img)
    print("Saved:", path)

# ================= LOAD DETECTORS =================
person_model = YOLO(PERSON_MODEL)
logo_model   = YOLO(LOGO_MODEL)

# ================= LOAD CLASSES =================
with open(CLASSES_JSON) as f:
    classes = json.load(f)

# ================= LOAD EFFICIENTNET =================
clf = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
clf.classifier[1] = torch.nn.Linear(1280, len(classes))
clf.load_state_dict(torch.load(EFF_MODEL, map_location=device))
clf.to(device).eval()

# IMPORTANT: normalization for EfficientNet
tfm = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def classify_logo(crop):
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    x = tfm(pil).unsqueeze(0).to(device)

    with torch.no_grad():
        out = clf(x)
        prob = torch.softmax(out,1)
        conf, pred = prob.max(1)

    return classes[pred.item()], conf.item()

# ================= LOAD IMAGE =================
pil_img = Image.open(IMAGE_PATH).convert("RGB")
img = np.array(pil_img)[:, :, ::-1]

# ================= PERSON DETECTION =================
persons = []
results = person_model(img, conf=0.4)

for r in results:
    for box, cls in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy()):
        if int(cls) == 0:
            persons.append(tuple(map(int, box)))

print("Persons found:", len(persons))

# ================= PIPELINE =================
for pi,(px1,py1,px2,py2) in enumerate(persons):
    person_crop = img[py1:py2, px1:px2]
    save_debug(f"person_{pi}.jpg", person_crop)

    logo_results = logo_model(person_crop, conf=0.25)

    for li, r in enumerate(logo_results):
        for lbox in r.boxes.xyxy.cpu().numpy():
            lx1,ly1,lx2,ly2 = map(int,lbox)
            logo_crop = person_crop[ly1:ly2, lx1:lx2]

            if logo_crop.size == 0:
                continue

            save_debug(f"logo_{pi}_{li}.jpg", logo_crop)

            brand, conf = classify_logo(logo_crop)

            if conf < 0.5:
                brand = "Unknown"

            print(f"Person {pi}: {brand} ({conf:.2f})")

            display = logo_crop.copy()
            cv2.putText(display, f"{brand} {conf:.2f}",
                        (5,20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0,255,0), 2)

            save_debug(f"classified_{pi}_{li}.jpg", display)

print("Done.")
