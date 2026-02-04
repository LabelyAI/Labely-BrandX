import os
import cv2
import random
import xml.etree.ElementTree as ET
from tqdm import tqdm
from sklearn.model_selection import train_test_split

DATA_DIR = r"D:\LabelySAM\BrandX\dataset\LogoDet-3K\clothes"
OUT_DIR  = r"D:\LabelySAM\BrandX\dataset\brandx_ready"

random.seed(42)
data = []

print("Scanning dataset...")

# ---------- gather crop metadata only ----------
for root_dir, _, files in os.walk(DATA_DIR):
    for file in files:
        if not file.endswith(".xml"):
            continue

        xml_path = os.path.join(root_dir, file)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        img_name = root.find("filename").text
        img_path = os.path.join(root_dir, img_name)

        if not os.path.exists(img_path):
            continue

        for obj in root.findall(".//object"):
            label = obj.find("name").text.strip()
            box = obj.find("bndbox")

            x1 = int(box.find("xmin").text)
            y1 = int(box.find("ymin").text)
            x2 = int(box.find("xmax").text)
            y2 = int(box.find("ymax").text)

            data.append((img_path, label, x1, y1, x2, y2))

print("Total logo instances:", len(data))

# ---------- split ----------
train, test = train_test_split(data, test_size=0.2, random_state=42)
train, val  = train_test_split(train, test_size=0.2, random_state=42)

splits = {"train": train, "val": val, "test": test}

# ---------- write crops directly ----------
for split, items in splits.items():
    print(f"Writing {split} set...")
    for i, (img_path, label, x1, y1, x2, y2) in enumerate(tqdm(items)):
        img = cv2.imread(img_path)
        crop = img[y1:y2, x1:x2]

        label_dir = os.path.join(OUT_DIR, split, label)
        os.makedirs(label_dir, exist_ok=True)

        out_path = os.path.join(label_dir, f"{split}_{i}.jpg")
        cv2.imwrite(out_path, crop)

print("✅ Dataset prepared successfully at:", OUT_DIR)
