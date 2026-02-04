import os
import xml.etree.ElementTree as ET
import cv2

DATA_DIR = "dataset/logodet"   # folder with images + xml
OUT_IMG = "dataset/yolo/images"
OUT_LBL = "dataset/yolo/labels"

os.makedirs(OUT_IMG, exist_ok=True)
os.makedirs(OUT_LBL, exist_ok=True)

class_map = {"logo": 0}  # class-agnostic detector

for file in os.listdir(DATA_DIR):
    if not file.endswith(".xml"):
        continue

    tree = ET.parse(os.path.join(DATA_DIR, file))
    root = tree.getroot()

    img_name = root.find("filename").text
    img_path = os.path.join(DATA_DIR, img_name)
    img = cv2.imread(img_path)
    h, w, _ = img.shape

    cv2.imwrite(os.path.join(OUT_IMG, img_name), img)

    with open(os.path.join(OUT_LBL, img_name.replace(".jpg",".txt")), "w") as f:
        for obj in root.findall(".//object"):
            box = obj.find("bndbox")
            x1 = int(box.find("xmin").text)
            y1 = int(box.find("ymin").text)
            x2 = int(box.find("xmax").text)
            y2 = int(box.find("ymax").text)

            xc = ((x1+x2)/2)/w
            yc = ((y1+y2)/2)/h
            bw = (x2-x1)/w
            bh = (y2-y1)/h

            f.write(f"0 {xc} {yc} {bw} {bh}\n")

print("Converted dataset.")
