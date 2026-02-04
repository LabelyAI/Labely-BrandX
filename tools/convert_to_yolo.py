import os
import pandas as pd
import cv2
from tqdm import tqdm

CSV = "../dataset/annotations/annotations.csv"
IMG_DIR = "../dataset/raw/images"
OUT_IMG = "../dataset/yolo/images"
OUT_LBL = "../dataset/yolo/labels"

os.makedirs(OUT_IMG, exist_ok=True)
os.makedirs(OUT_LBL, exist_ok=True)

df = pd.read_csv(CSV)
classes = {c:i for i,c in enumerate(df.label.unique())}

for img_name, group in tqdm(df.groupby("image")):
    img_path = os.path.join(IMG_DIR, img_name)
    img = cv2.imread(img_path)
    h,w,_ = img.shape

    # copy image
    cv2.imwrite(os.path.join(OUT_IMG, img_name), img)

    # write YOLO label file
    with open(os.path.join(OUT_LBL, img_name.replace(".jpg",".txt")), "w") as f:
        for _,row in group.iterrows():
            x_center = ((row.xmin + row.xmax)/2)/w
            y_center = ((row.ymin + row.ymax)/2)/h
            width = (row.xmax-row.xmin)/w
            height = (row.ymax-row.ymin)/h

            f.write(f"{classes[row.label]} {x_center} {y_center} {width} {height}\n")

print("YOLO dataset ready.")
