import os, cv2, pandas as pd
from tqdm import tqdm

CSV = "labely/brandx/dataset/annotations.csv"
IMG_DIR = "labely/brandx/dataset/images"
OUT_DIR = "labely/brandx/dataset/train"

df = pd.read_csv(CSV)

for i,row in tqdm(df.iterrows(), total=len(df)):
    img = cv2.imread(os.path.join(IMG_DIR, row.image))
    x1,y1,x2,y2 = int(row.xmin),int(row.ymin),int(row.xmax),int(row.ymax)

    crop = img[y1:y2, x1:x2]
    label_dir = os.path.join(OUT_DIR, row.label)
    os.makedirs(label_dir, exist_ok=True)
    cv2.imwrite(os.path.join(label_dir, f"{i}.jpg"), crop)
