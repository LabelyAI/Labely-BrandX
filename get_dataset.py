import kagglehub
import shutil
import os

# Desired project path
TARGET_DIR = "D:\LabelySAM\BrandX\dataset"
os.makedirs(TARGET_DIR, exist_ok=True)

# Download dataset (to cache)
download_path = kagglehub.dataset_download("lyly99/logodet3k")

print("Downloaded to:", download_path)

# Move files into your project folder
for item in os.listdir(download_path):
    src = os.path.join(download_path, item)
    dst = os.path.join(TARGET_DIR, item)
    if os.path.isdir(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)

print("Dataset now in:", TARGET_DIR)
