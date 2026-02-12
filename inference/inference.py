import cv2
import torch
import os
import numpy as np
from ultralytics import YOLO
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
from torch.nn import functional as F
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# ================= SETTINGS =================
PERSON_MODEL = r"D:\LabelySAM\BrandX\models\yolov8n.pt"
LOGO_MODEL   = r"D:\LabelySAM\BrandX\models\logos_yolo.pt"
IMAGE_PATH   = "dff.jpg"
REFERENCE_FOLDER = r"D:\LabelySAM\BrandX\my_logo"

SIM_THRESHOLD = 0.45
device = "cuda" if torch.cuda.is_available() else "cpu"

# ================= LOAD MODELS =================
person_model = YOLO(PERSON_MODEL)
logo_model   = YOLO(LOGO_MODEL)

model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
model.classifier = torch.nn.Identity()
model.to(device).eval()

tfm = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ================= EMBEDDING =================
def get_embedding(crop):
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    x = tfm(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(x)
    emb = F.normalize(emb, dim=1)
    return emb.cpu().numpy()

# ================= REFERENCE EMBEDDINGS =================
ref_embeddings = []
for file in os.listdir(REFERENCE_FOLDER):
    path = os.path.join(REFERENCE_FOLDER, file)
    img = cv2.imread(path)
    if img is None:
        continue
    ref_embeddings.append(get_embedding(img))

if len(ref_embeddings) == 0:
    raise ValueError("No reference images found")

ref_embeddings = np.vstack(ref_embeddings)
reference_embedding = np.mean(ref_embeddings, axis=0, keepdims=True)
reference_embedding = reference_embedding / np.linalg.norm(reference_embedding)

print("Reference embedding ready.")

# ================= LOAD IMAGE =================
img = cv2.imread(IMAGE_PATH)
output_img = img.copy()

all_embeddings = []
labels_for_plot = []

# ================= PERSON DETECTION =================
person_results = person_model(img, conf=0.4)

for r in person_results:
    for box, cls in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy()):
        if int(cls) != 0:
            continue

        px1,py1,px2,py2 = map(int, box)
        cv2.rectangle(output_img,(px1,py1),(px2,py2),(255,0,0),2)

        person_crop = img[py1:py2, px1:px2]

        # ================= LOGO DETECTION =================
        logo_results = logo_model(person_crop, conf=0.25)

        for lr in logo_results:
            for lbox in lr.boxes.xyxy.cpu().numpy():

                lx1,ly1,lx2,ly2 = map(int,lbox)
                gx1 = px1 + lx1
                gy1 = py1 + ly1
                gx2 = px1 + lx2
                gy2 = py1 + ly2

                logo_crop = person_crop[ly1:ly2, lx1:lx2]
                if logo_crop.size == 0:
                    continue

                emb = get_embedding(logo_crop)
                similarity = np.dot(reference_embedding, emb.T).item()

                all_embeddings.append(emb.squeeze())
                labels_for_plot.append("match" if similarity > SIM_THRESHOLD else "other")

                if similarity > SIM_THRESHOLD:
                    color = (0,255,0)
                    label = f"MATCH {similarity:.2f}"
                else:
                    color = (0,0,255)
                    label = f"NO {similarity:.2f}"

                cv2.rectangle(output_img,(gx1,gy1),(gx2,gy2),color,2)
                cv2.putText(output_img,label,(gx1,gy1-5),
                            cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)

cv2.imwrite("final_output.jpg", output_img)
print("Saved final_output.jpg")
# ================= INTERPRETABLE EMBEDDING VISUALIZATION =================

if len(all_embeddings) > 0:

    # Stack everything
    ref_embs = ref_embeddings
    det_embs = np.vstack(all_embeddings)

    # Combine for PCA
    X = np.vstack([ref_embs, det_embs])

    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    # Split back
    ref_2d = X_2d[:len(ref_embs)]
    det_2d = X_2d[len(ref_embs):]

    centroid_2d = np.mean(ref_2d, axis=0)

    plt.figure(figsize=(8,8))

    # Plot reference images
    plt.scatter(ref_2d[:,0], ref_2d[:,1],
                c='blue', s=100, label='Reference Images')

    # Plot centroid
    plt.scatter(centroid_2d[0], centroid_2d[1],
                c='black', s=200, marker='X', label='Reference Centroid')

    # Plot detected logos
    for i in range(len(det_2d)):
        if labels_for_plot[i] == "match":
            plt.scatter(det_2d[i,0], det_2d[i,1],
                        c='green', s=120, label='Match' if i==0 else "")
        else:
            plt.scatter(det_2d[i,0], det_2d[i,1],
                        c='red', s=120, label='No Match' if i==0 else "")

        # draw line to centroid (visualizing similarity)
        plt.plot([centroid_2d[0], det_2d[i,0]],
                 [centroid_2d[1], det_2d[i,1]],
                 linestyle='dashed', alpha=0.4)

    # Draw decision boundary circle
    # approximate boundary radius from threshold
    distances = np.linalg.norm(ref_2d - centroid_2d, axis=1)
    boundary_radius = np.mean(distances) * 1.5

    circle = plt.Circle(centroid_2d, boundary_radius,
                        fill=False, linestyle='--',
                        color='gray', label='Decision Boundary')
    plt.gca().add_patch(circle)

    plt.title("Reference Learning & Decision Visualization")
    plt.legend()
    plt.grid(True)

    plt.savefig("embedding_decision_plot.png")
    plt.show()
