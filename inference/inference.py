import torch
import numpy as np
import cv2
import json
from PIL import Image
from torchvision import models, transforms
import torch
import numpy as np
from sam3.sam3.model_builder import build_sam3_image_model
from sam3.sam3.model.sam3_image_processor import Sam3Processor

# ---- Load SAM3 once ----
device = "cuda" if torch.cuda.is_available() else "cpu"

sam_model = build_sam3_image_model(
    bpe_path="sam3/assets/bpe_simple_vocab_16e6.txt.gz",
    checkpoint_path="sam3_weights/sam3.pt"
).to(device).eval()

processor = Sam3Processor(sam_model)


def sam3_predict(image_rgb, prompt="logos"):
    """
    Runs SAM3 and returns list of binary masks (numpy arrays)
    """
    state = processor.set_image(image_rgb)

    outputs = sam_model.predict_inst(
        state,
        prompts=[prompt],
        conf_thresh=0.35
    )

    masks = outputs["masks"]  # Tensor [N, H, W]
    return masks.cpu().numpy()

# ===================== CONFIG =====================
IMAGE_PATH = "test.jpg"
MODEL_PATH = "labely/brandx/brandx_resnet18.pth"
CLASS_PATH = "labely/brandx/classes.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===================== UTIL =====================

def mask_to_bbox(mask):
    """Convert binary mask → bounding box"""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return xs.min(), ys.min(), xs.max(), ys.max()

# ===================== LOAD CLASSES =====================

with open(CLASS_PATH) as f:
    classes = json.load(f)

NUM_CLASSES = len(classes)

# ===================== LOAD CLASSIFIER =====================

model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ===================== LOAD IMAGE =====================

image = cv2.imread(IMAGE_PATH)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# ===================== RUN SAM3 =====================

# YOU must replace this with your SAM3 inference call
# Expected output: list of binary masks (numpy arrays)
sam3_masks = sam3_predict(image_rgb, prompt="logos")

# ===================== CLASSIFY REGIONS =====================

for mask in sam3_masks:
    bbox = mask_to_bbox(mask)
    if bbox is None:
        continue

    x1, y1, x2, y2 = bbox
    crop = image_rgb[y1:y2, x1:x2]

    pil_crop = Image.fromarray(crop)
    inp = transform(pil_crop).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model(inp).argmax(1).item()

    label = classes[pred]

    # Draw box + label
    cv2.rectangle(image, (x1,y1), (x2,y2), (0,255,0), 2)
    cv2.putText(image, label, (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

# ===================== SHOW RESULT =====================

cv2.imshow("BrandX Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
