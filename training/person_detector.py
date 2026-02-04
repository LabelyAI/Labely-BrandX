from ultralytics import YOLO
import numpy as np

class PersonDetector:
    def __init__(self, model_path="yolov8n.pt", conf=0.4):
        """
        model_path: YOLO model file
        conf: confidence threshold
        """
        self.model = YOLO(model_path)
        self.conf = conf

    def detect(self, image: np.ndarray):
        """
        image: OpenCV image (BGR numpy array)
        returns: list of person bounding boxes [(x1,y1,x2,y2), ...]
        """
        results = self.model(image, conf=self.conf)

        persons = []

        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()

            for box, cls in zip(boxes, classes):
                if int(cls) == 0:  # COCO class 0 = person
                    x1, y1, x2, y2 = map(int, box)
                    persons.append((x1, y1, x2, y2))

        return persons
