import os
from PIL import Image, ImageDraw
import random 
import numpy as np
import matplotlib.pyplot as plt

#python3 train.py --img 256 --cfg yolov5m.yaml --hyp hyp.scratch.yaml --batch 32 --epochs 100 --data bloodcelldetection.yaml --weights yolov5m.pt --workers 24 --name yolov5m_BC_det

#python3 detect.py --source ../yolov5/images/test/ --weights runs/train/yolov5m_BC_det/weights/best.pt --conf 0.25 --name yolov5m_BC_det

#python3 val.py --weights runs/train/yolo_BC_det4/weights/best.pt --data bloodcelldetection.yaml --task test --name yolo_det4

detections_dir = "runs/detect/yolo_BC_det2/"
detection_images = [os.path.join(detections_dir, x) for x in os.listdir(detections_dir)]

random_detection_image = Image.open(random.choice(detection_images))
plt.imshow(np.array(random_detection_image))