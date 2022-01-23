import os
from PIL import Image, ImageDraw
import random 
import numpy as np
import matplotlib.pyplot as plt

python3 train.py --img 256 --cfg yolov5s.yaml --hyp hyp.scratch.yaml --batch 32 --epochs 300 --data bloodcelldetection.yaml --weights yolov5s.pt --workers 24 --name yolov5s_BC_det_300ep

python3 detect.py --img 256 --source ../yolov5/images/test/ --weights runs/train/yolov5s_BC_det_300ep/weights/best.pt --conf 0.25 --name yolov5s_BC_det_300ep

python3 val.py --img 256 --weights runs/train/yolov5s_BC_det_300ep/weights/best.pt --data bloodcelldetection.yaml --task test --name yolov5s_BC_det_300ep

detections_dir = "runs/detect/yolo_BC_det2/"
detection_images = [os.path.join(detections_dir, x) for x in os.listdir(detections_dir)]

random_detection_image = Image.open(random.choice(detection_images))
plt.imshow(np.array(random_detection_image))