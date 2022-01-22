import pandas as pd
import os
from sklearn.model_selection import train_test_split
import shutil

'''import torch
from IPython.display import Image  # for displaying images
import os 
import random
import shutil
import xml.etree.ElementTree as ET
from xml.dom import minidom
from tqdm import tqdm
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt'''

#random.seed(123)

# Load data
data = pd.read_csv('annotations.csv',engine='python')

# Display 5 random samples
#print(data.sample(5))

# Split data into train and test sets in 8:2 ratio
images = data.image.unique()
#train = images[:80]
#test = images[80:]
#print(test)
d = data[data['image'].isin({'image-25.png'})]
#print(d.image.values[0])



def processData(data):
    # Initialise the info dict 
    info_dict = {}
    info_dict['bboxes'] = []
    info_dict['filename'] = data.image.values[0]
    info_dict['image_size'] = tuple([256,256,3])
    
    for row in data.itertuples():
        # Get details of the bounding box 
        bbox = {}
        bbox["class"] = row.label
        bbox["xmin"] = row.xmin
        bbox["ymin"] = row.ymin
        bbox["xmax"] = row.xmax
        bbox["ymax"] = row.ymax
        info_dict['bboxes'].append(bbox)
    return info_dict

# Dictionary that maps class names to IDs
class_name_to_id_mapping = {"wbc": 0,
                           "rbc": 1}        
    
# Convert the info dict to the required yolo format and write it to disk
def convert_to_yolov5(info_dict):
    print_buffer = []
    
    # For each bounding box
    for b in info_dict["bboxes"]:
        try:
            class_id = class_name_to_id_mapping[b["class"]]
        except KeyError:
            print("Invalid Class. Must be one from ", class_name_to_id_mapping.keys())
        
        # Transform the bbox co-ordinates as per the format required by YOLO v5
        b_center_x = (b["xmin"] + b["xmax"]) / 2 
        b_center_y = (b["ymin"] + b["ymax"]) / 2
        b_width    = (b["xmax"] - b["xmin"])
        b_height   = (b["ymax"] - b["ymin"])
        
        # Normalise the co-ordinates by the dimensions of the image
        image_w, image_h, image_c = info_dict["image_size"]  
        b_center_x /= image_w 
        b_center_y /= image_h 
        b_width    /= image_w 
        b_height   /= image_h 
        
        #Write the bbox details to the file 
        print_buffer.append("{} {:.3f} {:.3f} {:.3f} {:.3f}".format(class_id, b_center_x, b_center_y, b_width, b_height))
        
    # Name of the file which we have to save 
    save_file_name = os.path.join("annotations", info_dict["filename"].replace("png", "txt"))
    
    # Save the annotation to disk
    print("\n".join(print_buffer), file= open(save_file_name, "w"))
    
# Convert all annotations to YOLOv5 annotation format

for image in images:
    d = data[data['image'].isin({image})]
    convert_to_yolov5(processData(d))

# Read images and annotations
images = [os.path.join('images', x) for x in os.listdir('images') if x[-3:] == "png"]
annotations = [os.path.join('annotations', x) for x in os.listdir('annotations') if x[-3:] == "txt"]

images.sort()
print(images)
annotations.sort()

# Split the dataset into train-valid-test splits 
train_images, val_images, train_annotations, val_annotations = train_test_split(images, annotations, test_size = 0.2, random_state = 1)
val_images, test_images, val_annotations, test_annotations = train_test_split(val_images, val_annotations, test_size = 0.5, random_state = 1)

print(val_images)

#Utility function to move images 
def move_files_to_folder(list_of_files, destination_folder):
    for f in list_of_files:
        try:
            shutil.move(f, destination_folder)
        except:
            print(f)
            assert False

# Move the splits into their folders
move_files_to_folder(train_images, 'images/train')
move_files_to_folder(val_images, 'images/val/')
move_files_to_folder(test_images, 'images/test/')
move_files_to_folder(train_annotations, 'annotations/train/')
move_files_to_folder(val_annotations, 'annotations/val/')
move_files_to_folder(test_annotations, 'annotations/test/')