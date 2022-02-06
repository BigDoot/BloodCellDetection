# BloodCellDetection
The dataset contains annotated red blood cells (RBC) and white blood cells (WBC) from peripheral blood smear taken from a light microscope. It contains 100 annotated images (256x256 pixels). Figure 1 displayed below is one of the annotated images labelled ‘image-1.png’, and Figure 2 is a list of annotations tagged to it. The annotations include the image name, the xmin/ymin/xmax/ymax values which describe the coordinates of 2 corners of each identified blood cell’s bounding box. Furthermore, each blood cell is labelled. There are 2 possible labels for each blood cell: ‘wbc’ for white blood cells and ‘rbc’ for red blood cells.

The dataset is from https://www.kaggle.com/draaslan/blood-cell-detection-dataset

![image](https://user-images.githubusercontent.com/48408342/152689717-20f77ac0-f945-46f4-90e2-6fccc618cf26.png)
![image](https://user-images.githubusercontent.com/48408342/152689718-6622c509-fa66-4d8a-b343-7c0ec736f47e.png)

The annotations are visualised in Figure 3 below. We label the white blood cells as 0 and red blood cells as 1.

![image](https://user-images.githubusercontent.com/48408342/152689728-6fc0ff01-6be9-4a82-8308-06bdf4d0a6a3.png)

Figure 3.

Dataset arrangement:
The dataset is split into training, validation and test sets in a 8:1:1 ratio using sklearn’s train_test_split

# Deep Learning Model 1 - YOLOv5

**Data flow**:
We first have to process and prepare the images and annotations to suit the specified format of the YOLOv5 deep learning model. YOLOv5 requires the annotations to be in a very specific format, in the form of a .txt file as shown in Figure 4 below. 

![image](https://user-images.githubusercontent.com/48408342/152689744-9e4a851e-51b3-4ef5-af9e-b5fb6099504a.png)

Each line represents an object (blood cell), in the format of class label, x and y coordinates (normalised by dimensions of the image) of the centre of the object’s bounding box, as well as the width and height of the object.
In order to convert the annotations in the CSV provided into this format, we perform some simple math to find the centre coordinates of the bounding boxes from the corners. For each image, we convert all the annotations into the specified .txt format. 
At this point, we split the dataset into the aforementioned train/validation/test split and move the images and annotations to the respective folders (also specified by YOLOv5 documentation and training guide).

YOLOv5 provides a few pretrained models of different sizes with varying number of parameters. We pick the YOLOv5s, which is the smallest and fastest model as our hardware capability is limited.

As YOLOv5 provide scripts for training, testing and validating the models, the process is very simple with a few commands:

![image](https://user-images.githubusercontent.com/48408342/152689761-d2bf455c-6963-46cb-8b09-2e71148104ae.png)

Detection and classification benchmark – 0.25

Hyperparameters – 300 epochs. The rest are the default.

**YOLO results:**

![image](https://user-images.githubusercontent.com/48408342/152689791-c5486628-03dc-4c3f-8dce-6e6b76e75e6d.png)
![image](https://user-images.githubusercontent.com/48408342/152689797-71411b9a-f683-4f4a-bd19-13cba2370ee8.png)
![image](https://user-images.githubusercontent.com/48408342/152689801-91500ad8-8968-4f82-a22d-c6cf5a410155.png)
![image](https://user-images.githubusercontent.com/48408342/152689804-5d82e401-eaa9-413e-83be-1f743615ec6a.png)
![image](https://user-images.githubusercontent.com/48408342/152689807-f1b1b544-7795-4aa4-8a35-b432d77fbece.png)

The model predicts the test set almost perfectly accurate. Precision and recall are almost 1.

# Deep Learning Model 2 - Faster R-CNN
