# odyssey-project
Identification of archaeological sites in aerial image using deep learning

This work was developed within the scope of the ODISSEY project financed by PORTUGAL2020 program.
This project aims to develop an integrated platform for geographic information aimed at archaeologists and heritage technicians. Identification and detection are performed automatically through deep learning techniques based on data from nonintrusive methods, such as LiDAR. However, a pre-processing and the annotation of these data is necessary to make possible the use of deep learning algorithms. In this work, the data acquired in the area of the district Viana do Castelo located in the north of Portugal were used, where there are 136 archaeological objects identified and known as ’mamoas’. For the identification of this object were used YOLOv5, Mask R-CNN and CNN models. A study is also carried out on the impact of the sizes of the bounding box to be used in the annotation of ’mamoa’. The custom CNN algorithm is only used for the best dimension found with YOLOv5 and Mask R-CNN. In the end, an inference is made to the entire image of the district to verify the behavior of the models obtained and to discover possible archaeological objects not yet identified by human action.

## First approach using archaeological data
This first work was based on this paper, https://doi.org/10.3390/rs14030553, which contains supplementary materials, for example, the dataset and the code used. The authors used the mask R-CNN algorithm, which uses the pixellib library. The Mask R-CNN is based on Fast R-CNN, but while this last one has two outs by the object, bounding boxes and class, the mask R-CNN has three outs, because it has the mask of the object which is a region of interest. 

The dataset is composed of images obtained through a helicopter. Firstly was used the labelme tool to create the annotated data with JSON files in COCO format. 
The number total of images is 384, and the files were split into 287 images to train, 71 for validation and 26 for the test. Each JSON file is composed of images ID, category name, pixel values, bounding boxes coordinates and the dimensions of images. There are 3 classes of archaeological objects, qanat, ruined structures and mounded sites.

I used this dataset and I reply the work done. The train was done through google colab, because it allows the use GPU Tesla K80 with 12GB of RAM. Note that tensorflow and keras version were 2.3.0 and 2.4.0, respectively. The train uses a pre-trained model, "mask_rcnn_coco.h5", which contains 80 classes of COCO dataset. The number of epochs used was 100, but could be lower, because after some time the validation loss remained more or less constant. The train uses augmentation, which uses rotation and noise techniques, this allows to the model learn different representations of objects and decrease the overfitting.

I decided to use the YOLOv5 algorithm, just to learn and get more experience, and I used exactly the same dataset. But, it was necessary to convert the annotations from COCO format to YOLO format. For that, I wrote the script coco_to_yolo.ipynb which read the JSON files and gets all relevant information, mainly the classes ID and bounding boxes coordinates. These bounding boxes have a format: xmin, ymin, width, height in COCO format, while the yolo format is: x_center, y_center, width_yolo, height_yolo. Note that these values of yolo are normalized, so I divided the original values by the dimensions of images.
The results were not better than the mask R-CNN approach, due to the same reason, and the other important problem is related to the data unbalanced because we have 124 images to ruined structures, 68 of qanats and 94 of mounded sites. 

# LiDAR Data

## Annotation
![Imagem9](https://user-images.githubusercontent.com/33499431/164008890-1f9409a9-16c7-4df5-82c7-ec96872ac317.jpg)

The preprocessing of data has the main objective to get the dataset with images and annotations. Initially, there is a LRM (Local Relief Model) which is a visualization technique applied to DTM (Digital Terrain Model) and the respective shapefiles with the geocoordinates of archaeological objects. The LRM allows performing the relief visualization of terrain. The format of LRM is a TIF file and occupies a lot of space of memory, thus, it is not present in this GitHub. The research area is a district of North Portugal named Viana do Castelo and the archaeological object in study is named 'mamoa'.

Google Earth Engine was used to visualize the LRM and to get the dataset. A first approach was developed the script, "ArchaeologicalObjects_earth_engine.txt", to get an image per mamoa and other images with not mamoa. This dataset could be used with a machine-learning algorithm or with a simple costum CNN (Convolutional Neural Network). But, in this case, we want to study different approaches using deep learning algorithms, like YOLO and R-CNN. So, the objective is to have one or more mamoas per image with the respective bounding boxes. For this, the script "CreateDataSet_earth_engine.txt" was written. Firstly, real-world coordinates were converted to pixel coordinates. So, the LRM was split into tiles and saved the ones that had the number of mamoas greater than 0. The images are saved with 640x640 pixels of resolution on a scale of 0.5m/px. An important note is that script was written in javaScript language in the google earth engine, and there was a problem with client-server code when I tried to save the images. So this code was rewritten in python language on google colab using the ee library (earth engine), this code is present on "CreateDataset.ipynb" file. Each image has associated a text file with the bounding boxes coordinates in YOLO format. 

The dataset was composed of a total of 80 images and it was split into 60% for train, 20% for validation, and 20% for test, using the splitfolders library. But, this dataset was very small and for deep learning are needed a lot of data. To resolve this the albumentations library was used to augment the train and validation dataset. It was used transpose, horizontal and vertical flip, RGB shift, blur and ColorJitter.

There is an objective to explore different dimensions of bounding boxes. So the annotation was done with these six different dimensions: 15x15, 20x20, 22x22, 25x25, 30x30, 35x35 (meters).  

## Processing
To train were used three different algorithms, such as YOLOv5, Mask R-CNN and custom CNN. YOLOv5 was trained with a normal dataset and with K-Fold cross-validation. To this last was used model ensemble to use the models obtained in the final of each fold.

The code used YOLOv5 can be found in this GitHub link: https://github.com/ultralytics/yolov5.git
The code used Mask R-CNN can be found in this Github link: https://github.com/matterport/Mask_RCNN.git

The train with YOLOv5 and Mask R-CNN was done with Google Colaboratory which allows to use GPU Tesla K80 or Tesla T4, so this performs the train more quickly and allows the use of the adequated computational resources. 

YOLOv5 uses PyTorch and Mask R-CNN uses Tensorflow.
CNN was used with Keras and TensorFlow.

In case of YOLOv5 and Mask R-CNN was used pre-trained models, so the train was done by transfer learning. This choice was done because this approach has advantages and the dataset is small. It was used 'yolov5s.pt' (small) and 'mask_rcnn_coco.h5'.

The train with CNN was very fast because the images are in grayscale and are very small. Note that in this case, each image represents a mamoa. After getting the results with YOLO and Mask R-CNN and the study with different dimensions of bounding boxes done, it was found that 20x20 meters had good results, so this dimension was used to images in the custom CNN algorithm. The images were cropped with a scale of 0.5m/px, so these images have 40x40 px of dimension.


The next image represents an example of a result of detection after the train with YOLOv5.
![51_2](https://github.com/fabriciobotelho99/odyssey-project/blob/135bcccf801c542861babe2525e28a6124e3ddef/processing%20mamoas/51_2.jpg)[width: 200px]



## LRM Image Inference

https://uapt33090-my.sharepoint.com/:f:/g/personal/fabriciobotelho_ua_pt/EkkEsBrAW9BMqK81DDKxEDQBLQUudfsYd3JYZP4MWR7sKg?e=zmnaZu

