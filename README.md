# odyssey-project
Identification of archaeological sites in aerial image using deep learning

## first approach using archaeological data
This first work was based on this paper, https://doi.org/10.3390/rs14030553, which contains supplementary materials, for example, the dataset and the code used. The authors used the mask R-CNN algorithm, which uses the pixellib library. The Mask R-CNN is based on Fast R-CNN, but while this last one has two outs by the object, bounding boxes and class, the mask R-CNN has three outs, because it has the mask of the object which is a region of interest. 

The dataset is composed of images obtained through a helicopter. Firstly was used the labelme tool to create the annotated data with JSON files in COCO format. 
The number total of images is 384, and the files were split into 287 images to train, 71 for validation and 26 for the test. Each JSON file is composed of images ID, category name, pixel values, bounding boxes coordinates and the dimensions of images. There are 3 classes of archaeological objects, qanat, ruined structures and mounded sites.

I used this dataset and I reply the work done. The train was done through google colab, because it allows the use GPU Tesla K80 with 12GB of RAM. Note that tensorflow and keras version were 2.3.0 and 2.4.0, respectively. The train uses a pre-trained model, "mask_rcnn_coco.h5", which contains 80 classes of COCO dataset. The number of epochs used was 100, but could be lower, because after some time the validation loss remained more or less constant. The train uses augmentation, which uses rotation and noise techniques, this allows to the model learn different representations of objects and decrease the overfitting.
The batch size used was 4, but experimented with value 2, which obtained slightly better results. The mean average precision was about 20% and the validation loss was about 1.80 and 1.7, which is not very good. These results can be justified because the dataset used is very small to deep learning approach, we would need more images. 
The documentation of pixellib says that maybe we could need about 300 images for each class to do the train and get better results. Note that here we have just 287 images for the three classes.

I decided to use the YOLOv5 algorithm, just to learn and get more experience, and I used exactly the same dataset. But, it was necessary to convert the annotations from COCO format to YOLO format. For that, I wrote the script coco_to_yolo.ipynb which read the JSON files and gets all relevant information, mainly the classes ID and bounding boxes coordinates. These bounding boxes have a format: xmin, ymin, width, height in COCO format, while the yolo format is: x_center, y_center, width_yolo, height_yolo. Note that these values of yolo are normalized, so I divided the original values by the dimensions of images.
The results were not better than the mask R-CNN approach, due to the same reason, and the other important problem is related to the data unbalanced because we have 124 images to ruined structures, 68 of qanats and 94 of mounded sites. 

# LiDAR Data

## Preprocessing
![image1](https://drive.google.com/file/d/13kJYMnDL3svsPA5nGzFI83KKxdrl6Cqd/view?usp=sharing)

The preprocessing of data has the main objective to get the dataset with images and annotations. Initially, there is a LRM (Local Relief Model) which is a visualization technique applied to DTM (Digital Terrain Model) and the respective shapefiles with the geocoordinates of archaeological objects. The LRM allows performing the relief visualization of terrain. The format of LRM is a TIF file and occupies a lot of space of memory, thus, it is not present in this GitHub. The research area is a district of North Portugal named Viana do Castelo and the archaeological object in study is named "mamoa".

Google Earth Engine was used to visualize the LRM and to get the dataset. A first approach was developed the script, "ArchaeologicalObjects_earth_engine.txt", to get an image per mamoa and other images with not mamoa. This dataset could be used with a machine-learning algorithm or with a simple CNN (Convolutional Neural Network). But, in this case, we want to study different approaches using deep learning algorithms, like YOLO and R-CNN. So, the objective is to have one or more mamoas per image with the respective bounding boxes. For this, the script "CreateDataSet_earth_engine.txt" was written. Firstly, real-world coordinates were converted to pixel coordinates. So, the LRM was split into tiles and saved the ones that had the number of mamoas greater than 0. The images are saved with 320x320 pixels of resolution on a scale of 1m/px. An important note is that script was written in javaScript language in the google earth engine, and there was a problem with client-server code when I tried to save the images. So this code was rewritten in python language on google colab using the ee library (earth engine), this code is present on "CreateDataset.ipynb" file. Each image has associated a text file with the bounding boxes coordinates in YOLO format. 

The dataset was composed of a total of 80 images and it was split into 60% for train, 20% for validation, and 20% for test, using the splitfolders library. But, this dataset was very small and for deep learning are needed a lot of data. To resolve this the albumentations library was used to augment the train and validation dataset. It was used transpose, horizontal and vertical flip, RGB shift, blur and ColorJitter. In the end, the dataset is composed of a total of 720 images (528 to train, 176 to validation and 16 to test).

## Processing

