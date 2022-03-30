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


