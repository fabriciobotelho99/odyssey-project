# odyssey-project
Identification of archaeological sites in aerial image using deep learning

## first approach using archaeological data
This first work was based on this paper, https://doi.org/10.3390/rs14030553, that contains the supplementary materials, as example the dataset and the code used. The authors used the mask R-CNN algorithm, witch uses pixellib library. The Mask R-CNN is based on Fast R-CNN, but while this last one has two outs by object, bounding boxes and class, the mask R-CNN has three outs, because it has the mask of object whitch is a region of interest. 

The dataset is composed with images obtained trhough a helicopter. Firstly was used the labelme tool to create the annotated data with JSON files in COCO format. 
The number total of images is 384, and the files was splited in 287 images to train, 71 to validation and 26 to the test. Each JSON file is composed by images ID, cathegory name, pixel values, bboxes coordinates and the dimentions of images. There are 3 classes of archaeological objects, qanat, ruined structure and mounded sites.

I used this dataset and I reply the works done. The train was done throught google colab, because it allow use a GPU Tesla K80 with 12GB of RAM. Note that tensorflow and keras version were 2.3.0 and 2.4.0, respectivaly. The train uses a pre-trained model, "mask_rcnn_coco.h5", whitch contains 80 classes of COCO dataset. The number of epochs used were 100, but could be lower, beacuse after some time the validation loss remained more or less constant. The train uses augmentation, that uses rotation and noise tecniques, this allow to the model learn diferent representations of objects and decrease the overfitting.
The bacth size used was 4, but experimented with value 2, which obtained slightly better results. The mean average precision was about 20% and the validation loss about 1.80 and 1.7, that is not very good. This results can be justified, because the dataset used is verry small to deep learning approach, we would need more images. 
The documnetation of pixellib says that maybe we could need about 300 images for each class to do the train and get better results. Note that here we have just 287 images to the three classes.




