# Microorganism Image Classifier
A neural network for microorganism image classification.

## Description:
This is a collection of datasets and neural networks to detect or classify microorganisms from microscope images. Provided here are all the necessary scripts, datasets, and weights. So far this project either detects or classifies the following organisms:

<p align="center">Protists - 10 species:</p>

<sub>*Colsterium ehrenbergii - Cylindrocystis brebissonii - Lepocinclis spirogyroides - Micrasterias rotata - Paramecium bursaria - Peridinium spec. - Pinnularia neomajor - Pleurotaenium ehrenbergii - Pyrocystis lunula - Volvox tertius*</sub>

<p align="center">Amoebas:</p>

<sub>*Just detection of generic cells and differentiates between the active and inactive stages of the life cycle.*</sub>

<p align="center">Nematodes:</p>

<sub>*Either detection of generic nematodes for biomass calculation or classifies nematodes according to feeding habits*</sub>

## Available datasets and trained weight files
All datasets used are available here for download, along with their neural network weights for detection/classification.

|Dataset Name                                                                                                    |Network     |Weights                                                                    |mAP or Accuracy|
|----------------------------------------------------------------------------------------------------------------|------------|---------------------------------------------------------------------------|---------------|
|[Amoeba Active/Inactive Dataset](https://www.dropbox.com/s/vf2ftfige4vu0ie/Amoeba.tar.bz2?dl=0)                 |YOLOv3      |[Weights](https://www.dropbox.com/s/x044cdo7kznoeuf/Amoeba.h5?dl=0)        |0.6473         |
|[Cell Detection Dataset](https://www.dropbox.com/s/2woe91t03rw9kbm/Cells.tar.bz2?dl=0)                          |YOLOv3      |[Weights](https://www.dropbox.com/s/yukp34x3gaubd4u/Cells.h5?dl=0)         |0.9549         |
|[Nematode Detection Dataset](https://www.dropbox.com/s/5leewk48vj6ip6l/Nematodes_Detect.tar.bz2?dl=0)           |YOLOv3      |[Weights](https://www.dropbox.com/s/z638ml32x7i3kef/Nematodes.h5?dl=0)     |0.8867         |
|[Nematode Feeding Classification Dataset](https://www.dropbox.com/s/dwhvmdx6xc4chaf/Nematodes_Feed.tar.bz2?dl=0)|ResNet50 CNN|[Weights](https://www.dropbox.com/s/oba72fd9nlryauf/Nematodes_Feed.h5?dl=0)|0.9909         |
|[Algae Classification Dataset](https://www.dropbox.com/s/ioiw2pcynpcaq4k/Algae.tar.bz2?dl=0)                    |YOLOv3      |[Weights]()|               |
|[Nematode Biomass Dataset]()               |Mask-RCNN   |[Weights]()|               |

## How to use:
This is a [Video]() on how to use this setup.

### Update your system and install libraries.
This setup works on GNU/Linux Ubuntu 18.04+ using Python 3.6+. To use this script you will need to first update your system and install the dependencies, within a virtual environment, using the following commands:

`sudo apt update`

`sudo apt full-upgrade`

`sudo apt install python3-pip python3-tk`

`python3 -m venv env`

`source env/bin/activate`

`pip3 install numpy keras tensorflow seaborn tkintertable matplotlib imgaug scipy scipy pillow scikit-image imutils h5py opencv-contrib-python "IPython[all]"`

`deactivate`

### Setting up a dataset
If you want to develop your own dataset and train it follow these steps otherwise skip to section **Detection**. For help use this command:

`python SinfNet.py --help` or `python SinfNet.py -h`

#### For object detection
1. Collect images containing your objects. Even though the network can process different image formats, it is best to stick with the .jpg image format.

2. Make a directory called *dataset* and within it in make the following directories:

        ./dataset/Annotations
        ./dataset/Train
        ./dataset/Valid
        ./dataset/Valid_Annotations

Your images should be in *./dataset/Train*. It is best to stick to this structure with these names exactly, otherwise you will have to change these path names within each relevant script, so stick to these names and keep it simple.

If you would like to augment the images that have object detection annotations use the following command:

`python SinfNet.py --augment_object NUMBER INPUT_FORMAT OUTPUT_FORMAT` example `python SinfNet.py -ao 10 txt xml`

Where NUMBER is the number of augments to each image, INPUT_FORMAT is the file format of the images annotations, and OUTPUT_FORMAT is the desired output format of the augmented annotations. This will generate a new directory called *Augmented* (and *Augmented_Annotations* for object detection) with all the saved augmented images in it. Only these augmented images should be used for training (by moving them to *./dataset/Train*) and not mixed with the original images (by moving the original images to *./dataset/Valid* and their .xml annotation to *./dataset/Valid_Annotations*).

3. Open the web-based GUI annotation tool using the following command:

`python SinfNet.py --via` or `python SinfNet.py -v`

You must have FireFox for this to work. There is a bug with this tool where the boxes are described in an unconventional way: the fix is to define the width and hight from the start of the image instead of an addition to the axis value [VIA->BBOX W: x+w H: y+h | BBOX->VIA W: x-w H: y-h]

Use this tool only for polygon annotation that will be used with instance segmentation, use the following command for bounding box annotation:

`python SinfNet.py --bbox` or `python SinfNet.py -b`

You will be prompted to add the labels, then type *end* to finish adding the label and start the GUI program. A new directory called *BBox_Annotations* will be generated that will contain the annotations in .txt format.

4. Watch the video to understand how to annotate using this tool, or read the manual under *Help > Getting Started*.

5. Use the mouse to generate a bounding box around each object of interest and label it.

6. Once finished, download your annotations as a .csv file.

7. Convert the .csv file to a .xml file using the following command:

`python SinfNet.py --translate_bbox IMAGE_DIRECTORY ANNOTATION_INPUT ANNOTATION_OUTPUT INPUT_FORMAT OUTPUTFORMAT` or `python SinfNet.py -tb ./dataset/Train ./dataset/Annotations ./dataset/Translations txt xml`

Where IMAGE_DIRECTORY is the path to the directory of images, ANNOTATION_INPUT the path to the directory with the files to be converted, ANNOTATION_OUTPUT the path to the directory where the converted files are to be saved, INPUT_FORMAT the input file format OUTPUTFORMAT the format to convert to.

identify only the sigle file for the .csv and .json formats, txt and xml must identify the directory the multiple files reside in.

8. Do not delete the .csv file, rather save it in case you want to rename any label.

#### For instance segmentation

1. Follow the same steps as object detection except use polygons instead of squares to annotate the objects. The difference is to save the annotation as a JSON file [from top right Annotations > Export Annotations (as json)] and add this file to the directory of the images it annotates (train annotation in the Train directory and validation annotations to the Valid directory).










If you would like to augment the images use the following command

`python SinfNet.py --augment_segment NUMBER INPUT_FORMAT OUTPUT_FORMAT` example `python SinfNet.py -as 10 txt csv`











#### For classification
1. The dataset should be have the following directory architecture. Within each directory a directory of the classes that includes all the images of that class, as such:

    ./dataset/Test/
                  class1/
                        image1.jpg
                        image2.jpg
                  class2/
                        image1.jpg
                        image2.jpg

    ./dataset/Train
                  class1/
                        image1.jpg
                        image2.jpg
                  class2/
                        image1.jpg
                        image2.jpg
                        
    ./dataset/Valid
                  class1/
                        image1.jpg
                        image2.jpg
                  class2/
                        image1.jpg
                        image2.jpg

2. If you would like to augment the images use the following command:

`python SinfNet.py --augment NUMBER` example `python SinfNet.py -a 10`

Where NUMBER is the number of augments to each image. This will generate a new directory called *Augmented* with all the saved augmented images in it. Only these augmented images should be used for training (by moving them to *./dataset/Train/CLASS*) and not mixed with the original images (by moving the original images to *./dataset/Test/CLASS*).

3. Shuffle your dataset and randomly split each class into a *Training* (60% of images), *Testing* (20% of images), and *Validation* (20% of images) sets using the following command:

`shuf -n NUMBER -e ./dataset/Train/CLASS | xargs -i mv {} ./dataset/Valid/CLASS`

Where NUMBER is the number of images that will be moved (calculated as 60% or 20% of the total images in the dataset), and CLASS is the specific class for the images. This is to ensure that the sets are randomly split before training the neural network.

### Training the neural network
#### For object detection
1. Use the following command to train the network on your dataset:

`python SinfNet.py --yolo_train WEIGHTS PROJECT_NAME LABELS` for example `python SinfNet.py -yt Amoeba Amoeba Active Inactive`

The WEIGHTS is the name of the output weight.h5 file, the PROJECT_NAME is just a name for your project (must be included), and LABELS is a list of all the labels in the dataset (just the labels written with space between them).

2. The network is resource heavy and requires a large GPU and more than 16GB of RAM to run (depending on dataset size). Therefore some cloud GPU services may not work and a larger system is required.

3. A logs directory will be generated containing the training logs. View the data using the following command if you have tensorboard installed:

`tensorboard --logdir=./logs`

4. The .h5 file is the weights file used for image detection.

5. If the training is interrupted, you can use the .h5 file to continue where you left off using the exact same training command in step 1.

#### For instance segmentation

1. Follow the same steps as object detection, use the following command to train:

`python SinfNet.py --mrcnn_train LABEL` for example `python SinfNet.py -mt Active`

At this moment, the script only takes one label at a time. But labels can be added by including pre-trained weights as such:

`python SinfNet.py --mrcnn_train LABEL WEIGHTS.h5` for example `python SinfNet.py -mt Active Amoeba.h5`

At the end of the training a directory will be generated which includes the log files and weight files. You should save the last weight file that results from the full run of the neural network.

#### For classification
1. You can train the images on a CNN using the following command:

`python SinfNet.py --cnn_train CNN` or `python SinfNet.py -ct CNN`

Where CNN is the name of the convolutional neural network that you want to use. Choose one of the following [VGG16, VGG19, ResNet50, DenseNet201].

2. Training, loss, and confusion matrix figures will be generated after the training is done. An evaluation on the test set will be performed and the result printed on the terminal.

### Detection
#### For object detection
1. Download the relevant weights file (links available in table above) or generate the file from the steps above.

2. Predict/Detect objects in your image/video/webcam using the following command:

`python SinfNet.py --yolo_predict WEIGHTS.h5 FILENAME LABELS` example `python SinfNet.py -yp Amoeba.h5 image.jpg Active Inactive`

Where WEIGHTS.h5 is the weights file, the FILENAME can be either a .jpg image, .mp4 video, or a webcam input, and the LABELS is a list of all the labels in the dataset (just the labels written with space between them).

#### For instance segmentation

1. Follow the same steps as object detection, use the following command to predict/detect:

`python SinfNet.py --mrcnn_predict WEIGHTS.h5 FILENAME LABELS` example `python SinfNet.py -mp Amoeba.h5 image.jpg BG Active Inactive`

Always include BG (Background) as the first label.

#### For classification
1. Download the relevant weights file (links available in table above) or generate the file from the steps above.

2. To run a classification use the following command:

`python SinfNet.py --cnn_predict CNN WEIGHTS FILENAME` or `python SinfNet.py -cp ResNet50 Nematodes.h5 image.jpg`

Where the CNN is the name of the network that was used to generate the WEIGHTS.h5 file (using different networks from the weights file does not work), WEIGHTS is the name of the .h5 weights file, and FILENAME is the name of the image file.

## Auto Annotation:
Since manual annotations are time consuming, we can use the neural network to annotate new images to build a new dataset (instead of annotating 1000s of images manually), make sure you use the Cell.h5 weights file if you want to predict only the cells in the images. You must insure your images are made up of a pure cell strain.

1. Change the prediction threshold from 0.5 to 0.8 in line 1207: `obj_thresh, nms_thresh = 0.50, 0.45` to `obj_thresh, nms_thresh = 0.80, 0.45`, and comment out the last line of the script `#cv2.imwrite ...` line 1265 of the YOLOv3.py script as to not generate images.

3. Use the following command to loop through all images and predict the cells:

`for f in ./DIRECTORY/*; do python YOLOv3.py -d WEIGHTS.h5 $f >> DIRECTORY; done`

Where DIRECTORY is the name of the directory that contains all the images.

4. Then use the following command to generate the Annotation text files:

`python SinfNet.py --convert DIRECTORY` or `python SinfNet.py -c amoeba`

5. Check all images to make sure the the annotations are correct, and to correct minor errors.

The annotations are as good as the training of the network, which is not 100%, therefore a human must go over the annotated images to fix any minor mistakes.

**Contributing to our dataset**
If you would like to add images to our dataset (any type of microscopic organism) make sure that each species has 200 annotated images where each image is sharp and taken from a bright-field light microscope at 400x magnification. Please contact me so we can work together.

## Table of commands:
|Command                                         |Description                                                                |
|------------------------------------------------|---------------------------------------------------------------------------|
python SinfNet.py -h                             |Help                                                                       |
python SinfNet.py -a NUMBER OF IMAGES            |Augment                                                                    |
python SinfNet.py -v                             |Open weg-based immage annotator                                            |
python SinfNet.py -b                             |BBox                                                                       |
python SinfNet.py -c DIRECTORY                   |Convert Bash output to .xml                                                |
python SinfNet.py -tb                            |Convert between different bbox annotation formats(txt, csv, coco-json, xml)|
python SinfNet.py -yt WEIGHTS PROJECT_NAME LABELS|YOLOv3 network train                                                       |
python SinfNet.py -yp WEIGHTS FILENAME LABELS    |YOLOv3 network predict                                                     |
python SinfNet.py -ct CNN                        |CNN network train                                                          |
python SinfNet.py -cp CNN WEIGHTS FILENAME       |CNN network classify                                                       |
python SinfNet.py -mp WEIGHTS.h5 FILENAME LABELS |Mask-RCNN network predict                                                  |
python SinfNet.py -mt LABEL                      |Mask-RCNN network train                                                    |
python SinfNet.py -mt LABEL WEIGHTS.h5           |Mask-RCNN network train with pre-trained weights                           |

## Funders:
* [Experiment](https://experiment.com/)
* [Microsoft](https://www.microsoft.com/en-us/ai/ai-for-earth-tech-resources)

## References:
When using any part of this project kindly reference the following:

* 

## TODO:

* Make Video

* Add reference
