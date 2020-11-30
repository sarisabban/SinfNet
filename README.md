# Microorganism Image Classifier
A collection of datasets and neural networks for microorganism image classification.

## Contributers:
Sari Sabban

Tarik Alafif

Abdullah Alotebi

## Description:
This is a collection of datasets and neural networks to detect or classify microorganisms from microscope images. Provided here are all the necessary scripts, datasets, and weights. So far this project either detects or classifies the following organisms:

<p align="center">Protists - 10 species:</p>

<sub>*Colsterium ehrenbergii - Cylindrocystis brebissonii - Lepocinclis spirogyroides - Micrasterias rotata - Paramecium bursaria - Peridinium spec. - Pinnularia neomajor - Pleurotaenium ehrenbergii - Pyrocystis lunula - Volvox tertius*</sub>

<p align="center">Amoebas:</p>

<sub>*Just detection of generic cells and differentiates between the active and inactive stages of the life cycle.*</sub>

<p align="center">Nematodes:</p>

<sub>*Either classifies nematodes according to trophic level (CNN), or detects generic nematodes (Object Detection), or detects nematodes pixel wise (Semantic Segmentation) for biomass calculation*</sub>

## Available datasets and trained weight files
All datasets used are available here for download, along with their neural network weights for detection/classification.

|Dataset Name                                                                                                       |Network |Weights                                                                    |mAP or Accuracy|
|-------------------------------------------------------------------------------------------------------------------|--------|---------------------------------------------------------------------------|---------------|
|[Amoeba Active/Inactive Dataset](https://www.dropbox.com/s/vf2ftfige4vu0ie/Amoeba.tar.bz2?dl=0)                    |YOLOv3  |[Weights](https://www.dropbox.com/s/x044cdo7kznoeuf/Amoeba.h5?dl=0)        |0.6473         |
|[Cell Detection Dataset](https://www.dropbox.com/s/2woe91t03rw9kbm/Cells.tar.bz2?dl=0)                             |YOLOv3  |[Weights](https://www.dropbox.com/s/yukp34x3gaubd4u/Cells.h5?dl=0)         |0.9549         |
|[Nematode Detection Dataset](https://www.dropbox.com/s/5leewk48vj6ip6l/Nematodes_Detect.tar.bz2?dl=0)              |YOLOv3  |[Weights](https://www.dropbox.com/s/z638ml32x7i3kef/Nematodes.h5?dl=0)     |0.8867         |
|[Nematode Trophic Classification Dataset](https://www.dropbox.com/s/dwhvmdx6xc4chaf/Nematodes_Trophic.tar.bz2?dl=0)|ResNet50|[Weights](https://www.dropbox.com/s/oba72fd9nlryauf/Nematodes_Feed.h5?dl=0)|0.9909         |
|[Nematode Semantic Dataset](https://www.dropbox.com/s/779le560wt159x4/Nematodes_Semantic.tar.bz2?dl=0)             |UNet    |[Weights](https://www.dropbox.com/s/cf7g62fil44r2mj/unet_binary.h5?dl=0)   |0.95896        |
|[Protist Classification Dataset](https://www.dropbox.com/s/ioiw2pcynpcaq4k/Protists.tar.bz2?dl=0)                  |YOLOv3  |[Weights]()|               |

## How to use:
This is a [Video]() on how to use this setup.

### Update your system and install libraries.
This setup works on GNU/Linux Ubuntu 18.04+ using Python 3.6+. To use this script you will need to first update your system and install the dependencies, within a virtual environment, using the following commands:

`sudo apt update`

`sudo apt full-upgrade`

`sudo apt install python3-pip python3-tk`

`python3 -m venv env`

`source env/bin/activate`

`pip3 install numpy keras tensorflow seaborn tkintertable matplotlib imgaug scipy pillow scikit-image imutils h5py opencv-contrib-python pydensecrf`

`deactivate`

### Setting up a dataset
If you want to develop your own dataset and train it follow these steps otherwise skip to section **Detection**. For help use this command:

`python SinfNet.py --help` or `python SinfNet.py -h`

#### For object detection
1.1 Collect images containing your objects. Even though the network can process different image formats, it is best to stick with the .jpg image format.

1.2 If you can generate a total-slide scan (a single image of an entire slide) you can segment this large image into smaller images to build your dataset, use the following command to segment:

`python SinfNet.py --segment FILENAME WIDTH HIGHT` or `python SinfNet.py -S diatoms.jpg 2584 1936`

2. Make a directory called *dataset* and within it in make the following directories:

        ./dataset/Annotations
        ./dataset/Train
        ./dataset/Valid
        ./dataset/Valid_Annotations

Your images should be in *./dataset/Train*. It is best to stick to this structure with these names exactly, otherwise you will have to change these path names within each relevant script, so stick to these names and keep it simple.

If you would like to augment the images that have object detection annotations (bounding boxes) use the following command:

`python SinfNet.py --augment_bbox NUMBER INPUT_FORMAT OUTPUT_FORMAT` example `python SinfNet.py -ab 10 txt xml`

Where NUMBER is the number of augments to each image, INPUT_FORMAT is the file format of the images annotations, and OUTPUT_FORMAT is the desired output format of the augmented annotations. This will generate a new directory called *Augmented* (and *Augmented_Annotations* for object detection) with all the saved augmented images in it. Only these augmented images should be used for training (by moving them to *./dataset/Train*) and not mixed with the original images (by moving the original images to *./dataset/Valid* and their .xml annotation to *./dataset/Valid_Annotations*).

3. Open the web-based GUI annotation tool using the following command:

`python SinfNet.py --via` or `python SinfNet.py -v`

You must have FireFox for this to work.

Use this tool for or bounding box annotation that will be used with object detection or polygon annotation that will be used with semantic segmentation. There is another tool only for bounding box annotation,  but it is prefeared not to used it, yet can be accessed using the following command:

`python SinfNet.py --bbox` or `python SinfNet.py -b`

You will be prompted to add the labels, then type *end* to finish adding the label and start the GUI program. A new directory called *BBox_Annotations* will be generated that will contain the annotations in .txt format.

4. Watch the video to understand how to annotate using this tool, or read the manual under *Help > Getting Started*.

5. Use the mouse to generate a bounding box around each object of interest and label it.

6. Once finished, download your annotations as a .csv file.

7. Convert the .csv file to a .xml file using the following command:

`python SinfNet.py --translate_bbox IMAGE_DIRECTORY ANNOTATION_INPUT ANNOTATION_OUTPUT INPUT_FORMAT OUTPUTFORMAT` or `python SinfNet.py -tb ./dataset/Train ./dataset/Annotations ./dataset/Translations csv xml`

Where IMAGE_DIRECTORY is the path to the directory of images, ANNOTATION_INPUT the path to the directory with the files to be converted, ANNOTATION_OUTPUT the path to the directory where the converted files are to be saved, INPUT_FORMAT the input file format OUTPUTFORMAT the format to convert to.

identify only the sigle file for the .csv and .json formats, txt and xml must identify the directory the multiple files reside in.

8. Do not delete the .csv file, rather save it in case you want to rename any label.

#### For semantic segmentation

1. The *dataset* directory structure will be as follows:

        ./dataset/Test
        ./dataset/Test_Annotations
        ./dataset/Train
        ./dataset/Train_Annotations

2. The neural network only takes images with dimentions that are multiples of 32, therefore it is possible to crop the images to satisfy this restriction using the following command:

`python SinfNet.py --crop IMAGE_DIRECTORY` or `python SinfNet.py -C ./dataset/Train`

A new direcory called Cropped will be generated with the cropped images in it.

3. Follow the same steps as object detection except use polygons instead of squares to annotate the objects. The difference is to save the annotation as a .csv file [from top right Annotations > Export Annotations (as csv)] and rename appropriatly. Then you will have to convert the .csv file to multiple .json files using the following command:

`python SinfNet.py --translate_poly IMAGE_DIRECTORY ANNOTATION_INPUT ANNOTATION_OUTPUT INPUT_FORMAT OUTPUTFORMAT` or `python SinfNet.py -tp ./dataset/Train ./dataset/Nematode.csv ./dataset/Train_Annotations csv json`

Where IMAGE_DIRECTORY is the path to the directory of images, ANNOTATION_INPUT the path to the directory with the files to be converted, ANNOTATION_OUTPUT the path to the directory where the converted files are to be saved, INPUT_FORMAT the input file format OUTPUTFORMAT the format to convert to. identify only the sigle file for the single .csv file or the directory with the .json files.

If you would like to augment the images use the following command:

`python SinfNet.py --augment_polygon CSV NUMBER` example `python SinfNet.py -ap ./dataset/Train/Nematodes.csv 10`

Where CSV is the .csv file that contains the polygone annotations for the entire dataset. Then translate the .csv to .json files as above for training (augmented images go to *./dataset/Train* and the augmented annotation to *./dataset/Train_Annotations*, while the original images go to *./dataset/Test* and their annotations to *./dataset/Test_Annotations*).

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

#### For semantic segmentation
1. Follow the same steps as object detection, use the following command to train:

`python SinfNet.py --semantic_train MODE LABELS` for example `python SinfNet.py -st multi Active Inactive Inactive`

Where MODE can be either binary (for single or multiple classes being coloured white with a black background) or multi (for single or multiple classes being coloured differently on a black background). And as with object detection, including pre-trained weights is possible:

`python SinfNet.py --semantic_train LABELS` for example `python SinfNet.py -st Active Inactive`

The weights directory (*./models*) must be present in the same working directory as the SinfNet.py script for this command to work.

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

#### For semantic segmentation
1. Use the following command to predict/detect:

`python SinfNet.py --semantic_predict FILENAME` example `python SinfNet.py -sp image.jpg`

The weights directory (*./models*) must be present in the same working directory as the SinfNet.py script for this command to work.

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

## Nematode Biomass Approximation:
`python SinfNet.py --biomass W H D w h P` or `python SinfNet.py -B 1000 1000 100 256 256 15892`

Where W is the width of the image in μm, H is the hight of the image in μm, D is the depth of the image in μm, w is the width of the image in pixels, h is the hight of the image in pixels, P is the number of white (positive detection) pixels in the detection/prediction output of the UNet semantic segmentation neural network.

## Table of commands:
|Command                                                                                          |Description                                                                |
|-------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------|
python SinfNet.py -h                                                                              |Help                                                                       |
python SinfNet.py -a  NUMBER                                                                      |Augment images                                                             |
python SinfNet.py -ab NUMBER INPUT_FORMAT OUTPUT_FORMAT                                           |Augment images with bounding boxes                                         |
python SinfNet.py -ap CSV NUMBER                                                                  |Augment images with bounding polygons                                      |
python SinfNet.py -v                                                                              |Open weg-based immage annotator                                            |
python SinfNet.py -b                                                                              |BBox                                                                       |
python SinfNet.py -c DIRECTORY                                                                    |Convert Bash output to .xml                                                |
python SinfNet.py -C DIRECTORY                                                                    |Crops images to make their dimentions multiples of 32                      |
python SinfNet.py -tb IMAGE_DIRECTORY ANNOTATION_INPUT ANNOTATION_OUTPUT INPUT_FORMAT OUTPUTFORMAT|Convert between different bbox annotation formats(txt, csv, coco-json, xml)|
python SinfNet.py -tp IMAGE_DIRECTORY ANNOTATION_INPUT ANNOTATION_OUTPUT INPUT_FORMAT OUTPUTFORMAT|Convert between different polygon formats(csv, json)                       |
python SinfNet.py -yt WEIGHTS PROJECT_NAME LABELS                                                 |YOLOv3 network train                                                       |
python SinfNet.py -yp WEIGHTS FILENAME LABELS                                                     |YOLOv3 network predict                                                     |
python SinfNet.py -ct CNN                                                                         |CNN network train                                                          |
python SinfNet.py -cp CNN WEIGHTS FILENAME                                                        |CNN network classify                                                       |
python SinfNet.py -sp FILENAME                                                                    |UNet network predict                                                       |
python SinfNet.py -st MODE LABELS                                                                 |UNet network train                                                         |
python SinfNet.py -B W H D w h P                                                                  |Calculates nematode biomass from UNet output                               |
python SinfNet.py --segment FILENAME WIDTH HIGHT                                                  |Segments a large total-slide scan image into smaller images                |

## Funders:
* [Experiment](https://experiment.com/)
* [Microsoft](https://www.microsoft.com/en-us/ai/ai-for-earth-tech-resources)

## References:
When using any part of this project kindly reference the following:

* 

## TODO:

* Make Video
