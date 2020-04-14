# Microorganism Image Classifier
A neural network for microorganism image classification.

## Description:
This is a collection of datasets and neural networks to detection or classify microorganisms from microscope images. Provided here are all the necessary scripts, datasets, and weights. So far this project either detects or clasifies the following organisms:

<p align="center">Protists - 10 species:</p>

<sub>*Colsterium ehrenbergii - Cylindrocystis brebissonii - Lepocinclis spirogyroides - Micrasterias rotata - Paramecium bursaria - Peridinium spec. - Pinnularia neomajor - Pleurotaenium ehrenbergii - Pyrocystis lunula - Volvox tertius*</sub>

<p align="center">Amoebas:</p>
<sub>*Just detection of generic cells and differentiates between the active and inactive stages of the life cycle.*</sub>

<p align="center">Nematodes:</p>
<sub>*Either detection of generic nematodes for biomass calculation or classifies nematodes according to feeding habbits*</sub>

## Available datasets and trained weight files
All datasets used are available here for download, along with their neural network weights for detection/classification.

|Dataset Name                                                                                                    |Network     |Weights                                                                             |mAP or Accuracy|
|----------------------------------------------------------------------------------------------------------------|------------|------------------------------------------------------------------------------------|---------------|
|[Amoeba Active/Inactive Dataset]()         |YOLOv3      |[Weights]()|0.6473         |
|[Cell Detection Dataset]()                 |YOLOv3      |[Weights]()|0.9549         |
|[Nematode Detection Dataset]()             |YOLOv3      |[Weights]()|0.8867         |
|[Nematode Feeding Classification Dataset]()|ResNet50 CNN|[Weights]()|0.9909         |
|[Nematode Biomass Dataset]()               |Mask-RCNN   |[Weights]()|               |
|[Algae Classification Dataset]() or [Algae Classification Dataset_Augmented]()|YOLOv3|[Weights]()|0.7118         |

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

##### For object detection
1. Collect images containing your objects. Even though the network can process different image formats, it is best to stick with the .jpg image format.

2. Make a directory called *dataset* and within it in make the following directories:

        ./dataset/Annotations
        ./dataset/BBox_Annotations
        ./dataset/BBox_Test
        ./dataset/Check
        ./dataset/Predictions
        ./dataset/Test
        ./dataset/Train
        ./dataset/Valid

Your images should be in *./dataset/Train*. It is best to stick to this structure with these names exactly, otherwise you will have to change these path names within each relevant script, so stick to these names and keep it simple.

If you would like to augment the images use the following command:

`python SinfNet.py --augment NUMBER` example `python SinfNet.py -a 10`

Where NUMBER is the number of augments to each image. This will generate a new directory called *Augmented* with all the saved augmented images in it. Only these augmented images should be used for training (by moving them to *./dataset/Train*) and not mixed with the original images (by moving the original images to *./dataset/Test*).

3. Open the web-based GUI annotation tool using the following command:

`python SinfNet.py --via` or `python SinfNet.py -v`

You must have FireFox for this to work.

4. Watch the video to understand how to annotate using this tool, or read the manual under *Help > Getting Started*.

5. Use the mouse to generate a bounding box around each object of interest and label it.

6. Once finished, download your annotations as a .csv file.

7. Convert the .csv file to a .xml file using the following command:

`python SinfNet.py --translate_csv` or `python SinfNet.py -tc`

All .xml annotations will be moved to the *./dataset/Annotations* directory.

8. Do not delete the .csv file, rather save it incase you want to rename any label.

##### For classification
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

Where NUMBER is the number of images that will be moved (calculated as 60% or 20% of the total images in the dataset), and CLASS is the spesific class for the images. This is to ensure that the sets are randomly split before training the neural network.

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

#### For classification
1. You can train the images on a CNN using the following command:

`python SinfNet.py --cnn_train CNN` or `python SinfNet.py -ct CNN`

Where CNN is the name of the convolutional neural network that you want to use. Choose one of the following [VGG16, VGG19, ResNet50, DenseNet201].

2. Training, loss, and confusion matrix figures will be generated after the training is done. An evaluation on the test set will be performed and the result printed on the terminal.

### Detection
#### For object detection
1. Download the relevant weights file (links available in table above) or generate the file from the steps above.

2. Detect objects in your image/video/webcam using the following command:

`python SinfNet.py --yolo_predict WEIGHTS.h5 FILENAME LABELS` example `python SinfNet.py -yp Amoeba.h5 image.jpg Active Inactive`

Where WEIGHTS.h5 is the weights file, the FILENAME can be either a .jpg image, .mp4 video, or a webcam input, and the LABELS is a list of all the labels in the dataset (just the labels written with space between them).

#### For classification
1. Download the relevant weights file (links available in table above) or generate the file from the steps above.

2. To run a classification use the following command:

`python SinfNet.py --cnn_predict CNN WEIGHTS FILENAME` or `python SinfNet.py -cp ResNet50 Nematodes.h5 image.jpg`

Where the CNN is the name of the network that was used to generate the WEIGHTS.h5 file (using different networks from the weights file does not work), WEIGHTS is the name of the .h5 weights file, and FILENAME is the name of the image file.



















## Auto Annotation:
The Cells dataset was developed to make annotating images with different cells easier.

1. Change the detection threshold from 0.5 to 0.8 in line 1365 `obj_thresh, nms_thresh = 0.50, 0.45` to `obj_thresh, nms_thresh = 0.80, 0.45`.

2. Comment out the last line of the script `#cv2.imwrite ...` line 1434 of the YOLOv3.py script as to not generate images.

3. Use the following command to loop through all images and detect the cells:

`for f in ./DIRECTORY/*; do python YOLOv3.py -d WEIGHTS.h5 $f >> DIRECTORY; done`

Where DIRECTORY is the name of the directory that contains all the images.

4. Then use the following command to generate the BBox_Annotation text files:

`python SinfNet.py --convert` or `python SinfNet.py -c`

5. Check all images to make sure the the annotations are correct, and to correct minor errors.

6. Translate the text files into .xml files.

**You can add to an existing dataset**. Since manual annotations are time consuming, this same neural network can be used to annotate new images to build a new dataset (instead of annotating 1000s of images manually), make sure you use the Cell.h5 weights file if you want to detect only the cells in the images. You must insure your images are made up of a pure cell strain. Use the following command to loop through all images in a directory and annotate them:

`for i in IMAGE_DIRECTORY/*; do f="${i##*/}"; python YOLOv3.py -d Cell.h5 $i > ./"${f%.*}".txt; rm ./"${i##*/}"; sed -i "s/[^ ]*$/Cell/" ./"${f%.*}".txt; cat ./"${f%.*}".txt | wc -l > temp && cat ./"${f%.*}".txt >> temp && mv temp ./"${f%.*}".txt; mv ./"${f%.*}".txt ./BBox_Test; done`

I know this command is ugly, but it works. The only thing you have to change is the *IMAGE_DIRECTORY* at the start of the command. The annotation is as good as the training of the network, which is not 100%, therefore a human must go over the annotated images to fix any minor mistakes.

**Contributing to our dataset**
If you would like to add images to our dataset (any type of microscopic organism) make sure that each species has 200 annotated images where each image is sharp and taken from a brightfield light miscroscope at 400x magnification. Please contact me so we can work together.















## Table of commands:
|Command                                         |Description                    |
|------------------------------------------------|-------------------------------|
python SinfNet.py -h                             |Help                           |
python SinfNet.py -a NUMBER OF IMAGES            |Augment                        |
python SinfNet.py -v                             |Open weg-based immage annotator|
python SinfNet.py -b                             |BBox (NOT USED)                |
python SinfNet.py -tc                            |Convert .cvs to .xml           |
python SinfNet.py -tx                            |Convert .txt to .xml (NOT USED)|
python SinfNet.py -yt WEIGHTS PROJECT_NAME LABELS|YOLOv3 network train           |
python SinfNet.py -yp WEIGHTS FILENAME LABELS    |YOLOv3 network detect          |
python SinfNet.py -ct CNN                        |CNN network train              |
python SinfNet.py -cp CNN WEIGHTS FILENAME       |CNN network classify           |

## Funders:
* [Experiment](https://experiment.com/)
* [Microsoft](https://www.microsoft.com/en-us/ai/ai-for-earth-tech-resources)

## References:
When using any part of this project kindly reference the following:
* 

## TODO:
* Make Video
* Add reference
