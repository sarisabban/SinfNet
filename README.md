# Protist Image Classifier
A neural network for protists image classification.

## Description:
This is a script that uses a real-time object detection convolutional neural network called YOLOv3 to detection cells present in an image, or classify different species of protists from a microscope image. Provided here are all the necessary scripts to develop a database, train the **YOLOv3** networks, and perform a detection. Microscope image datasets and pre-trained weights are also available where we trained this neural network to detect cells or to classify the following species within an image:

<p align="center">Protist dataset - 10 species:</p>

<sub>*Colsterium ehrenbergii - Cylindrocystis brebissonii - Lepocinclis spirogyroides - Micrasterias rotata - Paramecium bursaria - Peridinium spec. - Pinnularia neomajor - Pleurotaenium ehrenbergii - Pyrocystis lunula - Volvox tertius*</sub>

The cell detection setup it mainly used to construct the protist dataset, since each protist species require at least 1000 annotated images, the cell detection neural network is use to auto annotate microscope cell images and construct a dataset, that is why it is provided here.

## Available datasets and trained weight files
You can download these datasets to add to them and re-train the network to develop it, or to simply replicate our work:

|Dataset Name                                                                                                    |Weights                                                                      |mAP or Accuracy|
|----------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------|---------------|
|[Amoeba Active/Inactive Dataset](https://www.dropbox.com/s/247tkdqd9cskn00/Amoeba.tar.bz2?dl=0)                 |[YOLOv3 Weights](https://www.dropbox.com/s/x044cdo7kznoeuf/Amoeba.h5?dl=0)   |0.6473         |
|[Cell Detection Dataset](https://www.dropbox.com/s/pl5vi4rr8nsea37/Cells.tar.bz2?dl=0)                          |[YOLOv3 Weights](https://www.dropbox.com/s/yukp34x3gaubd4u/Cells.h5?dl=0)    |0.9549         |
|[Algae Classification Dataset](https://www.dropbox.com/s/jcvdicvl1eg6o6z/Algae.tar.bz2?dl=0) or [Algae Classification Dataset_Augmented](https://www.dropbox.com/s/a78qfnuaedcspxm/Algae.tar?dl=0)                    |[YOLOv3 Weights](https://www.dropbox.com/s/4o9peiulwizsa72/Algae.h5?dl=0)    |0.7118         |
|[Nematode Detection Dataset](https://www.dropbox.com/s/ah7rhrf7f0etfw5/Nematodes.tar.bz2?dl=0)                  |[YOLOv3 Weights](https://www.dropbox.com/s/z638ml32x7i3kef/Nematodes.h5?dl=0)|0.8867         |
|[Nematode Feeding Classification Dataset]()    |[CNN Weights]()   |         |

## How to use:
This is a [Video]() on how to use this setup.

### Update your system and install libraries.
This script works on GNU/Linux Ubuntu 18.04 and over using Python 3.6 and over. To use this script you will need to first update your system and install the dependencies using the following commands:

`sudo apt update`

`sudo apt full-upgrade`

`sudo apt install python3-pip python3-opencv`

`pip3 install numpy keras tensorflow tkintertable matplotlib imgaug scipy`

### Setting up a dataset
If you want to develop your own dataset follow these steps:

1. Collect images containing your objects. Even though the network can process different image formats, it is best to stick with the .jpg image format.
2. Make a directory called dataset and within it in make the following directories: Images, BBox_Annotations, Annotations, and Check. You should have the following structure:

        *./dataset/Annotations*

        *./dataset/BBox_Annotations*

        *./dataset/BBox_Test*

        *./dataset/Check*

        *./dataset/Predictions*

        *./dataset/Test*

        *./dataset/Train*

        *./dataset/Valid*

This command will quickly set it up:

`mkdir -p ./dataset/Annotations ./dataset/Train ./dataset/Test ./dataset/BBox_Test ./dataset/BBox_Test_Predictions ./dataset/BBox_Annotations ./dataset/Check ./dataset/Valid`

Your images should be in *./dataset/Images* of course. It is best to stick to this structure with these names exactly, otherwise you will have to change these path names within each relevant script. So just stick to these to keep it simple.
If you would like to augment the images use the following command:

`python3 ProtiClass.py --augment` or `python3 ProtiClass.py -a`

This will generate a new directory with all the saved augmented images on it, and these augmented images should be used for training.

Regarding using the CNN neural network for image classification, the dataset should be have the following directory architecture. Within each directory a directory of the classes that includes all the images of that class, as such:

    *./dataset/Test/*

                  *class1/*

                        *image1.jpg*

                        *image2.jpg*

                  *class2/*
                        *image1.jpg*

                        *image2.jpg*

                  *class3/
                        *image1.jpg*

                        *image2.jpg*

    *./dataset/Train*

                  *class1/*

                        *image1.jpg*

                        *image2.jpg*

                  *class2/*
                        *image1.jpg*

                        *image2.jpg*

                  *class3/
                        *image1.jpg*

                        *image2.jpg*
    *./dataset/Valid*

                  *class1/*

                        *image1.jpg*

                        *image2.jpg*

                  *class2/*
                        *image1.jpg*

                        *image2.jpg*

                  *class3/
                        *image1.jpg*

                        *image2.jpg*

3. Open the GUI annotation tool using the following command:

`python3 ProtiClass.py --bbox` or `python3 ProtiClass.py -b`

4. You will be prompted to enter the labels, enter a label and press enter to enter a new label. Type `end` to start labelling when you have enters all your desired labels.
5. (Should be already setup), click "Image Input Folder" on the top left to choose the directory that contains the images (./dataset/Images).
6. (Should be already setup), click "Label Output Folder" on the top left to choose the directory that will save the labels (./dataset/BBox_Annotations).
7. Click "Load Dir" on the top right to load your choices (nothing will happen). Note: It is better to stick to the default dataset paths mentioned in step 3, otherwise you will have to changes to different paths from within the code in some scripts. The images may not scale very well, make sure you see the entire image and not just part of it, change the values (currently at 700) in line 280 of the BBox.py script accordingly (larger values = more zoomed image).
8. You must click "Next" to load the images (but it will skip the first image, so go back to it) then previous to go back to the first image.
9. Use the mouse to generate a bounding box around each object of interest.
10. Label each box with the labels from the drop down menu on the right and clicking "Confirm Class".
11. Click "Next >>" to save the labels and move on to the next image (images are not loaded by filename order).
12. Once finished, check to make sure that your annotations are correct by using the following command:

`python3 ProtiClass.py --check` or `python3 ProtiClass.py -k`

This will generate a new directory called ./dataset/Check with the images showing their annotations.

13. The annotations are in text (.txt) file format and they need to be in XML format, to convert run the following command:

`python3 ProtiClass.py --translate` or `python3 ProtiClass.py -t`

This will generate a new directory called ./dataset/Annotations and this directory will be used in the neural network.

14. If you want to rename a label throughout the dataset use the following command

`python3 ProtiClass.py --rename OLD NEW`  or `python3 ProtiClass.py -r OLD NEW`

For help use this command:

`python3 ProtiClass.py --help` or `python3 ProtiClass.py -h`

15. Add some images to the Test direcotry which will be used to test the accuracy of the final trained network (on images the network has never seen). Annotate these images and add the annotations to ./dataset/BBox_Test. After you train the neural network go back to the dataset directory and run the following command (make sure you change the WEIGHTS.h5 to your corresponding weights filename):

`for i in Test/*; do f="${i##*/}"; python3 YOLOv3.py -d WEIGHTS.h5 $i > ./"${f%.*}".txt; rm ./"${i##*/}"; cat ./"${f%.*}".txt | wc -l > temp && cat ./"${f%.*}".txt >> temp && mv temp ./"${f%.*}".txt; mv ./"${f%.*}".txt ./BBox_Test; done`

This command will run the neural network to predict all images in the test directory and outputs its own BBOX text files. Using the information in the ./dataset/BBox_Test and ./dataset/BBox_Test_Predictions you can evaluate how accurate the neural network is at correctly classifying the cells. You can run the evaluation using this command:

`python3 ProtiClass.py --eval` or `python3 ProtiClass.py -e`

The output will be percent accuracy.

### Training the neural network
#### CNN
1. You can train the images on a CNN using the following command:
2. The labels for each class should be directories within the *./dataset/Train* directory including the images of only that class.
3. There must be images in the *./dataset/Train*, *./dataset/Test*, and *./dataset/Valid* directories for this architecture to work.

`python3 ProtiClass.py --cnn_train CNN` or `python3 ProtiClass.py -ct CNN`

Where CNN stands for one of the following CNN architectures: VGG16, VGG19, ResNet50, or DenseNet201
The network will look for the images in the *./dataset/Images* directory and will augment them 10 times before pushing them through the neural network.

#### Object
1. On line 53 of the YOLOv3.py script add all your labels in a list as such ["label 1", "label 2", "label 3"], and on lines 57 and 58 change your output file names.
2. The network is resource heavy and required a large GPU and more than 16GB of RAM to run. Therefore some cloud GPU cloud services may not work and a larger system is required.
3. To see the help menu use the following command:

`python3 YOLOv3.py -h`

4. Run training using the following command:

`python3 YOLOv3.py -t`

5. If the neural network training does not go well, you will have to change the network hyper parameters which are found in lines 49-79 of the script file.
6. The logs directory contains the training logs. View the data using the following command:

`tensorboard --logdir=./logs`

7. The .h5 file is the weights file used for image detection.

8. If the training is interrupted, you can use the weight.h5 file to continue where you left off.

### Detection
#### CNN
1. To run a prediction use the following command:

`python3 ProtiClass.py --cnn_predict CNN IMAGE` or `python3 ProtiClass.py -cp CNN IMAGE`

You must have the weights.h5 file in the same directory as the ProtiClass.py script and identify which CNN the the weights where trained on.

#### Object
If you just want to run a detection without developing a dataset nor re-training the network you can just run this command right now using the weights of our trained network.
1. Download the relevant weights file, links available above.
2. Run image detection using the following command:

`python3 YOLOv3.py -d WEIGHTS.h5 FILENAME`

The FILENAME in YOLOv3.py can be either a .jpg image, .mp4 video or a webcam.

**You can add to an existing dataset**. Since manual annotations is time consuming, this same neural network can be used to annotate new images to expand the current dataset (instead of annotating 1000s of images manually), make sure you use the Cell.h5 weights since you want to detect only the cells in the images. Thus you must insure your image is made up of a pure strain of cells. Use the following command to loop through all images in a directory and annotate them.:

`for i in IMAGE_DIRECTORY/*; do f="${i##*/}"; python3 YOLOv3.py -d Cell.h5 $i > ./"${f%.*}".txt; rm ./"${i##*/}"; sed -i "s/[^ ]*$/Cell/" ./"${f%.*}".txt; cat ./"${f%.*}".txt | wc -l > temp && cat ./"${f%.*}".txt >> temp && mv temp ./"${f%.*}".txt; mv ./"${f%.*}".txt ./BBox_Test; done`

I know the command is ugly, but it works. The only thing you have to change is the *IMAGE_DIRECTORY* at the start of the command. The annotation is as good as the training of the network, which is not 100%, therefore a human must go over the annotated images using the ProtiClass.py script as in step 4 to fix any mistakes. Make sure you repeat steps 12 and 13 to check and translate the new annotations. Annotations may have some mistakes therefore checking the annotations is very important.

**Contributing to our dataset**
If you would like to add images to our dataset (any type of protist cell) make sure that each species has 2000 annotated images where each image is sharp and taken from a brightfield light miscroscope at 400x magnification. Please contact me so we can work together.

## Auto Annotation:
The Cells dataset was developed to make annotating images with different cells easier.

1. Change the detection threshold from 0.5 to 0.8 in line 1365 `obj_thresh, nms_thresh = 0.50, 0.45` to `obj_thresh, nms_thresh = 0.80, 0.45`.

2. Comment out the last line of the script `#cv2.imwrite ...` line 1434 of the YOLOv3.py script as to not generate images.

3. Use the following command to loop through all images and detect the cells:

`for f in ./DIRECTORY/*; do python3 YOLOv3.py -d WEIGHTS.h5 $f >> DIRECTORY; done`

Where DIRECTORY is the name of the directory that contains all the images.

4. Then use the following command to generate the BBox_Annotation text files:

`python3 ProtiClass.py --convert` or `python3 ProtiClass.py -c`

5. Check all images to make sure the the annotations are correct, and to correct minor errors.

6. Translate the text files into .xml files.

## Funders:
* [Experiment](https://experiment.com/)
* [Microsoft](https://www.microsoft.com/en-us/ai/ai-for-earth-tech-resources)

## References:
When using these scripts kindly reference the following:
* 
