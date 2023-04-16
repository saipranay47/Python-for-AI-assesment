# Python-for-AI-assesment
Weed Classification in Karnataka Region
=======================================

This is a deep learning project for identifying the type of weed growing in the farmlands of the Karnataka Region. The project uses a dataset of three different classes of weeds: CELOSIA ARGENTEA L, CROWFOOT GRASS, and PURPLE CHLORIS. The deep learning model is trained using TensorFlow and Keras and uses the MobileNetV2 architecture for feature extraction.

Dataset
-------

The dataset used in this project is provided in the `dataset` folder. The dataset contains images of weeds collected from farmlands in the Karnataka Region. The images are split into three classes based on the type of weed: CELOSIA ARGENTEA L, CROWFOOT GRASS, and PURPLE CHLORIS.

Model Training
--------------

The deep learning model is trained using the `weed_classification_karnataka.py` script. The script uses TensorFlow and Keras to build and train the model. The pre-trained MobileNetV2 model is used as the feature extractor, followed by a few additional layers for classification. The model is trained on the images in the `dataset` folder using an ImageDataGenerator. The trained model is saved as `weed_classification_karnataka_model.h5`.

Inference
---------

To use the trained model for inference, you can modify the `weed_classification_karnataka.py` script. The script can either take an input image file for prediction or use a live webcam feed for real-time prediction.

Requirements
------------

-   Python 3.6+
-   TensorFlow 2.0+
-   Keras 2.3+
-   OpenCV 4.0+

### Inference

#### Image Prediction

To use an input image file for prediction, first place the image file in the repository directory and rename it as `test.png`. Then, run the following command:

Copy code

`python weed_classification_karnataka.py`

The script will load the trained model from `weed_classification_karnataka_model.h5`, predict the class of the weed in the image, and display the result on the image.

#### Webcam Prediction

To use a live webcam feed for real-time prediction, run the following command:

Copy code

`python weed_classification_karnataka.py`

The script will use your webcam to capture a live feed, predict the class of the weed in the frame, and display the result on the frame. Press 'q' to exit the script.
