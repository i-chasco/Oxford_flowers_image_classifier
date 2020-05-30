### Date created
This project was created May 28th, 2020.

### Project Title
Oxford_flowers_image_classifier

### Description
Keras/TensorFlow project done as part of the Udacity's Introduction to Machine Learning Nanodegree. It consists on a modified version of Google's MobileNet Convolutional Neural Network, modified with a last layer to adap it to a 102-class flower classifier.

The classifier was trained with the oxford_flowers102 dataset from TensorFlow.

The saved model can be run from the predict.py script and can work with any image (it will try to classify it as one of the 102 flower classes that the model was trained on).

### Files used
predict.py  - main script
helper.py - contains helper functions
label_map.json - a label map that will allow the model to give flower name predictions instead of just the predicted class number
flower_classifier_project.h5 - keras model of the modified and pretrained model
Project_Image_Classifier_Project.ipynb - notebook used for the model creation (check it if you need details on the model creation and training process)

### How to use the script
File containing the script: predict.py

Basic usage:
python predict.py image_path saved_model_path

Additional options:
--top_k: the model gives class probabilities for the top k number of classes. By default it only predicts and gives probabilities for the top 1 class
--category_names: the model used the provided label map (in json format) to assign label names to the predictions (the model predicts label numbers by default)

Usage with options (example with real file paths):
python predict.py blue_grape_hyacinth.jpg flower_classifier_project.h5 --top_k 4 --category_names label_map.json  --> returns predictions with class names and probabilities for the top 4 classes

### Credits
Project guidance provided by the Udacity instructors.
