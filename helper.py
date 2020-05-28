import argparse
import json
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt



def process_image(image):
    '''
    Input: image path
    Output: Numpy array of image with dimmensions reduced to (224 x 224 x 3) and pixel values
    normalized so they are between 0 and 1
    '''
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (224 , 224))
    image /= 255
    image = image.numpy()
    return image

def load_model(model_path):
    '''
    Input: model path
    Output: loaded keras model
    '''
    loaded_model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer':hub.KerasLayer})
    return loaded_model

def predict(image_path, model, top_k=1):
    '''
    Input: image_path, model_path, top_k categories
    Output: if top_k == 1: predicted class and class probabilities
            if top_k >2: an array p
    '''
    image = np.asarray(Image.open(image_path))
    processed_image = process_image(image)
    expanded_processed_image = np.expand_dims(processed_image, axis=0)
    prediction_vector = model.predict(expanded_processed_image)

    probs = np.sort(prediction_vector)[0][::-1][:top_k]
    class_probs = np.argsort(prediction_vector)[0][::-1][:top_k]

    return probs, class_probs, image


def plot(image, probs, class_probs, json_path=None, top_k=1):
    '''
    Input: image as numpy array, probability array, class probability array, path to json with class number:class
            name dictionary, top_k classes to show
    Output: Plot with image titled as the most probable class and histogram with class probabilities
    '''
    if json_path:
        with open(json_path, 'r') as f:
            class_names = json.load(f)

    fig, (ax1, ax2) = plt.subplots(figsize=(7,7), ncols=2)
    ax1.imshow(image, cmap = plt.cm.binary)
    ax1.axis('off')
    # If category names was selected, use category names for the plot. Else use the category number
    if json_path:
        ax1.set_title(class_names[str(class_probs[0] + 1)])
        ax2.set_yticklabels([class_names[str(i + 1)] for i in class_probs], size='medium');
    else:
        ax1.set_title(f'Class {str(class_probs[0])}')
        ax2.set_yticklabels([str(class_probs[i]) for i, n in enumerate(class_probs)], size='medium');
    ax2.set_ylabel("Classes")
    ax2.set_xlabel("Probability")
    ax2.barh(np.arange(top_k), probs)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(top_k))
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()
    plt.show()
