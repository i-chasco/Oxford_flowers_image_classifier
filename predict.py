import argparse
import json
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from helper import load_model, predict, process_image, plot
from sys import argv

json_path = None

parser = argparse.ArgumentParser(description='option parser')

parser.add_argument('image_path', action="store")
parser.add_argument('saved_model_path', action="store")
parser.add_argument('--top_k', action="store",
                    dest="top_k", type=int, default=1)
parser.add_argument('--category_names', action="store",
                    dest="json_path", default=None)
args = parser.parse_args()

image_path = args.image_path
saved_model_path = args.saved_model_path
top_k = args.top_k
json_path = args.json_path

def main(image_path, saved_model_path, top_k=1, json_path=None):
    '''
    Input: image_path, saved_model_path, top_k, json_path
    Output: plot with the photo provided with the predicted class as title, together with a histogram
            with top_k most likely classes and their probabilities.
    '''
    model = load_model(saved_model_path)
    probs, class_probs, image = predict(image_path, model, top_k)
    plot(image, probs, class_probs, json_path, top_k)

if __name__ == "__main__":
    main(image_path, saved_model_path, top_k, json_path)




