import argparse
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
from PIL import Image
import numpy as np
import json

def process_image(image):
    """Process the image to be suitable for the model."""
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.image.resize(image, (224, 224))
    image /= 255.0
    return image.numpy()

def predict(image_path, model, top_k=5):
    """Predict the class of an image using a trained model."""
    im = Image.open(image_path)
    image = np.asarray(im)
    processed_image = process_image(image)
    image_batch = np.expand_dims(processed_image, axis=0)
    predictions = model.predict(image_batch)
    top_k_indices = predictions[0].argsort()[-top_k:][::-1]
    top_k_probs = predictions[0][top_k_indices]
    top_k_classes = [str(index) for index in top_k_indices]
    return top_k_probs, top_k_classes

def parse_args():
    parser = argparse.ArgumentParser(
        description='Predict flower name from an image along with the probability of that name.')
    parser.add_argument('image_path', type=str, help='Path to the image')
    parser.add_argument('model_path', type=str, help='Path to the trained model')
    parser.add_argument('--top_k', type=int, default=5, help='Return the top K most likely classes')
    parser.add_argument('--category_names', type=str, default=None,
                        help='Path to a JSON file mapping labels to flower names')
    return parser.parse_args()

def load_model(model_path):
    model = keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer})
    return model

def load_category_names(category_names_path):
    if category_names_path:
        with open(category_names_path, 'r') as f:
            category_names = json.load(f)
        return category_names
    return None

def load_class_names(json_file):
    with open(json_file, 'r') as f:
        class_names = json.load(f)
    return class_names

def main():
    args = parse_args()
    model = keras.models.load_model(args.model_path, custom_objects={'KerasLayer': hub.KerasLayer})
    probs, classes = predict(args.image_path, model, args.top_k)
    
    # Map classes to names if category_names is provided
    if args.category_names:
        class_names = load_class_names(args.category_names)
        classes = [class_names.get(c, "Unknown") for c in classes]
    
    # Print out the results
    print("Top K Predictions:")
    for prob, class_name in zip(probs, classes):
        print(f"Class: {class_name}, Probability: {prob}")

if __name__ == '__main__':
    main()
