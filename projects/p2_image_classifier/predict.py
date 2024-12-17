import argparse
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
from PIL import Image
import numpy as np
import json

def preprocess_image(image):
    """Prepare the input image for the model by resizing and normalizing it."""
    image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
    resized_image = tf.image.resize(image_tensor, (224, 224))
    normalized_image = resized_image / 255.0
    return normalized_image.numpy()

def prediction(image_path, model, num_top_predictions=5):
    """Use the trained model to predict the top classes for an input image."""
    try:
        img = Image.open(image_path)
        img_array = np.asarray(img)
    except Exception as err:
        raise ValueError(f"Failed to load image from path '{image_path}': {err}")
    
    processed_image = preprocess_image(img_array)
    input_tensor = np.expand_dims(processed_image, axis=0)
    
    predictions = model.predict(input_tensor)
    top_indices = np.argsort(predictions[0])[-num_top_predictions:][::-1]

    top_probabilities = predictions[0][top_indices]
    top_classes = [str(idx) for idx in top_indices]
    return top_probabilities, top_classes

def parse_cl_arguments():
    """Parse command-line arguments using argparse."""
    parser = argparse.ArgumentParser(description='Predict the class of a flower from an image and display the probabilities.')
    parser.add_argument('image_path', type=str, help='File path to the input image')
    parser.add_argument('model_path', type=str, help='Path to the saved trained model')
    parser.add_argument('--top_k', type=int, default=5, help='Number of top predictions to display')
    parser.add_argument('--category_names', type=str, default=None, help='Path to JSON file mapping labels to flower names')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    return parser.parse_args()

def load_model(model_path):
    print(model_path)
    try:
        return keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer})
    except Exception as err:
        raise ValueError(f"Error loading model from '{model_path}': {err}")

def load_label_mapping(json_filepath):
    if json_filepath:
        try:
            with open(json_filepath, 'r') as file:
                return json.load(file)
        except Exception as err:
            raise ValueError(f"Error loading label mapping from '{json_filepath}': {err}")
    return {}

def display_predictions(probabilities, class_labels):
    print("Top Predictions:")
    for prob, label in zip(probabilities, class_labels):
        print(f"Class: {label}, Probability: {prob:.4f}")


def main():
    args = parse_cl_arguments()
    model = load_model(args.model_path)
    probabilities, class_labels = prediction(args.image_path, model, args.top_k)
    
    if args.category_names:
        label_mapping = load_label_mapping(args.category_names)
        class_labels = [label_mapping.get(cls, "Unknown") for cls in class_labels]
    
    display_predictions(probabilities, class_labels)
if __name__ == '__main__':
    main()
