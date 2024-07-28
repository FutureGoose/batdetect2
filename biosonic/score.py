"""
This script calculates macro-averaged accuracy and accuracy for the detection outputs.

It processes the results output data from running inference with batdetect2.
When running 'detect' with batdetect2, you get one JSON file (and one CSV file) as the result of inference on each audio file.

This script recursively traverses all files in the specified folders, collecting data from all JSON files. The metric is calculated by comparing 
the true label (the 'id' entry, also known as the filename) with the overall model prediction from the file (the 'class_name' prediction).
"""


import os
import json
import glob
from pathlib import Path
from sklearn.metrics import accuracy_score
import numpy as np


def load_json_files(data_dir):
    """Load all JSON files from directory."""
    json_files = glob.glob(os.path.join(data_dir, '**/*.json'), recursive=True)
    print(f"Found {len(json_files)} JSON files.\n")
    if len(json_files) == 0:
        print(f"No JSON files found in directory: {data_dir}")

    data = []
    for json_file in json_files:
        # print(f"Loading JSON file: {json_file}")
        with open(json_file, 'r') as f:
            data.append(json.load(f))

    return data


def extract_true_label(filename):
    """Extract true class from filename."""
    # split the filename by underscores
    parts = filename.split('_')
    
    # find the first part that contains a number and stop there
    for i, part in enumerate(parts):
        if any(char.isdigit() for char in part):
            break
    
    # join the parts up to the first number to form the true label
    true_label = ' '.join(parts[:i])
    
    return true_label


def extract_labels_and_predictions(data):
    """Extract true labels and predictions from json files."""
    y_true = []
    y_pred = []
    for entry in data:
        # collect y_true
        true_class = extract_true_label(entry['id'])
        y_true.append(true_class)
        # collect y_pred
        pred_class = entry['class_name'].lower()
        y_pred.append(pred_class)
    
    return y_true, y_pred


def macro_averaged_accuracy(y_true_labels, y_pred_labels):
    "Calculates macro-averaged accuracy"
    # dict with list for adding binary pred to each class
    class_predictions = {x: [] for x in np.unique(y_true_labels)}
    accuracies = []

    # print(f'{class_predictions=}\n')
    # print(f'{y_true_labels=}\n')
    # print(f'{y_pred_labels=}\n')

    # positives and negatives as binary
    for i in range(len(y_true_labels)):
        y_true = y_true_labels[i]
        y_pred = y_pred_labels[i]
        # if same, it's a positive
        if y_true == y_pred:
            class_predictions[y_true].append(1)
        # if not, it's a negative, regardless of pred class
        else:
            class_predictions[y_true].append(0)

    # mean of binaries equals accuracy
    for v in class_predictions.values():
        class_accuracy = np.mean(v)
        accuracies.append(class_accuracy)

    # mean of accuracies equals macro averaged accuracy
    macro_avg_accuracy = np.mean(accuracies)

    return macro_avg_accuracy


def main(data_dir):
    """Main function to process data and print metrics."""
    # load json files
    data = load_json_files(data_dir)

    # get predictions
    y_true, y_pred = extract_labels_and_predictions(data)

    # calculate and print macro-averaged accuracy
    macro_avg_acc = macro_averaged_accuracy(y_true, y_pred)
    print(f"Macro-Averaged Accuracy: {macro_avg_acc}\n")

    # calculate and print overall accuracy
    overall_acc = accuracy_score(y_true, y_pred)
    print(f"Overall Accuracy: {overall_acc}\n")

if __name__ == "__main__":
    # universal path handling
    script_dir = Path(__file__).resolve().parent
    data_dir = (script_dir / "../data/results").resolve()
    print(f"Data directory: {data_dir}")
    main(data_dir)