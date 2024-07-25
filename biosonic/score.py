"""
This script processes the results output data from running inference with batdetect2.
When running 'detect' with batdetect2, you get one JSON file (and one CSV file) as the result of inference on each audio file.
The script calculates the metrics AUROC, AUPRC, and Balanced Accuracy for the detection outputs.
It recursively goes through all files in the specified folders, collects data from all JSON files, and aggregates them.
Each JSON file corresponds to an audio file and contains multiple predictions due to various patterns in the spectrograms.
This function compares every single prediction per JSON file with the true label, which is derived from the filename.
The filename is present in the 'id' entry at the bottom of each JSON file. Hence, there is only one true label per JSON file/audio file.
"""


import os
import json
import glob
from pathlib import Path
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, precision_recall_curve, auc


def load_json_files(data_dir):
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
    """Gets true class from id entry"""
    # Split the filename by underscores
    parts = filename.split('_')
    
    # Find the first part that contains a number and stop there
    for i, part in enumerate(parts):
        if any(char.isdigit() for char in part):
            break
    
    # Join the parts up to the first number to form the true label
    true_label = ' '.join(parts[:i])
    
    return true_label


def extract_labels_and_predictions(data):
    y_true = []
    y_pred = []
    for entry in data:
        true_class = extract_true_label(entry['id'])
        for annotation in entry['annotation']:
            y_true.append(true_class)
            y_pred.append(annotation['class'])
    
    return y_true, y_pred


def calculate_metrics(y_true, y_pred):
    # convert class names to binary labels
    classes = sorted(list(set(y_true)))
    print(classes)
   
    y_true_binary = [1 if y == classes[0] else 0 for y in y_true]
    y_pred_binary = [1 if y == classes[0] else 0 for y in y_pred]   


    # auroc only possible with atleast 1 TP
    if len(set(y_true_binary)) == 1:
        auroc = None
    else:
        auroc = roc_auc_score(y_true_binary, y_pred_binary)
    balanced_acc = balanced_accuracy_score(y_true_binary, y_pred_binary)
    
    precision, recall, _ = precision_recall_curve(y_true_binary, y_pred_binary)
    auprc = auc(recall, precision)
    
    return auroc, balanced_acc, auprc


def main(data_dir):
    data = load_json_files(data_dir)
    y_true, y_pred = extract_labels_and_predictions(data)
    auroc, balanced_acc, auprc = calculate_metrics(y_true, y_pred)

    # when no tps are present
    if auroc is not None:
        print(f"AUROC: {auroc}")
    else:
        print("AUROC: Not defined (only one class present in y_true)")

    print(f"AUPRC: {auprc}")
    print(f"Balanced Accuracy: {balanced_acc}")

if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    data_dir = (script_dir / "../data/results").resolve()
    print(f"Data directory: {data_dir}")

    main(data_dir)