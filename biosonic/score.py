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


# classification threshold
THRESHOLD = 0.7


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
        # extract true class from filename
        true_class = extract_true_label(entry['id'])
        for annotation in entry['annotation']:
            y_true.append(true_class)
            # if predicted class matches true class, use class_prob
            if annotation['class'] == true_class:
                y_pred.append((annotation['class'], annotation['class_prob']))
            # if predicted class does not match true class, use 1 - class_prob
            else:
                y_pred.append((annotation['class'], 1 - annotation['class_prob']))
    
    return y_true, y_pred


def calculate_metrics(y_true, y_pred):
    # normalize the case of class names for accurate comparison
    y_pred = [(y[0].lower(), y[1]) for y in y_pred]

    y_true_binary = []
    y_pred_binary = []
    y_scores = []
    
    for i in range(len(y_true)):
        # get probabilities for AUC
        y_scores.append(y_pred[i][1])

        # True Positive: Prediction and actual are the same, and probability is above threshold
        if y_true[i] == y_pred[i][0] and y_pred[i][1] >= THRESHOLD:
            y_true_binary.append(1)
            y_pred_binary.append(1)
        # False Negative: Prediction and actual are the same, but probability is below threshold
        elif y_true[i] == y_pred[i][0] and y_pred[i][1] < THRESHOLD:
            y_true_binary.append(1)
            y_pred_binary.append(0)
        # False Positive: Prediction and actual are different, but probability is above threshold
        elif y_true[i] != y_pred[i][0] and y_pred[i][1] >= THRESHOLD:
            y_true_binary.append(0)
            y_pred_binary.append(1)
        # True Negative: Prediction and actual are different, and probability is below threshold
        else:
            y_true_binary.append(0)
            y_pred_binary.append(0)


    # auroc only possible with atleast 1 TP
    if len(set(y_true_binary)) == 1:
        auroc = None
    else:
        auroc = roc_auc_score(y_true_binary, y_scores)

    balanced_acc = balanced_accuracy_score(y_true_binary, y_pred_binary)
    precision, recall, _ = precision_recall_curve(y_true_binary, y_scores)
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