import os
import json
import glob
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, precision_recall_curve, auc


def load_json_files(data_dir):
    json_files = glob.glob(os.path.join(data_dir, '**/*.json'), recursive=True)
    data = []
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data.append(json.load(f))
    return data



def extract_true_label(filename):
    """Gets true class from id entry"""
    # relevant parts
    parts = filename.split('_')[:2]
    # format result to match class names
    result = f'{parts[0].capitalize()} {parts[1]}'
    #print(result)
    return result


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
    classes = list(set(y_true))
    y_true_binary = [1 if y == classes[0] else 0 for y in y_true]
    y_pred_binary = [1 if y == classes[0] else 0 for y in y_pred]

    #print("Unique classes in y_true:", set(y_true))
    #print("Unique classes in y_pred:", set(y_pred))
    #print("y_true:", y_true)
    #print("y_pred:", y_pred)

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
    data_dir = "C:/wagon/code/biosonic_local/batdetect2/data/results"
    main(data_dir)