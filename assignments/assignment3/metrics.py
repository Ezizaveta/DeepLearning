import numpy as np

def binary_classification_metrics(prediction, ground_truth):
    tp = np.sum([a and b for a, b in zip(prediction, ground_truth)])
    tn = np.sum([not a and not b for a, b in zip(prediction, ground_truth)])
    fp = np.sum([a and not b for a, b in zip(prediction, ground_truth)])
    fn = np.sum([not a and b for a, b in zip(prediction, ground_truth)])

    accuracy = (tp + tn) / len(ground_truth)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    return np.where(prediction == ground_truth)[0].shape[0] / prediction.shape[0]

#def multiclass_accuracy(prediction, ground_truth):
#    return 0
