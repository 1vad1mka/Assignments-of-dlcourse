def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    import numpy as np
    
    confusion_matrix = np.zeros((2, 2))
    tp, tn, fp, fn = 0, 0, 0, 0
    
    for i in range(len(ground_truth)):
        if (ground_truth[i] == prediction[i]) and (ground_truth[i] == True):
            tp += 1
        elif (ground_truth[i] == prediction[i]) and (ground_truth[i] == False):
            tn += 1
        elif (ground_truth[i] == False) and (prediction[i] == True):
            fp += 1
        elif (ground_truth[i] == True) and (prediction[i] == False):
            fn += 1
    
    confusion_matrix[0][0] = tp
    confusion_matrix[0][1] = fp
    confusion_matrix[1][0] = fn
    confusion_matrix[1][1] = tn
    
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    f1 = 2 * (precision * recall) / (precision + recall)

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    
    import numpy as np

    # TODO: Implement computing accuracy
    return (ground_truth == prediction).sum() / ground_truth.shape[0]
