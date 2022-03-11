import numpy as np


def binary_classification_metrics(y_pred, y_true):
    """
    Computes metrics for binary classification
    Arguments:
    y_pred, np array (num_samples) - model predictions
    y_true, np array (num_samples) - true labels
    Returns:
    precision, recall, f1, accuracy - classification metrics
    """

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score

    y_pred = np.array(y_pred.astype(int).astype(str), dtype = object)
    codes = y_true + y_pred
    TP = np.sum( (codes == "11").astype(int) )
    TN = np.sum( (codes == "00").astype(int) )
    FP = np.sum( (codes == "01").astype(int) )
    FN = np.sum( (codes == "10").astype(int) )
    accuracy = (TP + TN)/(TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = precision * recall * 2 / (precision + recall)
    return precision, recall, f1, accuracy



def multiclass_accuracy(y_pred, y_true):
    """
    Computes metrics for multiclass classification
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true labels
    Returns:
    accuracy - ratio of accurate predictions to total samples
    """
    y_pred = np.array(y_pred.astype(int).astype(str), dtype=object)
    codes = y_true + y_pred
    All = codes.shape[0]
    TP = np.sum(
                (codes == "00").astype(int) |
                (codes == "11").astype(int) |
                (codes == "22").astype(int) |
                (codes == "33").astype(int) |
                (codes == "44").astype(int) |
                (codes == "55").astype(int) |
                (codes == "66").astype(int) |
                (codes == "77").astype(int) |
                (codes == "88").astype(int) |
                (codes == "99").astype(int)
                )
    accuracy = TP / All
    return accuracy



def r_squared(y_pred, y_true):
    """mse
    Computes r-squared for regression
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    r2 - r-squared value
    """
    delta_mean = np.sum((y_true - np.mean(y_true)) ** 2)
    delta_pred = np.sum((y_true - y_pred) ** 2)
    R_sq = 1 - delta_pred/delta_mean
    return R_sq


def mse(y_pred, y_true):
    """
    Computes mean squared error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mse - mean squared error
    """
    MSE = np.sum((y_true - y_pred) ** 2 / y_true.shape[0])
    return MSE



def mae(y_pred, y_true):
    """
    Computes mean absolut error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mae - mean absolut error
    """
    MAE = np.sum(np.abs(y_true - y_pred) / y_true.shape[0])
    return MAE
    b