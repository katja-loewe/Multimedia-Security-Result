import cv2
import numpy as np
#import sys
#import os
import matplotlib.pyplot as plt
#import scipy.misc
#import scipy.ndimage
#import skimage.filters
import sklearn.metrics
from sklearn.metrics import jaccard_score

"""
functions from 
https://techtutorialsx.com/2018/06/02/python-opencv-converting-an-image-to-gray-scale/
"""

def _assert_valid_lists(groundtruth_list, predicted_list):
    assert len(groundtruth_list) == len(predicted_list)
    for unique_element in np.unique(groundtruth_list).tolist():
        assert unique_element in [0, 1]


def _all_class_1_predicted_as_class_1(groundtruth_list, predicted_list):
    _assert_valid_lists(groundtruth_list, predicted_list)
    return np.unique(groundtruth_list).tolist() == np.unique(predicted_list).tolist() == [1]


def _all_class_0_predicted_as_class_0(groundtruth_list, predicted_list):
    _assert_valid_lists(groundtruth_list, predicted_list)
    return np.unique(groundtruth_list).tolist() == np.unique(predicted_list).tolist() == [0]


def get_confusion_matrix_elements(groundtruth_list, predicted_list):
    """
    Return confusion matrix elements covering edge cases
    :param groundtruth_list list of groundtruth elements
    :param predicted_list list of predicted elements
    :return returns confusion matrix elements i.e TN, FP, FN, TP in that order and as floats
    returned as floats to make it feasible for float division for further calculations on them
    """
    _assert_valid_lists(groundtruth_list, predicted_list)

    if _all_class_1_predicted_as_class_1(groundtruth_list, predicted_list) is True:
        tn, fp, fn, tp = 0, 0, 0, np.float64(len(groundtruth_list))

    elif _all_class_0_predicted_as_class_0(groundtruth_list, predicted_list) is True:
        tn, fp, fn, tp = np.float64(len(groundtruth_list)), 0, 0, 0

    else:
        tn, fp, fn, tp = sklearn.metrics.confusion_matrix(groundtruth_list, predicted_list).ravel()
        tn, fp, fn, tp = np.float64(tn), np.float64(fp), np.float64(fn), np.float64(tp)

    return tn, fp, fn, tp



def _all_class_0_predicted_as_class_1(groundtruth_list, predicted_list):
    _assert_valid_lists(groundtruth_list, predicted_list)
    return np.unique(groundtruth_list).tolist() == [0] and np.unique(predicted_list).tolist() == [1]


def _all_class_1_predicted_as_class_0(groundtruth_list, predicted_list):
    _assert_valid_lists(groundtruth_list, predicted_list)
    return np.unique(groundtruth_list).tolist() == [1] and np.unique(predicted_list).tolist() == [0]


def _mcc_denominator_zero(groundtruth_list, predicted_list, tn, fp, fn, tp):
    _assert_valid_lists(groundtruth_list, predicted_list)
    return (tn == 0 and fn == 0) or (tn == 0 and fp == 0) or (tp == 0 and fp == 0) or (tp == 0 and fn == 0)


def get_f1_score(groundtruth_list, predicted_list):
    """
    Return f1 score covering edge cases
    :param groundtruth_list list of groundtruth elements
    :param predicted_list list of predicted elements
    :return returns f1 score
    """
    _assert_valid_lists(groundtruth_list, predicted_list)
    tn, fp, fn, tp = get_confusion_matrix_elements(groundtruth_list, predicted_list)

    if _all_class_0_predicted_as_class_0(groundtruth_list, predicted_list) is True:
        f1_score = 1
    elif _all_class_1_predicted_as_class_1(groundtruth_list, predicted_list) is True:
        f1_score = 1
    else:
        f1_score = (2 * tp) / ((2 * tp) + fp + fn)

    return f1_score


def get_mcc(groundtruth_list, predicted_list):
    """
    Return mcc covering edge cases
    :param groundtruth_list list of groundtruth elements
    :param predicted_list list of predicted elements
    :return returns mcc
    """
    _assert_valid_lists(groundtruth_list, predicted_list)
    tn, fp, fn, tp = get_confusion_matrix_elements(groundtruth_list, predicted_list)

    if _all_class_0_predicted_as_class_0(groundtruth_list, predicted_list) is True:
        mcc = 1
    elif _all_class_1_predicted_as_class_1(groundtruth_list, predicted_list) is True:
        mcc = 1
    elif _all_class_1_predicted_as_class_0(groundtruth_list, predicted_list) is True:
        mcc = -1
    elif _all_class_0_predicted_as_class_1(groundtruth_list, predicted_list) is True:
        mcc = -1
    elif _mcc_denominator_zero(groundtruth_list, predicted_list, tn, fp, fn, tp) is True:
        mcc = -1
    else:
        mcc = ((tp * tn) - (fp * fn)) / (
            np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))

    return mcc


def get_accuracy(groundtruth_list, predicted_list):
    """
    Return accuracy
    :param groundtruth_list list of elements
    :param predicted_list list of elements
    :return returns accuracy
    """
    _assert_valid_lists(groundtruth_list, predicted_list)
    tn, fp, fn, tp = get_confusion_matrix_elements(groundtruth_list, predicted_list)

    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total

    return accuracy

def get_jaccard_coefficient(groundtruth_list, predicted_list):
    """
    :param groundtruth_list list of elements
    :param predicted_list list of elements
    :return returns jaccard similarity coefficient
    """
    jaccard_coef = jaccard_score(groundtruth_list, predicted_list)

    return jaccard_coef

def get_validation_metrics(groundtruth_list, predicted_list):
    """
    Return validation metrics dictionary with accuracy, f1 score, mcc after
    comparing ground truth and predicted image
    :param groundtruth_list list of elements
    :param predicted_list list of elements
    :return returns a dictionary with accuracy, f1 score, and mcc as keys
    one could add other stats like FPR, FNR, TP, TN, FP, FN etc
    """
    _assert_valid_lists(groundtruth_list, predicted_list)
    validation_metrics = {}

    validation_metrics["accuracy"] = get_accuracy(groundtruth_list, predicted_list)
    validation_metrics["f1_score"] = get_f1_score(groundtruth_list, predicted_list)
    validation_metrics["mcc"] = get_mcc(groundtruth_list, predicted_list)
    validation_metrics["jaccard_similarity"] = get_jaccard_coefficient(groundtruth_list, predicted_list)

    return validation_metrics

def get_confusion_matrix_intersection_mats(groundtruth, predicted):
    """
    Returns a dictionary of 4 boolean numpy arrays containing True at TP, FP, FN, TN.
    """
    confusion_matrix_arrs = {}

    groundtruth_inverse = np.logical_not(groundtruth)
    predicted_inverse = np.logical_not(predicted)

    confusion_matrix_arrs["tp"] = np.logical_and(groundtruth, predicted)
    confusion_matrix_arrs["tn"] = np.logical_and(groundtruth_inverse, predicted_inverse)
    confusion_matrix_arrs["fp"] = np.logical_and(groundtruth_inverse, predicted)
    confusion_matrix_arrs["fn"] = np.logical_and(groundtruth, predicted_inverse)

    return confusion_matrix_arrs


def get_confusion_matrix_overlaid_mask(image, groundtruth, predicted, alpha, colors):
    """
    Returns overlay the 'image' with a color mask where TP, FP, FN, TN are
    each a color given by the 'colors' dictionary
    """
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    masks = get_confusion_matrix_intersection_mats(groundtruth, predicted)
    color_mask = np.zeros_like(image)

    for label, mask in masks.items():
        color = colors[label]
        mask_rgb = np.zeros_like(image)
        mask_rgb[mask != 0] = color
        color_mask += mask_rgb

    return cv2.addWeighted(image, alpha, color_mask, 1 - alpha, 0)

def evaluate_mcc(ground_truth, predicted, forged_img):



    grayscale = cv2.cvtColor(forged_img, cv2.COLOR_BGR2GRAY)

    grayscale = 255 - grayscale
    groundtruth_scaled = ground_truth // 255
    predicted_scaled = predicted // 255

    groundtruth_list = (groundtruth_scaled).flatten().tolist()
    predicted_list = (predicted_scaled).flatten().tolist()
    validation_metrics = get_validation_metrics(groundtruth_list, predicted_list)
    print(validation_metrics)

    # visualize
    alpha = 0.7
    confusion_matrix_colors = {
        "tp": (0, 255, 255),  # cyan
        "fp": (255, 0, 255),  # magenta
        "fn": (255, 255, 0),  # yellow
        "tn": (0, 0, 0)  # black
    }

    validation_mask = get_confusion_matrix_overlaid_mask(255 - grayscale, ground_truth, predicted, alpha,
                                                             confusion_matrix_colors)
    print("Cyan - TP")
    print("Magenta - FP")
    print("Yellow - FN")
    print("Black - TN")

    plt.imshow(validation_mask)
    plt.axis('off')
    plt.title("confusion matrix overlay mask")
    plt.show()

    return validation_metrics


if __name__ == "__main__":
    ground_truth = cv2.imread("/content/drive/MyDrive/MEDIASECURITY_DATASET/comofod_small/001_B.png",0)
    predicted = cv2.imread("/content/drive/MyDrive/binary_output_masks/001_F_BC1 mantra_mask.png",0)
    forged_img = cv2.imread("/content/drive/MyDrive/MEDIASECURITY_DATASET/comofod_small/001_F_BC1_JP2K_Q95.jp2")

    grayscale = cv2.cvtColor(forged_img, cv2.COLOR_BGR2GRAY)

    grayscale = 255 - grayscale
    groundtruth_scaled = ground_truth // 255
    predicted_scaled = predicted // 255

    groundtruth_list = (groundtruth_scaled).flatten().tolist()
    predicted_list = (predicted_scaled).flatten().tolist()
    validation_metrics = get_validation_metrics(groundtruth_list, predicted_list)
    print(validation_metrics)

    # visualize
    alpha = 0.7
    confusion_matrix_colors = {
        "tp": (0, 255, 255),  # cyan
        "fp": (255, 0, 255),  # magenta
        "fn": (255, 255, 0),  # yellow
        "tn": (0, 0, 0)  # black
    }

    validation_mask = get_confusion_matrix_overlaid_mask(255 - grayscale, ground_truth, predicted, alpha,
                                                     confusion_matrix_colors)
    print("Cyan - TP")
    print("Magenta - FP")
    print("Yellow - FN")
    print("Black - TN")

    plt.imshow(validation_mask)
    plt.axis('off')
    plt.title("confusion matrix overlay mask")
    plt.show()