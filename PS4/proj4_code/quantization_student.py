import numpy as np
from sklearn.cluster import KMeans
from skimage.color import rgb2hsv, hsv2rgb
from typing import Tuple

import matplotlib.pyplot as plt


def quantizeRGB(origImg: np.ndarray, k: int) -> np.ndarray:
    """
    Quantize the RGB image along all 3 channels and assign values of the nearest cluster
    center to each pixel. Return the quantized image and cluster centers.

    NOTE: Use the sklearn.cluster.KMeans function for clustering and set random_state = 101
    
    Args:
        - origImg: Input RGB image with shape H x W x 3 and dtype "uint8"
        - k: Number of clusters

    Returns:
        - quantizedImg: Quantized image with shape H x W x 3 and dtype "uint8"
        - clusterCenterColors: k x 3 vector of hue centers
    """

    ######################################################################################
    ## TODO: YOUR CODE GOES HERE                                                        ##
    ######################################################################################

    h, w = origImg.shape[0], origImg.shape[1]
    flat_img = origImg.reshape(h*w, 3)
    quantizedImg = np.zeros_like(flat_img)

    km = KMeans(n_clusters=k, random_state=101)
    km.fit_predict(flat_img)
    clusterCenterColors = km.cluster_centers_
    labels = km.labels_
    for i in range(len(labels)):
        quantizedImg[i] = clusterCenterColors[labels[i]]

    quantizedImg = np.floor(quantizedImg.reshape(h, w, 3)).astype(int)

    return quantizedImg, clusterCenterColors

    ######################################################################################
    ## YOUR CODE ENDS HERE                                                              ##
    ######################################################################################


def quantizeHSV(origImg: np.ndarray, k: int) -> np.ndarray:
    """
    Convert the image to HSV, quantize the Hue channel and assign values of the nearest cluster
    center to each pixel. Return the quantized image and cluster centers.

    NOTE: Consider using skimage.color for colorspace conversion
    NOTE: Use the sklearn.cluster.KMeans function for clustering and set random_state = 101

    Args:
        - origImg: Input RGB image with shape H x W x 3 and dtype "uint8"
        - k: Number of clusters

    Returns:
        - quantizedImg: Quantized image with shape H x W x 3 and dtype "uint8"
        - clusterCenterHues: k x 1 vector of hue centers
    """
    
    ######################################################################################
    ## TODO: YOUR CODE GOES HERE                                                        ##
    ######################################################################################

    h, w = origImg.shape[0], origImg.shape[1]
    hsv_img = rgb2hsv(origImg)
    flat_img = hsv_img.reshape(h*w, 3)
    quantizedImg = np.copy(flat_img)
    hue_img = flat_img[:, 0].reshape(h*w, 1)
    # print(hue_img)

    km = KMeans(n_clusters=k, random_state=101)
    km.fit_predict(hue_img)
    clusterCenterHues = km.cluster_centers_
    labels = km.labels_

    for i in range(len(labels)):
        quantizedImg[i, 0] = clusterCenterHues[labels[i]]

    quantizedImg = np.floor(hsv2rgb(quantizedImg.reshape(h, w, 3))*255.0).astype(int)

    return quantizedImg, clusterCenterHues


    ######################################################################################
    ## YOUR CODE ENDS HERE                                                              ##
    ######################################################################################
    


def computeQuantizationError(origImg: np.ndarray, quantizedImg: np.ndarray) -> int:
    """
    Calculate the quantization error by finding the sum of squared differences between
    the original and quantized images. Implement a vectorized version (using numpy) of
    this error metric.

    Args:
        - origImg: Original input RGB image with shape H x W x 3 and dtype "uint8"
        - quantizedImg: Image obtained post-quantization with shape H x W x 3 and dtype "uint8"

    Returns
        - error: Quantization error
    """
    ######################################################################################
    ## TODO: YOUR CODE GOES HERE                                                        ##
    ######################################################################################

    quantizationError = np.sum(np.square(origImg - quantizedImg))

    ######################################################################################
    ## YOUR CODE ENDS HERE                                                              ##
    ######################################################################################
    return quantizationError
