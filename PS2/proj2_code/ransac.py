import math
from typing import Tuple

import numpy as np

from proj2_code import fundamental_matrix, two_view_data
from proj2_code.least_squares_fundamental_matrix import solve_F


def calculate_num_ransac_iterations(prob_success: float, 
                                    sample_size: int, 
                                    ind_prob_correct: float) -> int:
    """
    Calculate the number of RANSAC iterations needed for a given guarantee of success.

    Args:
    -   prob_success: float representing the desired guarantee of success
    -   sample_size: int the number of samples included in each RANSAC iteration
    -   ind_prob_success: float the probability that each element in a sample is correct

    Returns:
    -   num_samples: int the number of RANSAC iterations needed

    """
    num_samples = None

    ##############################
    # TODO: Student code goes here

    num_samples = np.ceil(np.log(1-prob_success)/np.log(1-ind_prob_correct**sample_size)).astype(int)

    ##############################

    return num_samples


def find_inliers(x_0s: np.ndarray, 
                 F: np.ndarray, 
                 x_1s: np.ndarray, 
                 threshold: float) -> np.ndarray:
    """ Find the inliers' indices for a given model.

    There are multiple methods you could use for calculating the error
    to determine your inliers vs outliers at each pass. However, we suggest
    using the magnitude of the line to point distance function we wrote for the
    optimization in part 2.

    Args:
    -   x_0s: A numpy array of shape (N, 3) representing the coordinates
                   of possibly matching points from the left image
    -   F: The proposed fundamental matrix
    -   x_1s: A numpy array of shape (N, 3) representing the coordinates
                   of possibly matching points from the right image
    -   threshold: the maximum error for a point correspondence to be
                    considered an inlier
    Each row in x_1s and x_0s is a proposed correspondence (e.g. row #42 of x_0s is a point that
    corresponds to row #42 of x_1s)

    Returns:
    -    inliers: 1D array of the indices of the inliers in x_0s and x_1s

    """

    inliers = None

    ##############################
    # TODO: Student code goes here

    errors = fundamental_matrix.signed_point_line_errors(x_0s, F, x_1s)
    d1 = np.array(errors[::2])
    d2 = np.array(errors[1::2])
    epi_err = np.square(d1) + np.square(d2)
    inliers = np.where(epi_err < threshold)[0]

    ##############################

    return inliers


def ransac_fundamental_matrix(x_0s: int, 
                              x_1s: int) -> Tuple[
                                  np.ndarray, np.ndarray, np.ndarray]:
    """Find the fundamental matrix with RANSAC.

    Use RANSAC to find the best fundamental matrix by
    randomly sampling interest points. You will call your
    solve_F() from part 2 of this assignment
    and calculate_num_ransac_iterations().

    You will also need to define a new function (see above) for finding
    inliers after you have calculated F for a given sample.

    Tips:
        0. You will need to determine your P, k, and p values.
            What is an acceptable rate of success? How many points
            do you want to sample? What is your estimate of the correspondence
            accuracy in your dataset?
        1. A potentially useful function is numpy.random.choice for
            creating your random samples
        2. You will want to call your function for solving F with the random
            sample and then you will want to call your function for finding
            the inliers.
        3. You will also need to choose an error threshold to separate your
            inliers from your outliers. We suggest a threshold of 1.

    Args:
    -   x_0s: A numpy array of shape (N, 2) representing the coordinates
                   of possibly matching points from the left image
    -   x_1s: A numpy array of shape (N, 2) representing the coordinates
                   of possibly matching points from the right image
    Each row is a proposed correspondence (e.g. row #42 of x_0s is a point that
    corresponds to row #42 of x_1s)

    Returns:
    -   best_F: A numpy array of shape (3, 3) representing the best fundamental
                matrix estimation
    -   inliers_x_0: A numpy array of shape (M, 2) representing the subset of
                   corresponding points from the left image that are inliers with
                   respect to best_F
    -   inliers_x_1: A numpy array of shape (M, 2) representing the subset of
                   corresponding points from the right image that are inliers with
                   respect to best_F

    """

    best_F = None
    inliers_x_0 = None
    inliers_x_1 = None

    ##############################
    # TODO: Student code goes here

    # Defined success criteria and find the number of RANSAC iterations we need to run
    P = 0.999     # success rate
    k = 20         # sample size
    p = 0.90      # individual success probability
    threshold = 1 # error threshold
    S = calculate_num_ransac_iterations(P, k, p)
    Fs = []
    inliers_num = []

    # Preprocess the data (N, 2) -> (N, 3)
    x_0s3, x_1s3 = two_view_data.preprocess_data(x_0s, x_1s)

    # Run iterations
    for i in range(S):
        # Select k-sized random sample
        rand_idx = np.random.choice(np.arange(x_0s.shape[0]), k)

        x0i, x1i = x_0s[rand_idx], x_1s[rand_idx]
        F = solve_F(x0i, x1i)
        inliers = find_inliers(x_0s3, F, x_1s3, threshold)
        Fs.append(F)
        inliers_num.append(len(inliers))

    best_F = Fs[np.argmax(inliers_num)]
    best_inliers_idx = find_inliers(x_0s3, best_F, x_1s3, threshold)
    inliers_x_0, inliers_x_1 = x_0s[best_inliers_idx], x_1s[best_inliers_idx]

    ##############################

    return best_F, inliers_x_0, inliers_x_1
