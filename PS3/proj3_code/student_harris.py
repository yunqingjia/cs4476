import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import maximum_filter
import pdb


def get_gaussian_kernel(ksize, sigma):
    """
    Generate a Gaussian kernel to be used in get_interest_points for calculating
    image gradients and a second moment matrix.
    You can call this function to get the 2D gaussian filter.
    
    This might be useful:
    2) Make sure the value sum to 1
    3) Some useful functions: cv2.getGaussianKernel

    Args:
    -   ksize: kernel size
    -   sigma: kernel standard deviation

    Returns:
    -   kernel: numpy nd-array of size [ksize, ksize]
    """
    
    kernel = None
    #############################################################################
    # TODO: YOUR GAUSSIAN KERNEL CODE HERE                                      #
    #############################################################################

    k_1D = cv2.getGaussianKernel(ksize, sigma)
    kernel = np.matmul(k_1D, k_1D.T)

    assert abs(np.sum(kernel)-1.0) <= 1e-6

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    
    return kernel

def my_filter2D(image, filt):
    """
    Compute a 2D convolution. Pad the border of the image using 0s.
    Any type of automatic convolution is not allowed (i.e. np.convolve, cv2.filter2D, etc.)

    Helpful functions: cv2.copyMakeBorder

    Args:
    -   image: A numpy array of shape (m,n,c),
                image may be grayscale of color (your choice)
    -   filt: filter that will be used in the convolution

    Returns:
    -   conv_image: image resulting from the convolution with the filter

    Note: It is okay to use nested for loops to implement this
    """
    conv_image = None

    #############################################################################
    # TODO: YOUR MY FILTER 2D CODE HERE                                         #
    #############################################################################

    n = np.int((filt.shape[0]-1)/2)
    pad_img = cv2.copyMakeBorder(image, n, n, n, n, cv2.BORDER_CONSTANT)
    conv_image = np.zeros(image.shape)
    filt = np.flip(filt, (0,1))

    for r in range(n, image.shape[0]+n):
        for c in range(n, image.shape[1]+n):
            subimg = pad_img[r-n:r+n+1, c-n:c+n+1]
            conv_image[r-n, c-n] = np.sum(np.multiply(filt, subimg))


    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return conv_image

def get_gradients(image):
    """
    Compute smoothed gradients Ix & Iy. This will be done using a sobel filter.
    Sobel filters can be used to approximate the image gradient
    
    Helpful functions: my_filter2D from above
    
    Args:
    -   image: A numpy array of shape (m,n) containing the image
               

    Returns:
    -   ix: numpy nd-array of shape (m,n) containing the image convolved with differentiated kernel in the x direction
    -   iy: numpy nd-array of shape (m,n) containing the image convolved with differentiated kernel in the y direction
    """
    
    ix, iy = None, None
    #############################################################################
    # TODO: YOUR IMAGE GRADIENTS CODE HERE                                      #
    #############################################################################

    # define Sobel filter matrices
    Mx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])

    My = np.array([
                   [-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]
                   ])

    ix = my_filter2D(image, Mx)
    iy = my_filter2D(image, My)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    
    return ix, iy

def remove_border_vals(image, x, y, c, window_size = 16):
    """
    Remove interest points that are too close to a border to allow SIFTfeature
    extraction. Make sure you remove all points where a window around
    that point cannot be formed.

    Args:
    -   image: image: A numpy array of shape (m,n,c),
        image may be grayscale of color (your choice)
    -   x: numpy array of shape (N,)
    -   y: numpy array of shape (N,)
    -   c: numpy array of shape (N,)
    -   window_size: int of the window size that we want to remove. (i.e. make sure all
        points in a window_size by window_size area can be formed around a point)
        Set this to 16 for unit testing. Treat the center point of this window as the bottom right
        of the center-most 4 pixels. This will be the same window used for SIFT.

    Returns:
    -   x: A numpy array of shape (N-#removed vals,) containing x-coordinates of interest points
    -   y: A numpy array of shape (N-#removed vals,) containing y-coordinates of interest points
    -   c (optional): numpy nd-array of dim (N-#removed vals,) containing the strength
    """

    #############################################################################
    # TODO: YOUR REMOVE BORDER VALS CODE HERE                                   #
    #############################################################################

    # print(len(x))
    w = np.int(np.floor(window_size/2))
    del_idx = []

    for i in range(len(x)):

        if (x[i] < w):
            del_idx.append(i)
        elif (x[i] >= (image.shape[0]-w)):
            del_idx.append(i)
        elif (y[i] < w):
            del_idx.append(i)
        elif (y[i] >= (image.shape[1]-w)):
            del_idx.append(i)

    # xnew = []
    # ynew = []
    # cnew = []

    # for i in range(len(x)-1, -1, -1):
    #     del_x = False
    #     del_y = False
    #     if (x[i] < w):

    #         continue

    x, y, c = np.delete(x, del_idx), np.delete(y, del_idx), np.delete(c, del_idx)
    # print(len(x))
    
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return x, y, c

def second_moments(ix, iy, ksize = 7, sigma = 10):
    """
    Given image gradients, ix and iy, compute sx2, sxsy, sy2 using a gaussian filter.
    (Refer to Eq 4.8 in Szeliski Sec 4.1.1 for exact equations)
    Helpful functions: my_filter2D

    Args:
    -   ix: numpy nd-array of shape (m,n) containing the gradient of the image with respect to x
    -   iy: numpy nd-array of shape (m,n) containing the gradient of the image with respect to y
    -   ksize: size of gaussian filter (set this to 7 for unit testing)
    -   sigma: deviation of gaussian filter (set this to 10 for unit testing)

    Returns:
    -   sx2: A numpy nd-array of shape (m,n) containing the result of convolving 
             a Gaussian kernel with ix*ix
    -   sy2: A numpy nd-array of shape (m,n) containing the result of convolving 
             a Gaussian kernel with iy*iy
    -   sxsy: (optional): A numpy nd-array of shape (m,n) containing the result of convolving 
             a Gaussian kernel with ix*iy
    """

    sx2, sy2, sxsy = None, None, None
    #############################################################################
    # TODO: YOUR SECOND MOMENTS CODE HERE                                       #
    #############################################################################

    kernel = get_gaussian_kernel(ksize, sigma)
    sx2 = my_filter2D(ix*ix, kernel)
    sy2 = my_filter2D(iy*iy, kernel)
    sxsy = my_filter2D(ix*iy, kernel)
    
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return sx2, sy2, sxsy

def corner_response(sx2, sy2, sxsy, alpha):

    """
    Given second moments calculate corner resposne.
    R = det(M) - alpha(trace(M)^2)
    where M = [[Sx2, SxSy],
                [SxSy, Sy2]]

    Args:
    -   sx2: A numpy nd-array of shape (m,n) containing the second moment in the x direction twice
    -   sy2: A numpy nd-array of shape (m,n) containing the second moment in the y direction twice
    -   sxsy: (optional): numpy nd-array of dim (m,n) containing the second moment in the x then the y direction
    -   alpha: empirical constant in Corner Resposne equaiton (set this to 0.05 for unit testing)

    Returns:
    -   R: Corner response score for each pixel
    """

    R = None
    #############################################################################
    # TODO: YOUR CORNER RESPONSE CODE HERE                                       #
    #############################################################################

    R = sx2*sy2 - sxsy*sxsy - alpha*np.square(sx2+sy2)
    
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return R

def non_max_suppression(R, neighborhood_size = 7):
    """
    Implement non maxima suppression. Take a matrix and return a matrix of the same size
    but only the max values in a neighborhood are non zero. We also do not want local
    maxima that are very small as well so remove all values that are below the global median.
    
    The lecture slides and textbook are a bit vague on how to do the
    non-maximum suppression once you've thresholded the cornerness score.
    You are free to experiment. For example, you could compute connected
    components and take the maximum value within each component.
    Alternatively, you could run a max() operator on each sliding window. You
    could use this to ensure that every interest point is at a local maximum
    of cornerness.

    Helpful functions: scipy.ndimage.filters.maximum_filter
    
    Args:
    -   R: numpy nd-array of shape (m, n)
    -   ksize: int that is the size of neighborhood to find local maxima (set this to 7 for unit testing)

    Returns:
    -   R_local_pts: numpy nd-array of shape (m, n) where only local maxima are non-zero 
    """

    R_local_pts = None
    
    #############################################################################
    # TODO: YOUR NON MAX SUPPRESSION CODE HERE                                  #
    #############################################################################

    # use the global median to threshold the R matrix
    # R[R < np.median(R)] = 0
    R_local_pts = np.zeros(R.shape)

    local_maxima = maximum_filter(R, neighborhood_size)
    # R_local_pts = np.copy(R)
    R[R != local_maxima] = 0
    R_local_pts = R
    R[R_local_pts < np.median(R_local_pts)] = 0

    # for i in range(R.shape[0]-neighborhood_size):
    #     for j in range(R.shape[1]-neighborhood_size):
    #         window = R[i:(i+neighborhood_size), j:(j+neighborhood_size)]
    #         ii, jj = np.unravel_index(window.argmax(), window.shape)
    #         R_local_pts[i+ii, j+jj] = np.max(window)


    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return R_local_pts
    
def get_interest_points(image, n_pts = 1500):
    """
    Implement the Harris corner detector (See Szeliski 4.1.1) to start with.

    If you're finding spurious interest point detections near the boundaries,
    it is safe to simply suppress the gradients / corners near the edges of
    the image.

    Useful in this function in order to (a) suppress boundary interest
    points (where a feature wouldn't fit entirely in the image, anyway)
    or (b) scale the image filters being used. Or you can ignore it.

    By default you do not need to make scale and orientation invariant
    local features.


    Args:
    -   image: A numpy array of shape (m,n,c),
                image may be grayscale of color (your choice)
    -   n_pts: integer of number of interest points to obtain

    Returns:
    -   x: A numpy array of shape (n_pts) containing x-coordinates of interest points
    -   y: A numpy array of shape (n_pts) containing y-coordinates of interest points
    -   R_local_pts: A numpy array of shape (m,n) containing cornerness response scores after
            non-maxima suppression and before removal of border scores
    -   confidences (optional): numpy nd-array of dim (n_pts) containing the strength
            of each interest point
    """
    x, y, R_local_pts, confidences = None, None, None, None
    

    #############################################################################
    # TODO: YOUR HARRIS CORNER DETECTOR CODE HERE                               #
    #############################################################################

    ix, iy = get_gradients(image)
    sx2, sy2, sxsy = second_moments(ix, iy, ksize=7, sigma=10)
    R = corner_response(sx2, sy2, sxsy, alpha=0.05)
    # print('raw R zeros: ' + str(R.nonzero()[0].shape))

    R_local_pts = non_max_suppression(R, neighborhood_size=7)
    # print('NMS: ' + str(R_local_pts.nonzero()[0].shape))

    x, y = R_local_pts.nonzero()[0], R_local_pts.nonzero()[1]
    confidences = R_local_pts[x, y]

    x, y, confidences = remove_border_vals(image, x, y, confidences, window_size=2)
    border = len(np.where(x>=image.shape[0]-8))
    # print(border)

    # remove border from R_local_pts matrix
    # w = np.int(16/2)
    # R_local_pts[0:w, :] = 0
    # R_local_pts[:, 0:w] = 0
    # R_local_pts[-w:, :] = 0
    # R_local_pts[:, -w:] = 0


    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    
    return x,y, R_local_pts, confidences


