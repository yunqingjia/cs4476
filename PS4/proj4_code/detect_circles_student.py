import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple
from scipy import misc, ndimage
from skimage import feature
from skimage.color import rgb2gray
from matplotlib.patches import Circle


def showCircles(
    img: np.ndarray,
    circles: np.ndarray,
    houghAccumulator: np.ndarray,
    showCenter: bool = False,
) -> None:
    """
    Function to plot the identified circles
    and associated centers in the input image.

    Args:
        - img: Input RGB image with shape H x W x 3 and dtype "uint8"
        - circles: An N x 3 numpy array containing the (x, y, radius)
            parameters associated with the detected circles
        - houghAccumulator: Accumulator array of size H x W
        - showCenter: Flag specifying whether to visualize the center
            or not
    """
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)

    ax1.set_aspect("equal")
    ax1.imshow(img)

    if (len(houghAccumulator.shape)==3):
        # print(houghAccumulator.shape[2])
        # for r in range(houghAccumulator.shape[2]):
        #     plt.figure()
        #     plt.imshow(houghAccumulator[:, :, r])
        pass
    else:
        ax2.imshow(houghAccumulator)

    for circle in circles:
        x, y, rad = circle
        circ = Circle((y, x), rad, color="black", fill=False, linewidth=1.5)
        ax1.add_patch(circ)
        if showCenter:
            ax1.scatter(y, x, color="black")
    plt.show()


def detectCircles(
    img: np.ndarray, radius: int, threshold: float, useGradient: bool = False
) -> Tuple[np.ndarray, np.ndarray]:

    """
    Implement a hough transform based circle detector that takes as input an
    image, a fixed radius, voting threshold and returns the centers of any detected
    circles of about that size and the hough space used for finding centers.

    NOTE: You are not allowed to use any existing hough transform detector function and
        are expected to implement the circle detection algorithm from scratch. As helper
        functions, you may use 
            - skimage.color.rgb2gray (for RGB to Grayscale conversion)
            - skimage.feature.canny (for edge detection)
            - denoising functions (if required)
        Additionally, you can use the showCircles function defined above to visualize
        the detected circles and the accumulator array.

    NOTE: You may have to tune the "sigma" parameter associated with your edge detector 
        to be able to detect the circles. For debugging, considering visualizing the
        intermediate outputs of your edge detector as well.

    For debugging, you can use im1.jpg to verify your implementation. See if you are able
    to detect circles of radii [75, 90, 100, 150]. Note that your implementation
    will be evaluated on a different image. For the sake of simplicity, you can assume
    that the test image will have the same basic color scheme as the provided image. Any
    hyper-parameters you tune for im1.jpg should also be applicable for the test image.

    Args:
        - img: Input RGB image with shape H x W x 3 and dtype "uint8"
        - radius: Radius of circle to be detected
        - threshold: Post-processing threshold to determine circle parameters
            from the accumulator array
        - useGradient: Flag that allows the user to optionally exploit the
            gradient direction measured at edge points.

    Returns:
        - circles: An N x 3 numpy array containing the (x, y, radius)
            parameters associated with the detected circles
        - houghAccumulator: Accumulator array of size H x W

    """
    ######################################################################################
    ## TODO: YOUR CODE GOES HERE                                                        ##
    ######################################################################################

    imgGray = rgb2gray(img)
    edgeMap = feature.canny(imgGray)
    # plt.figure()
    # plt.imshow(edgeMap)
    (x, y) = np.where(edgeMap)
    (h, w) = imgGray.shape
    houghAccumulator = np.zeros((h, w))

    if useGradient:
        # if using the gradient, use the gradient direction as theta
        dfdx = ndimage.sobel(imgGray, axis=0)
        dfdy = ndimage.sobel(imgGray, axis=1)
        theta = np.arctan2(dfdy, dfdx)

        for i in range(len(x)):
            a = np.rint(x[i] - radius*np.cos(theta[x[i], y[i]])).astype(int)
            b = np.rint(y[i] - radius*np.sin(theta[x[i], y[i]])).astype(int)

            # check to see if the center is inside the image
            if ((a < h) & (b < w)):
                houghAccumulator[a, b] += 1

    else:
        # if not using the gradient, iterate from 0-360 degrees 
        step_size = 12
        theta = np.arange(360+step_size, step=step_size)/180*np.pi
        
        for i in range(len(x)):
            for theta_ in theta:
                a = np.rint(x[i] + radius*np.cos(theta_)).astype(int)
                b = np.rint(y[i] + radius*np.sin(theta_)).astype(int)

                if ((a < h) & (b < w)):
                    houghAccumulator[a, b] += 1

    max_votes = np.max(houghAccumulator)
    (xc, yc) = np.where(houghAccumulator >= (threshold*max_votes))
    circles = np.vstack((np.vstack((xc, yc)), radius*np.ones(len(xc)))).T

    return circles, houghAccumulator

    ######################################################################################
    ## YOUR CODE ENDS HERE                                                              ##
    ######################################################################################

def detectCirclesUnknownRadii(
    img: np.ndarray,
    threshold: float,
    sigma: float = 1,
    useGradient: bool = False,
):

    imgGray = rgb2gray(img)
    edgeMap = feature.canny(imgGray, sigma)
    plt.figure()
    plt.imshow(edgeMap)
    (x, y) = np.where(edgeMap)
    (h, w) = imgGray.shape
    # define range of radius
    r_start, r_stop, r_step = 10, 100, 10
    radii = np.arange(r_start, r_stop+r_step, r_step)
    houghAccumulator = np.zeros((h, w, len(radii)))

    if useGradient:
        # if using the gradient, use the gradient direction as theta
        dfdx = ndimage.sobel(imgGray, axis=0)
        dfdy = ndimage.sobel(imgGray, axis=1)
        theta = np.arctan2(dfdy, dfdx)

        for i in range(len(x)):
            for radius in radii:
                a = np.rint(x[i] + radius*np.cos(theta[x[i], y[i]])).astype(int)
                b = np.rint(y[i] + radius*np.sin(theta[x[i], y[i]])).astype(int)

                # check to see if the center is inside the image
                if ((a < h) & (b < w) & ((radius//r_step) < len(radii))):
                    houghAccumulator[a, b, radius//r_step] += 1

    else:
        # if not using the gradient, iterate from 0-360 degrees 
        step_size = 12
        theta = np.arange(360+step_size, step=step_size)/180*np.pi

        for i in range(len(x)):
            for radius in radii:
                for theta_ in theta:
                    a = np.rint(x[i] + radius*np.cos(theta_)).astype(int)
                    b = np.rint(y[i] + radius*np.sin(theta_)).astype(int)

                # check to see if the center is inside the image
                if ((a < h) & (b < w) & ((radius//r_step) < len(radii))):
                    houghAccumulator[a, b, radius//r_step] += 1


    max_votes = np.max(houghAccumulator)
    
    (xc, yc, rc) = np.where(houghAccumulator >= (threshold*max_votes))
    circles = np.vstack((np.vstack((xc, yc)), rc*r_step)).T

    return circles, houghAccumulator