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

    ######################################################################################
    ## YOUR CODE ENDS HERE                                                              ##
    ######################################################################################

    raise NotImplementedError
