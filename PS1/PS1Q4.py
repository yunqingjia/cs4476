import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import io

class Prob4():
    def __init__(self):
        """Load input color image indoor.png and outdoor.png here as class variables."""
        ###### START CODE HERE ######

        self.imin = io.imread('indoor.png')
        self.imout = io.imread('outdoor.png')

        ###### END CODE HERE ######
        pass
    
    def prob_4_1(self):
        """Plot R,G,B channels separately and also their corresponding LAB space channels separately"""
        
        ###### START CODE HERE ######

        plt.imsave('in_R.png', self.imin[:, :, 0], cmap='gray')
        plt.imsave('in_G.png', self.imin[:, :, 1], cmap='gray')
        plt.imsave('in_B.png', self.imin[:, :, 2], cmap='gray')
        plt.imsave('out_R.png', self.imout[:, :, 0], cmap='gray')
        plt.imsave('out_G.png', self.imout[:, :, 1], cmap='gray')
        plt.imsave('out_B.png', self.imout[:, :, 2], cmap='gray')
        in_lab = cv2.cvtColor(self.imin, cv2.COLOR_RGB2Lab)
        out_lab = cv2.cvtColor(self.imout, cv2.COLOR_RGB2Lab)
        plt.imsave('in_L.png', in_lab[:, :, 0], cmap='gray')
        plt.imsave('in_A.png', in_lab[:, :, 1], cmap='gray')
        plt.imsave('in_B.png', in_lab[:, :, 2], cmap='gray')
        plt.imsave('out_L.png', out_lab[:, :, 0], cmap='gray')
        plt.imsave('out_A.png', out_lab[:, :, 1], cmap='gray')
        plt.imsave('out_B.png', out_lab[:, :, 2], cmap='gray')

        ###### END CODE HERE ######
        pass

    def prob_4_2(self):
        """
        Convert the loaded RGB image to HSV and return HSV matrix without using inbuilt functions. Return the HSV image as HSV. Make sure to use a 3 channeled RGB image with floating point values lying between 0 - 1 for the conversion to HSV.

        Returns:
            HSV image (3 channeled image with floating point values lying between 0 - 1 in each channel)
        """
        
        img = io.imread('inputPS1Q4.jpg') 
        img = img / 255.0
        
        ###### START CODE HERE ######

        V = img.max(2)
        m = img.min(2)
        C = V - m
        S = np.copy(C)
        S[V!=0.0] = C[V!=0.0] / V[V!=0.0]
        S[V==0.0] = 0.0
        H = np.copy(S)

        for x in range(V.shape[0]):
            for y in range(V.shape[1]):
                if (C[x][y] == 0):
                    H[x][y] = 0
                elif (V[x][y] == img[x][y][0]):
                    H[x][y] = (img[x][y][1] - img[x][y][2])/C[x][y]
                elif (V[x][y] == img[x][y][1]):
                    H[x][y] = (img[x][y][2] - img[x][y][0])/C[x][y] + 2
                elif (V[x][y] == img[x][y][2]):
                    H[x][y] = (img[x][y][0] - img[x][y][1])/C[x][y] + 4

                if (H[x][y] < 0):
                    H[x][y] = H[x][y]/6 + 1
                else:
                    H[x][y] = H[x][y]/6

        HSV = np.dstack((np.dstack((H,S)),V))
        plt.imsave('outputPS1Q4.png', HSV)

        ###### END CODE HERE ######
        pass
    
        ###### return HSV ######
        return HSV

        
if __name__ == '__main__':
    
    p4 = Prob4()
    
    p4.prob_4_1()

    HSV = p4.prob_4_2()





