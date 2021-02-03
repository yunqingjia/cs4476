import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import io

class Prob3():
    def __init__(self):
        """Load input color image inputPS1Q3.jpg here and assign it as a class variable"""
        ###### START CODE HERE ######

        self.A = io.imread('inputPS1Q3.jpg')

        ###### END CODE HERE ######
        pass
    
    def rgb2gray(self, rgb):
        """
        Do RGB to Gray image conversion here. Input is the RGB image and you must return the grayscale image as gray

        Returns:
            gray: grayscale image (1 channeled image with integer values lying between 0 - 255)
        """
        
        ###### START CODE HERE ######

        gray = np.dot(rgb.astype(np.float), [0.2989, 0.5870, 0.1140]).astype(np.uint8)

        ###### END CODE HERE ######
        pass
    
        ###### return gray ######
        return gray

    def prob_3_1(self):
        """
        Swap red and green color channels here, and return swapImg

        Returns:
            swapImg: RGB image with R and G channels swapped (3 channeled image with integer values lying between 0 - 255)
        """
        
        ###### START CODE HERE ######

        swapImg = np.zeros((self.A.shape))
        swapImg[:, :, 0], swapImg[:, :, 1], swapImg[:, :, 2] = self.A[:, :, 1], self.A[:, :, 0], self.A[:, :, 2]
        swapImg = swapImg.astype(np.uint8)
        plt.imshow(swapImg)
        plt.savefig('outputPS1Q3_1.png')
        plt.show()

        ###### END CODE HERE ######
        pass
    
        ###### return swapImg ######
        return swapImg
    
    def prob_3_2(self):
        """
        This function would simply call your rgb2gray function and return the grayscale image.

        Returns:
            grayImg: grayscale image (1 channeled image with integer values lying between 0 - 255)
        """
        
        ###### START CODE HERE ######

        grayImg = self.rgb2gray(self.A)
        plt.imshow(grayImg, cmap='gray')
        plt.savefig('outputPS1Q3_2.png')
        plt.show()

        ###### END CODE HERE ######
        pass
    
        ###### return grayImg ######
        return grayImg
    
    def prob_3_3(self):
        """
        Convert grayscale image to its negative.

        Returns:
            negativeImg: negative image (1 channeled image with integer values lying between 0 - 255)
        """
        
        ###### START CODE HERE ######

        negativeImg = 255 - self.rgb2gray(self.A)
        plt.imshow(negativeImg, cmap='gray')
        plt.savefig('outputPS1Q3_3.png')
        plt.show()

        ###### END CODE HERE ######
        pass
    
        ###### return negativeImg ######
        return negativeImg
    
    def prob_3_4(self):
        """
        Create mirror image of gray scale image here.
        
        Returns:
            mirrorImg: mirror image (1 channeled image with integer values lying between 0 - 255)
        """
        
        ###### START CODE HERE ######

        mirrorImg = self.rgb2gray(self.A)[:, ::-1]
        plt.imshow(mirrorImg, cmap='gray')
        plt.savefig('outputPS1Q3_4.png')
        plt.show()

        ###### END CODE HERE ######
        pass
    
        ###### return mirrorImg ######
        return mirrorImg
    
    def prob_3_5(self):
        """
        Average grayscale image with mirror image here.
        
        Returns:
            avgImg: average of grayscale and mirror image (1 channeled image with integer values lying between 0 - 255)
        """
        
        ###### START CODE HERE ######

        grayImg = self.rgb2gray(self.A).astype(np.float)
        mirrorImg = self.prob_3_4().astype(np.float)
        avgImg = ((grayImg + mirrorImg) / 2).astype(np.uint8)
        plt.imshow(avgImg, cmap='gray')
        plt.savefig('outputPS1Q3_5.png')
        plt.show()

        ###### END CODE HERE ######
        pass
    
        ###### return avgImg ######
        return avgImg
    
    def prob_3_6(self):
        """
        Create noise matrix N and save as noise.npy. Add N to grayscale image, clip to ensure that max value is 255.
        
        Returns:
            addNoiseImg: grayscale image after adding noise (1 channeled image with integer values lying between 0 - 255)
        """
        
        ###### START CODE HERE ######

        noise = np.load('noise.npy')
        grayImg = self.rgb2gray(self.A).astype(np.float)
        addNoiseImg = noise + grayImg
        addNoiseImg[addNoiseImg > 255] = 255
        plt.imshow(addNoiseImg, cmap='gray')
        plt.savefig('outputPS1Q3_6.png')
        plt.show()

        ###### END CODE HERE ######
        pass
    
        ###### return addNoiseImg ######
        return addNoiseImg
        
if __name__ == '__main__': 
    
    p3 = Prob3()
    
    swapImg = p3.prob_3_1()
    grayImg = p3.prob_3_2()
    negativeImg = p3.prob_3_3()
    mirrorImg = p3.prob_3_4()
    avgImg = p3.prob_3_5()
    addNoiseImg = p3.prob_3_6()