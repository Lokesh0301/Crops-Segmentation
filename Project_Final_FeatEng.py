''' Segmentation of Crops final project code by Group 10 (CSCE 5222) '''


'''Below are the packages that are used for our algorithm it also requires Python 3.9 version'''

import cv2
import numpy as np
from scipy import ndimage
from skimage import measure, color, io
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from PIL import Image, ImageEnhance
from skimage.filters import gabor
from pywt import dwt2

# Confusion Matrix function
def CFM(image, reference):
    ref1, ref2, ref3 = cv2.split(reference)

    FP = len(np.where(image - ref1 == 1)[0])
    FN = len(np.where(image - ref1 == -1)[0])
    TP = len(np.where(image + ref1 == 2)[0])
    TN = len(np.where(image + ref1 == 0)[0])

    cmat = [[TP, FN], [FP, TN]]

    accuracy = ((TP+TN) / (FP+TP+TN+FN))*100
    precision = (TP / (TP + FP))*100

    return precision,accuracy


'''Gabor Filering involves the following four functions 'get_image_energy','get_energy_density', 'get_magnitude' and 'applygabor' which calculates the texture of our forest. Here we are using image energy to decrease the bandwidth which reduces the influence of more textured from less textured areas '''


def get_image_energy(pixels):
    _, (cH, cV, cD) = dwt2(pixels.T, 'db1')
    energy = (cH ** 2 + cV ** 2 + cD ** 2).sum() / pixels.size
    return energy
def get_energy_density(pixels):
    energy = get_image_energy(pixels)
    energy_density = energy / (pixels.shape[0]*pixels.shape[1])
    return round(energy_density*100,5)
def get_magnitude(response):
    magnitude = np.array([np.sqrt(response[0][i][j]**2+response[1][i][j]**2)
                        for i in range(len(response[0])) for j in range(len(response[0][i]))])
    return magnitude


def applygabor(image1,theta,freq):
    image = Image.open(image1).convert('RGB')
    image_size = image.size
    converter = ImageEnhance.Color(image)
    image = converter.enhance(0.5)
    
    # Convert to grayscale
    image = image.convert('L')
    image = np.asanyarray(image)
    pixels = np.asarray(image, dtype="uint64")
    
    energy_density = get_energy_density(pixels)
    
    # Get fixed bandwidth using energy density
    bandwidth = abs(0.4*energy_density - 0.5)
    filt_real, filt_imag = gabor(image, frequency=freq, bandwidth=bandwidth, theta=theta)
    # get magnitude response
    magnitude = get_magnitude([filt_real, filt_imag])    
    im = Image.fromarray(magnitude.reshape(image_size)).convert('L')

    return im
     


#Function 'applyseg' is the algorithm to the find the segmented crops which takes two arguments original image and its path

def applyseg(img,path_img):
    # Splitting image into B, G, R channels
    B, G, R = cv2.split(img)


    # Convolution kernel
    convolution_kernel = np.array([[1, 0, 1],
                                   [0, 0, 0],
                                   [1, 0, 1]])
    KC = cv2.filter2D(G, -1, convolution_kernel)
    KC = 255 - KC

    # Converting image to HSV
    HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Splitting image into H, S,V channels
    H, S, V = cv2.split(HSV)

    # Weighted sum of different channels
    cells = (0.67 * S) + (0.33 * V) - (0.4 * KC)

    #Normalising the image
    cells_N = (cells / cells.max()) * 255

    #Converting image to 8 - bit unsigned integer
    cells = cells_N.astype(np.uint8)
    
    
    
    # Gabor Apply
    theta=0.785
    freq=1.6
    res_gabor=applygabor(path_img,theta,freq)
    fimg = np.asarray(res_gabor)
    _, thresh_fimg = cv2.threshold(fimg, 200, 255, cv2.THRESH_BINARY )

   
    # Thresholding the weighted sum
    ret1, thresh = cv2.threshold(cells, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    #
    kernel = np.ones((3, 3), np.uint8)
    #
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    #
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    #
    sure_bg = cv2.dilate(opening, kernel, iterations=10)
    #
    sure_bg_1 = cv2.erode(opening, kernel, iterations=10)

    # Applying distance transform
    dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 5)

    # getting local maxima
    local_max = peak_local_max(dist_transform, indices=False, min_distance=20, labels=thresh)
    #
    markers = ndimage.label(local_max, structure=np.ones((3, 3)))[0]

    # Applying Watershed Algorithm
    markers = watershed(dist_transform, markers, mask=thresh).astype((np.float64))
    img[markers == -1] = [0, 255, 255]

    # Converting image to RGB
    img = color.label2rgb(markers, bg_label=0)
    # Converting image to gray
    img2_gray = color.rgb2gray(img)
    img2_gray = cv2.normalize(img2_gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    _, img_bin = cv2.threshold(img2_gray, 0, 1, type=cv2.THRESH_BINARY)

    #
    res_img=np.zeros([2048,2048])
    for i in range(0,2048,1):
        for j in range(0,2048,1):
            if img_bin[i,j]==1 and thresh_fimg[i,j]==255:
               res_img[i,j]=0
            else:
               res_img[i,j]=img_bin[i,j] 
    return img_bin,res_img

for i in range(1,9,1):
    # Path for source image
    ''' We have renamed the files for our convenience crop1_org representing the original image and crop1_ref.tiff representing reference image
     crop1-field     
     crop2-L88a
     crop3-L88b
     crop4-L96b
     crop5-W107a
     crop6-W107b
     crop7-L96a
     crop8-L97a
     crop9-L97b 

     vcrop1-L88c
     vcrop2-L88d
     vcrop3-L96c
     vcrop4-L96d
     vcrop5-L97c
     vcrop6-L97d
     vcrop7-W107c
     vcrop8-W107d
     '''
    path_img = r"D:\UNT\Feature Engineering\Segmentation Project\Dataset\\"+"vcrop"+str(i)+".jpg"
    # Path for reference image
    path_ref = r"D:\UNT\Feature Engineering\Segmentation Project\Dataset\\"+"vcrop"+str(i)+"_ref.tiff"
    # Reading source image into workspace
    img = cv2.imread(path_img)
    # Reading reference image into workspace
    ref = cv2.imread(path_ref)
    # Calling Segmentation function which returns the segmented image from watershed segmentation and gabor filtered output (seg_res)
    _, seg_res = applyseg(img, path_img)
    # Calling Confusion Matrix function for evaluating the peformance of our algorithm
    prec, acc = CFM(seg_res, ref)
    print('Accuracy and precision of '+str(i)+' image are '+str(acc)+' '+str(prec))

