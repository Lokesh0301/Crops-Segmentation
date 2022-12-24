import cv2
import matplotlib.pyplot as plt

image = cv2.imread(r'D:\UNT\Feature Engineering\Segmentation Project\Dataset\crop9_ref.tiff')
image = image * 255
new_height = 1300

# dsize
dsize = (image.shape[1], new_height)
image = cv2.resize(image, dsize, interpolation = cv2.INTER_AREA)
cv2.imshow('image',image)

cv2.waitKey(0)
cv2.destroyAllWindows()