import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_corners(img):
    im = cv2.imread(img)

    if im is None:
        print("error")
        return
    
    image = np.copy(im)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray_image)

    harris_corners = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

    image[harris_corners > 0.01*harris_corners.max()] = [0,0,255]

    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    plt.title("img original")
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("cantos detectados")
    plt.axis('off')

    plt.show()  

if __name__ == "__main__":
    detect_corners('brasil.jpg')