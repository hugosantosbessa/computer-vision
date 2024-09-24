import cv2
import numpy as np
import matplotlib.pyplot as plt

#def herris_ransac(image_path):
def detect_corners(image_path):
    img = cv2.imread(image_path)

    if img is None:
        print("Error: Image not found")
        return
    
    # Make a copy of the image
    img_cpy= np.copy(img);
    
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(img_cpy, cv2.COLOR_BGR2GRAY)
    # Convert the image to float32
    gray = np.float32(gray_image);
    # Detect corners
    herris_corners = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.01)

    img_cpy[herris_corners > 0.01 * herris_corners.max()] = [0, 0, 255]

    # Show the image
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('cantos detectados')
    plt.axis
    plt.axis('off')

    # Show the image with the corners
    plt.subplot(122)
    plt.imshow(cv2.cvtColor(img_cpy, cv2.COLOR_BGR2RGB))
    plt.title('cantos detectados')
    plt.axis('off')

    plt.show()

if __name__ == "__main__":
    detect_corners('pontos_ransac.png')