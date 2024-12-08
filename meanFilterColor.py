import numpy as np
import cv2
from matplotlib import pyplot as plt

def meanFilterColored(I):
    # Check if the input is a colored image (RGB)
    if len(I.shape) == 3 and I.shape[2] == 3:
        # Separate the image into RGB channels
        R, G, B = cv2.split(I)
        
        # Create the arithmetic mean filter kernel (3x3 window)
        meanFilterKernel = np.ones((3, 3), np.float32) / 9
        
        # Apply the mean filter to each channel
        R_filtered = cv2.filter2D(R, -1, meanFilterKernel)
        G_filtered = cv2.filter2D(G, -1, meanFilterKernel)
        B_filtered = cv2.filter2D(B, -1, meanFilterKernel)
        
        # Combine the filtered channels back into a colored image
        meanFilteredImg = cv2.merge((R_filtered, G_filtered, B_filtered)).astype(np.uint8)
    else:
        # If the image is grayscale, just apply the mean filter
        meanFilterKernel = np.ones((3, 3), np.float32) / 9
        meanFilteredImg = cv2.filter2D(I, -1, meanFilterKernel).astype(np.uint8)
    
    # Display the original and mean-filtered images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(I, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(meanFilteredImg, cv2.COLOR_BGR2RGB))
    plt.title('Mean Filtered Image')
    plt.axis('off')
    
    plt.show()
    
    return meanFilteredImg

if __name__ == "__main__":
    # Load your image
    image_path = '/Users/Lorna/Pepper.tif'  # Change this to your image path
    image = cv2.imread(image_path)
    
    # Apply mean filter
    filtered_image = meanFilterColored(image)