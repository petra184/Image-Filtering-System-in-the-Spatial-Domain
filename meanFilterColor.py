import numpy as np
import cv2
from matplotlib import pyplot as plt

def mean_filter(I, mode="RGB"):
 
    # Check the input image type
    if len(I.shape) == 3 and I.shape[2] == 3:  # Colored image (RGB)
        if mode == "RGB":
            # Separate the image into RGB channels
            R, G, B = cv2.split(I)
            
            # Create the arithmetic mean filter kernel (3x3 window)
            mean_filter_kernel = np.ones((3, 3), np.float32) / 9
            
            # Apply the mean filter to each channel
            R_filtered = cv2.filter2D(R, -1, mean_filter_kernel)
            G_filtered = cv2.filter2D(G, -1, mean_filter_kernel)
            B_filtered = cv2.filter2D(B, -1, mean_filter_kernel)
            
            # Combine the filtered channels back into a colored image
            mean_filtered_img = cv2.merge((R_filtered, G_filtered, B_filtered)).astype(np.uint8)
        
        elif mode == "luminosity":
            # Convert the image to the luminosity channel
            luminosity = 0.2989 * I[:, :, 2] + 0.5870 * I[:, :, 1] + 0.1140 * I[:, :, 0]
            
            # Apply the mean filter to the luminosity channel
            mean_filter_kernel = np.ones((3, 3), np.float32) / 9
            luminosity_filtered = cv2.filter2D(luminosity.astype(np.float32), -1, mean_filter_kernel)
            
            # Return as a grayscale image
            mean_filtered_img = np.clip(luminosity_filtered, 0, 255).astype(np.uint8)
        
        else:
            raise ValueError("Invalid mode for colored image. Choose 'RGB' or 'luminosity'.")
    
    elif len(I.shape) == 2 or (len(I.shape) == 3 and I.shape[2] == 1):  # Grayscale image
        # Create the arithmetic mean filter kernel (3x3 window)
        mean_filter_kernel = np.ones((3, 3), np.float32) / 9
        
        # Apply the mean filter directly to the grayscale image
        mean_filtered_img = cv2.filter2D(I, -1, mean_filter_kernel).astype(np.uint8)
    
    else:
        raise ValueError("Unsupported image format. Input should be RGB or grayscale.")
    
    # Display the original and mean-filtered images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    if len(I.shape) == 3 and mode == "RGB":
        plt.imshow(cv2.cvtColor(I, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(I, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    if len(mean_filtered_img.shape) == 3:
        plt.imshow(cv2.cvtColor(mean_filtered_img, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(mean_filtered_img, cmap='gray')
    plt.title('Mean Filtered Image')
    plt.axis('off')
    
    plt.show()
    
    return mean_filtered_img

if __name__ == "__main__":
    # Load your image
    image_path = '/Users/Lorna/Pepper.tif'  # Change this to your image path
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # Load image with original format
    
    # Determine processing mode
    mode = "RGB"  # Options: "RGB", "luminosity", "grayscale"
    
    # Apply mean filter
    filtered_image = mean_filter(image, mode=mode)
