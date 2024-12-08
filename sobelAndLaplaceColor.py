import numpy as np
import cv2
import matplotlib.pyplot as plt

def sobelAndLaplaceColor(input_image, thresholdValue=100, process_mode='RGB'):
    # Read the image
    I = cv2.imread(input_image)

    # Check if the image is grayscale
    if len(I.shape) == 2 or I.shape[2] == 1:
        I = cv2.cvtColor(I, cv2.COLOR_GRAY2BGR)
    
    # Convert to double for calculations
    input_image = I.astype(np.float64)

    # Sobel Operator Masks
    Mx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # Horizontal mask
    My = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])  # Vertical mask

    # Laplace Operator Mask
    laplace_kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])

    # Convert to grayscale for luminosity processing
    if process_mode == 'LUM':
        I_gray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
        I = np.stack([I_gray] * 3, axis=-1)  # Treat grayscale as 3 identical channels

    # Pre-allocate matrices for Sobel and Laplace results
    sobel_image = np.zeros_like(I, dtype=np.uint8)
    laplace_image = np.zeros_like(I, dtype=np.uint8)

    # Edge Detection Process for Each Channel
    for channel in range(3):
        # Extract the channel
        channel_data = input_image[:, :, channel]

        # Sobel Filtering
        Gx = cv2.filter2D(channel_data, -1, Mx)
        Gy = cv2.filter2D(channel_data, -1, My)
        sobel_filtered = np.sqrt(Gx**2 + Gy**2)

        # Laplace Filtering
        laplace_filtered = cv2.filter2D(channel_data, -1, laplace_kernel)

        # Normalize results
        sobel_image[:, :, channel] = cv2.convertScaleAbs(sobel_filtered)
        laplace_image[:, :, channel] = cv2.convertScaleAbs(laplace_filtered)

    # Apply threshold for Laplacian
    laplace_image[laplace_image < thresholdValue] = 0

    # Display results
    plt.figure(figsize=(12, 8))
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(I, cv2.COLOR_BGR2RGB))
    plt.title('Input Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(sobel_image, cv2.COLOR_BGR2RGB))
    plt.title('Sobel Edge Detection')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(laplace_image, cv2.COLOR_BGR2RGB))
    plt.title('Laplace Edge Detection')
    plt.axis('off')
    
    plt.show()
    
    return sobel_image, laplace_image

if __name__ == "__main__":
    # Path to your image
    image_path = '/Users/Lorna/Pepper.tif'  # Change this to your image path
    threshold = 100  # Threshold for Laplacian
    process_mode = 'RGB'  # Options: 'RGB' or 'LUM'

    # Apply Sobel and Laplacian edge detection
    sobel_image, laplace_image = sobelAndLaplaceColor(image_path, threshold, process_mode)
