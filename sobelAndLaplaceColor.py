import numpy as np
import cv2
import matplotlib.pyplot as plt

def sobelAndLaplaceColor(input_image, thresholdValue=100):
    # Read and validate the input image
    I = cv2.imread(input_image)
    
    # Check if the image is colored (RGB)
    if I.shape[2] != 3:
        raise ValueError('Input must be a colored image (RGB).')
    
    # Convert to double for calculations
    input_image = I.astype(np.float64)

    # Sobel Operator Masks
    Mx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # Horizontal mask
    My = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])  # Vertical mask

    # Laplace Operator Mask
    laplace_kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])

    # Pre-allocate matrices for Sobel and Laplace results
    sobel_image = np.zeros_like(I, dtype=np.uint8)
    laplace_image = np.zeros_like(I, dtype=np.uint8)

    # Edge Detection Process for Each Channel
    for channel in range(3):
        # Extract the channel
        channel_data = input_image[:, :, channel]

        # Get image dimensions
        rows, cols = channel_data.shape

        # Pre-allocate the filtered images
        sobel_filtered = np.zeros((rows, cols))
        laplace_filtered = np.zeros((rows, cols))

        # Sobel Filtering
        for i in range(1, rows-1):
            for j in range(1, cols-1):
                # Extract the 3x3 region
                region = channel_data[i-1:i+2, j-1:j+2]

                # Gradient approximations
                Gx = np.sum(Mx * region)
                Gy = np.sum(My * region)

                # Calculate the magnitude
                sobel_filtered[i, j] = np.sqrt(Gx**2 + Gy**2)

        # Laplace Filtering
        for i in range(1, rows-1):
            for j in range(1, cols-1):
                # Extract the 3x3 region
                region = channel_data[i-1:i+2, j-1:j+2]

                # Laplacian approximation
                laplace_filtered[i, j] = np.sum(laplace_kernel * region)

        # Normalize results
        sobel_image[:, :, channel] = cv2.convertScaleAbs(sobel_filtered)
        laplace_image[:, :, channel] = cv2.convertScaleAbs(laplace_filtered)

    # Apply threshold for Laplacian if given
    laplace_image[laplace_image < thresholdValue] = 0

    # Convert logical image back to uint8
    laplace_image = laplace_image.astype(np.uint8)

    # Display results
    plt.figure()
    plt.imshow(cv2.cvtColor(I, cv2.COLOR_BGR2RGB))
    plt.title('Input Image')
    plt.axis('off')
    
    plt.figure()
    plt.imshow(cv2.cvtColor(sobel_image, cv2.COLOR_BGR2RGB))
    plt.title('Sobel Edge Detection')
    plt.axis('off')
    
    plt.figure()
    plt.imshow(cv2.cvtColor(laplace_image, cv2.COLOR_BGR2RGB))
    plt.title('Laplace Edge Detection')
    plt.axis('off')
    
    plt.show()
    
    return sobel_image, laplace_image

if __name__ == "__main__":
    # Path to your image
    image_path = '/Users/Lorna/Pepper.tif'  # Change this to your image path
    threshold = 100  # Threshold for Laplacian
    
    # Apply Sobel and Laplacian edge detection
    sobel_image, laplace_image = sobelAndLaplaceColor(image_path, threshold)
