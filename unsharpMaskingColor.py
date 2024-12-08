import numpy as np
import cv2
from matplotlib import pyplot as plt

def unsharpMaskingColor(input_image, k):
    # Read the image
    I = cv2.imread(input_image)
    
    # Check if the image is colored (RGB)
    if I.shape[2] != 3:
        raise ValueError('Input must be a colored image (RGB).')

    # Convert the image to double for calculations
    input_image = I.astype(np.float64)

    # Pre-allocate the sharpened image
    sharpened_image = np.zeros_like(input_image)

    # Define the 3x3 mean filter
    mean_filter = np.ones((3, 3), np.float32) / 9

    # Process each color channel independently
    for channel in range(3):
        # Extract the channel
        channel_data = input_image[:, :, channel]

        # Apply mean filter using convolution to handle boundary effects
        blurred_image = cv2.filter2D(channel_data, -1, mean_filter)

        # Calculate the mask
        mask = channel_data - blurred_image

        # Enhance the image using the sharpening factor k
        sharpened_channel = channel_data + k * mask

        # Clip values to [0, 255] range
        sharpened_channel = np.clip(sharpened_channel, 0, 255)

        # Store the result in the corresponding channel
        sharpened_image[:, :, channel] = sharpened_channel

    # Convert back to uint8 for display
    sharpened_image = sharpened_image.astype(np.uint8)

    # Display the original and sharpened images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(I, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2RGB))
    plt.title(f'Sharpened Image (k = {k})')
    plt.axis('off')
    
    plt.show()
    
    return sharpened_image

if __name__ == "__main__":
    # Path to your image
    image_path = '/Users/Lorna/Pepper.tif'  # Change this to your image path
    k = 1.5  # Sharpening factor, adjust as needed
    
    # Apply unsharp masking
    sharpened_image = unsharpMaskingColor(image_path, k)
