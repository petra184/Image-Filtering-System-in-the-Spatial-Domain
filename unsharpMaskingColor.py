import numpy as np
import cv2
from matplotlib import pyplot as plt

def unsharp_masking(input_image, k, mode="RGB"):
 
    # Read the image
    I = cv2.imread(input_image, cv2.IMREAD_UNCHANGED)

    if I is None:
        raise ValueError(f"Could not read the image at {input_image}. Check the path.")

    # Handle grayscale images
    if len(I.shape) == 2 or (len(I.shape) == 3 and I.shape[2] == 1):  # Grayscale
        mode = "grayscale"

    if mode not in ["RGB", "luminosity", "grayscale"]:
        raise ValueError("Invalid mode. Choose from 'RGB', 'luminosity', 'grayscale'.")

    # Convert to float64 for calculations
    input_image = I.astype(np.float64)

    # Define the 3x3 mean filter
    mean_filter = np.ones((3, 3), np.float32) / 9

    # Initialize sharpened image
    if mode == "RGB":
        sharpened_image = np.zeros_like(input_image)
        for channel in range(3):
            # Extract channel
            channel_data = input_image[:, :, channel]
            # Apply mean filter
            blurred_image = cv2.filter2D(channel_data, -1, mean_filter)
            # Calculate mask
            mask = channel_data - blurred_image
            # Enhance the image
            sharpened_channel = channel_data + k * mask
            # Clip to [0, 255]
            sharpened_image[:, :, channel] = np.clip(sharpened_channel, 0, 255)
    elif mode == "luminosity":
        # Convert to luminosity
        luminosity = 0.2989 * input_image[:, :, 2] + 0.5870 * input_image[:, :, 1] + 0.1140 * input_image[:, :, 0]
        # Apply mean filter
        blurred_image = cv2.filter2D(luminosity, -1, mean_filter)
        # Calculate mask
        mask = luminosity - blurred_image
        # Enhance luminosity
        sharpened_luminosity = luminosity + k * mask
        sharpened_image = np.clip(sharpened_luminosity, 0, 255).astype(np.uint8)
    elif mode == "grayscale":
        # Apply mean filter
        blurred_image = cv2.filter2D(input_image, -1, mean_filter)
        # Calculate mask
        mask = input_image - blurred_image
        # Enhance image
        sharpened_image = np.clip(input_image + k * mask, 0, 255).astype(np.uint8)
    else:
        raise ValueError(f"Unexpected mode: {mode}")

    # Convert back to uint8 for display
    sharpened_image = sharpened_image.astype(np.uint8)

    # Display original and sharpened images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    if mode in ["RGB", "luminosity"] and len(I.shape) == 3:
        plt.imshow(cv2.cvtColor(I, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(I, cmap="gray")
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    if mode in ["RGB", "luminosity"] and len(sharpened_image.shape) == 3:
        plt.imshow(cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(sharpened_image, cmap="gray")
    plt.title(f"Sharpened Image (k = {k}, mode = {mode})")
    plt.axis("off")

    plt.show()

    return sharpened_image

if __name__ == "__main__":
    # Path to your image
    image_path = '/Users/Lorna/Pepper.tif'  # Update this with the correct path
    k = 1.5  # Sharpening factor
    mode = "RGB"  # Options: "RGB", "luminosity", "grayscale"

    # Apply unsharp masking
    sharpened_image = unsharp_masking(image_path, k, mode=mode)
