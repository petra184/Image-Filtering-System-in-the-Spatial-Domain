import cv2
import numpy as np
import matplotlib.pyplot as plt

def meanAndSpatialFilter(I):
    if len(I.shape) == 3:  # Check if the image is colored
        # Separate the color channels
        R, G, B = cv2.split(I)
    else:  # If the image is grayscale
        R, G, B = I, I, I  # Duplicate for compatibility

    # Create the arithmetic mean filter kernel (3x3 window)
    meanFilterKernel = np.ones((3, 3), np.float32) / 9

    # Apply the arithmetic mean filter to each channel
    meanFilteredR = cv2.filter2D(R, -1, meanFilterKernel)
    meanFilteredG = cv2.filter2D(G, -1, meanFilterKernel)
    meanFilteredB = cv2.filter2D(B, -1, meanFilterKernel)
    meanFilteredImg = cv2.merge((meanFilteredR, meanFilteredG, meanFilteredB)).astype(np.uint8)

    # Create the spatial filter kernel (3x3 window)
    spatialFilterKernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.float32) / 16

    # Apply the spatial filter to each channel
    spatialFilteredR = cv2.filter2D(R, -1, spatialFilterKernel)
    spatialFilteredG = cv2.filter2D(G, -1, spatialFilterKernel)
    spatialFilteredB = cv2.filter2D(B, -1, spatialFilterKernel)
    spatialFilteredImg = cv2.merge((spatialFilteredR, spatialFilteredG, spatialFilteredB)).astype(np.uint8)

    return meanFilteredImg, spatialFilteredImg

def gaussianNoise(image, noiseCoeff):
    noise = np.random.normal(0, noiseCoeff * np.std(image.astype(float)), image.shape)
    noisy_image = cv2.add(image.astype(float), noise)
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image

def RMSEandPSNR(clean_img, noisy_img):
    mse = np.mean((clean_img.astype(float) - noisy_img.astype(float)) ** 2)
    rmse = np.sqrt(mse)
    psnr = 20 * np.log10(255.0 / rmse) if rmse > 0 else float('inf')
    return rmse, psnr

# Function for grayscale or luminosity
def processLuminosity(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Example usage
if __name__ == "__main__":
    image_path = 'Pepper.tif'  # Update this to your image path
    colorImage = cv2.imread(image_path)

    # Handle grayscale separately
    if len(colorImage.shape) == 2:  # Grayscale
        grayImage = colorImage
    else:  # Convert to luminosity
        grayImage = processLuminosity(colorImage)

    # Add Gaussian noise
    noiseCoeff = 0.2
    noisyImage = gaussianNoise(grayImage, noiseCoeff)

    # Apply filters
    meanFilteredImg, spatialFilteredImg = meanAndSpatialFilter(colorImage)

    # Display results
    plt.figure(figsize=(12, 6))
    plt.subplot(131), plt.imshow(cv2.cvtColor(colorImage, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
    plt.subplot(132), plt.imshow(noisyImage, cmap='gray'), plt.title('Noisy Image')
    plt.subplot(133), plt.imshow(meanFilteredImg), plt.title('Mean Filtered Image')
    plt.show()

    # Calculate RMSE and PSNR
    rmse, psnr = RMSEandPSNR(colorImage, meanFilteredImg)
    print(f"RMSE: {rmse}, PSNR: {psnr}")
