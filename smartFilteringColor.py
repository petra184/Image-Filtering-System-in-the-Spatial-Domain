import cv2
import numpy as np
import matplotlib.pyplot as plt

def meanAndSpatialFilter(I):
    if len(I.shape) == 3:
        # Separate the color channels
        R, G, B = cv2.split(I)
    else:
        # If the image is grayscale
        R, G, B = I, I, I

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

    # Display the original and filtered images
    plt.figure(figsize=(15, 5))
    plt.subplot(131), plt.imshow(cv2.cvtColor(I, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
    plt.subplot(132), plt.imshow(cv2.cvtColor(meanFilteredImg, cv2.COLOR_BGR2RGB)), plt.title('Mean Filtered Image')
    plt.subplot(133), plt.imshow(cv2.cvtColor(spatialFilteredImg, cv2.COLOR_BGR2RGB)), plt.title('Spatial Filtered Image')
    plt.show()

    return meanFilteredImg, spatialFilteredImg

# Read and display an example color image
colorImage = cv2.imread('Pepper.tif')
plt.figure(), plt.imshow(cv2.cvtColor(colorImage, cv2.COLOR_BGR2RGB)), plt.title('Original Color Image')

# Measure mean and standard deviation of each color channel
R, G, B = cv2.split(colorImage)
meanR = np.mean(R)
stdR = np.std(R.astype(float))
meanG = np.mean(G)
stdG = np.std(G.astype(float))
meanB = np.mean(B)
stdB = np.std(B.astype(float))

print(f'Mean of red channel: {meanR}')
print(f'Standard deviation of red channel: {stdR}')
print(f'Mean of green channel: {meanG}')
print(f'Standard deviation of green channel: {stdG}')
print(f'Mean of blue channel: {meanB}')
print(f'Standard deviation of blue channel: {stdB}')

# Function to add Gaussian noise
def gaussianNoise(image, noiseCoeff):
    noise = np.random.normal(0, noiseCoeff * np.std(image.astype(float)), image.shape)
    noisy_image = cv2.add(image.astype(float), noise)
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image

# Add Gaussian noise with σn equal to 0.2σ and 0.3σ
noiseCoeff1 = 0.2
noiseCoeff2 = 0.3

noisyImage1 = gaussianNoise(colorImage, noiseCoeff1)
noisyImage2 = gaussianNoise(colorImage, noiseCoeff2)

# Display the original and noisy images
plt.figure(figsize=(10, 5))
plt.subplot(221), plt.imshow(cv2.cvtColor(colorImage, cv2.COLOR_BGR2RGB)), plt.title('Original Clean Image')
plt.subplot(222), plt.imshow(cv2.cvtColor(noisyImage1, cv2.COLOR_BGR2RGB)), plt.title('Noisy Image (0.2 * std)')
plt.subplot(223), plt.imshow(cv2.cvtColor(noisyImage2, cv2.COLOR_BGR2RGB)), plt.title('Noisy Image (0.3 * std)')

# Apply mean and spatial filters
meanFilteredImg1, spatialFilteredImg1 = meanAndSpatialFilter(noisyImage1)
meanFilteredImg2, spatialFilteredImg2 = meanAndSpatialFilter(noisyImage2)

# Function to calculate RMSE and PSNR
def RMSEandPSNR(clean_img, noisy_img):
    mse = np.mean((clean_img.astype(float) - noisy_img.astype(float)) ** 2)
    rmse = np.sqrt(mse)
    psnr = 20 * np.log10(255.0 / rmse)
    return rmse, psnr

# Evaluate RMSE and PSNR
(rNoisy1, psnrNoisy1) = RMSEandPSNR(colorImage, noisyImage1)
(rFilteredMean1, psnrFilteredMean1) = RMSEandPSNR(colorImage, meanFilteredImg1)
(rFilteredSmart1, psnrFilteredSmart1) = RMSEandPSNR(colorImage, spatialFilteredImg1)

(rNoisy2, psnrNoisy2) = RMSEandPSNR(colorImage, noisyImage2)
(rFilteredMean2, psnrFilteredMean2) = RMSEandPSNR(colorImage, meanFilteredImg2)
(rFilteredSmart2, psnrFilteredSmart2) = RMSEandPSNR(colorImage, spatialFilteredImg2)

# Display the RMSE and PSNR values
print('Evaluation for Noisy Image (0.2 * std):')
print(f'RMSE with clean image: {rNoisy1}')
print(f'PSNR with clean image: {psnrNoisy1}')
print(f'RMSE of mean filtered image: {rFilteredMean1}')
print(f'PSNR of mean filtered image: {psnrFilteredMean1}')
print(f'RMSE of spatial filtered image: {rFilteredSmart1}')
print(f'PSNR of spatial filtered image: {psnrFilteredSmart1}')

print('Evaluation for Noisy Image (0.3 * std):')
print(f'RMSE with clean image: {rNoisy2}')
print(f'PSNR with clean image: {psnrNoisy2}')
print(f'RMSE of mean filtered image: {rFilteredMean2}')
print(f'PSNR of mean filtered image: {psnrFilteredMean2}')
print(f'RMSE of spatial filtered image: {rFilteredSmart2}')
print(f'PSNR of spatial filtered image: {psnrFilteredSmart2}')
