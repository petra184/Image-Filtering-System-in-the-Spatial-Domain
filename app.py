from flask import Flask, render_template, request, jsonify, send_file
from PIL import Image
import numpy as np
import io
import cv2
import base64
import random
import matplotlib.pyplot as plt

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/results')
def results():
    return render_template('results.html')

@app.route('/image_processing', methods=['POST'])
def image_processing():
    data = request.get_json()
    if 'image' not in data:
        return jsonify({"error": "No image data provided"}), 400

    processingChannel = data.get('processingChannel', None)
    noiseModeling = data.get('noiseModeling', 'gaussian')
    filter_to_use = data.get('filter', None)

    image_data = data['image']
    if ',' in image_data:  # Ensure the base64 string has a prefix to remove
        image_data = image_data.split(",")[1]

    try:
        image_bytes = io.BytesIO(base64.b64decode(image_data))
    except Exception as e:
        return jsonify({"error": f"Base64 decoding failed: {str(e)}"}), 400

    pil_image = Image.open(image_bytes)
    image_np = np.array(pil_image)  # Convert PIL to NumPy array
    
    if image_np.ndim == 2:  # Grayscale image
        image_cv = image_np
    else:
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    if noiseModeling == 'Gaussian':
        c = random.uniform(0.2, 0.8)
        noisy_image_cv = gaussian_noise(image_cv, c, mode=processingChannel)
    else:
        c = random.uniform(0.1, 0.2)
        noisy_image_cv = impulse_noise(image_cv, corruption_rate=c, mode = processingChannel)
    
    noisy_image_rgb = cv2.cvtColor(noisy_image_cv, cv2.COLOR_BGR2RGB)
    noisy_image_pil = Image.fromarray(noisy_image_rgb)
    buffered = io.BytesIO()
    noisy_image_pil.save(buffered, format="JPEG")
    noisy_image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    
    if filter_to_use == "Mean Filter":
        filtered_image = mean_filter(image_cv, processingChannel)
    elif filter_to_use == "Smart Filter":
        filtered_image = smart_filter(image_cv, processingChannel)
    elif filter_to_use == "Unsharp Masking":
        filtered_image = unsharp_masking(image_cv, k = 1.5, mode = processingChannel)
    elif filter_to_use == "Sobel Edge Detector":
        filtered_image = sobel_edge(image_cv, process_mode=processingChannel)
    else:
        filtered_image = laplace_edge(image_cv, threshold_value=10, process_mode=processingChannel)
        
    filtered_image_rgb = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB)
    filtered_image_pil = Image.fromarray(filtered_image_rgb)
    buffered2 = io.BytesIO()
    filtered_image_pil.save(buffered2, format="JPEG")
    filtered_image_base64 = base64.b64encode(buffered2.getvalue()).decode('utf-8')
    
    info = RMSEandPSNR(image_cv, filtered_image, processingChannel)
    stats_initial = image_stats(image_cv, mode="RGB")
    stats_noisy = image_stats(noisy_image_cv, mode="RGB")
    stats_filtered = image_stats(filtered_image, mode="RGB")
       
    return jsonify({
        "noisy_image": f"data:image/jpeg;base64,{noisy_image_base64}",
        "c_value": c,
        "filtered_image": f"data:image/jpeg;base64,{filtered_image_base64}",
        "rmse": info["RMSE"],
        "psnr": info["PSNR"],
        "stats_initial": stats_initial,
        "stats_noisy": stats_noisy,
        "stats_filtered": stats_filtered
        }), 200

#FUNCTIONS
def impulse_noise(image, corruption_rate, mode):
    corrupted_image = np.copy(image)

    if mode == "Luminosity":
        if corrupted_image.ndim == 3:
            corrupted_image = 0.2989 * corrupted_image[:, :, 0] + \
                              0.5870 * corrupted_image[:, :, 1] + \
                              0.1140 * corrupted_image[:, :, 2]

        rows, cols = corrupted_image.shape
        num_pixels_to_corrupt = round(corruption_rate * rows * cols)

        random_row_indices = np.random.randint(0, rows, size=num_pixels_to_corrupt)
        random_col_indices = np.random.randint(0, cols, size=num_pixels_to_corrupt)

        for i in range(num_pixels_to_corrupt):
            corrupted_image[random_row_indices[i], random_col_indices[i]] = np.random.randint(0, 256)

        return np.clip(corrupted_image, 0, 255).astype(np.uint8)

    elif mode == "RGB" and corrupted_image.ndim == 3:
        R, G, B = corrupted_image[:, :, 0], corrupted_image[:, :, 1], corrupted_image[:, :, 2]

        rows, cols = R.shape
        num_pixels_to_corrupt = round(corruption_rate * rows * cols)

        random_row_indices = np.random.randint(0, rows, size=num_pixels_to_corrupt)
        random_col_indices = np.random.randint(0, cols, size=num_pixels_to_corrupt)

        for i in range(num_pixels_to_corrupt):
            R[random_row_indices[i], random_col_indices[i]] = np.random.randint(0, 256)
            G[random_row_indices[i], random_col_indices[i]] = np.random.randint(0, 256)
            B[random_row_indices[i], random_col_indices[i]] = np.random.randint(0, 256)

        corrupted_image = np.stack([R, G, B], axis=2)

        return np.clip(corrupted_image, 0, 255).astype(np.uint8)

    else:
        raise ValueError("Invalid mode or dimensions of the input image.")

def gaussian_noise(I, noise_coeff, mode):
    I = I.astype(np.float64)  # Convert to double precision for calculations

    if mode == "Luminosity":
        if I.ndim == 3:
            I = 0.2989 * I[:, :, 0] + 0.5870 * I[:, :, 1] + 0.1140 * I[:, :, 2]

        std_dev = np.std(I)
        noise = noise_coeff * std_dev * np.random.randn(*I.shape)
        noisy_image = I + noise
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        return noisy_image

    elif mode == "RGB" and I.ndim == 3:
        R, G, B = I[:, :, 0], I[:, :, 1], I[:, :, 2]
        R_std, G_std, B_std = np.std(R), np.std(G), np.std(B)

        R_noise = noise_coeff * R_std * np.random.randn(*R.shape)
        G_noise = noise_coeff * G_std * np.random.randn(*G.shape)
        B_noise = noise_coeff * B_std * np.random.randn(*B.shape)

        R_noisy = np.clip(R + R_noise, 0, 255).astype(np.uint8)
        G_noisy = np.clip(G + G_noise, 0, 255).astype(np.uint8)
        B_noisy = np.clip(B + B_noise, 0, 255).astype(np.uint8)

        noisy_image = np.stack([R_noisy, G_noisy, B_noisy], axis=2)
        return noisy_image

    else:
        raise ValueError("Invalid mode or dimensions of the input image.")
    

def mean_filter(I, mode):
    if len(I.shape) == 3 and I.shape[2] == 3: 
        if mode == "RGB":
            R, G, B = cv2.split(I)
            mean_filter_kernel = np.ones((3, 3), np.float32) / 9
            
            R_filtered = cv2.filter2D(R, -1, mean_filter_kernel)
            G_filtered = cv2.filter2D(G, -1, mean_filter_kernel)
            B_filtered = cv2.filter2D(B, -1, mean_filter_kernel)
            mean_filtered_img = cv2.merge((R_filtered, G_filtered, B_filtered)).astype(np.uint8)
        
        elif mode == "Luminosity":
            luminosity = 0.2989 * I[:, :, 2] + 0.5870 * I[:, :, 1] + 0.1140 * I[:, :, 0]
            
            mean_filter_kernel = np.ones((3, 3), np.float32) / 9
            luminosity_filtered = cv2.filter2D(luminosity.astype(np.float32), -1, mean_filter_kernel)
            
            mean_filtered_img = np.clip(luminosity_filtered, 0, 255).astype(np.uint8)
        
        else:
            raise ValueError("Invalid mode for colored image. Choose 'RGB' or 'luminosity'.")
    
    elif len(I.shape) == 2 or (len(I.shape) == 3 and I.shape[2] == 1):
        mean_filter_kernel = np.ones((3, 3), np.float32) / 9
        mean_filtered_img = cv2.filter2D(I, -1, mean_filter_kernel).astype(np.uint8)
    
    else:
        raise ValueError("Unsupported image format. Input should be RGB or grayscale.")
    
    return mean_filtered_img

def smart_filter(I, mode):
    if len(I.shape) == 3 and I.shape[2] == 3:  # Check if the image is colored (RGB)
        if mode == "RGB":
            R, G, B = cv2.split(I)

            # Apply bilateral filtering to each channel
            R_filtered = cv2.bilateralFilter(R, d=9, sigmaColor=75, sigmaSpace=75)
            G_filtered = cv2.bilateralFilter(G, d=9, sigmaColor=75, sigmaSpace=75)
            B_filtered = cv2.bilateralFilter(B, d=9, sigmaColor=75, sigmaSpace=75)

            # Merge the filtered channels back into an image
            smart_filtered_img = cv2.merge((R_filtered, G_filtered, B_filtered)).astype(np.uint8)

        elif mode == "Luminosity":
            # Convert to luminosity grayscale
            luminosity = 0.2989 * I[:, :, 2] + 0.5870 * I[:, :, 1] + 0.1140 * I[:, :, 0]

            # Apply bilateral filter to the luminosity
            luminosity_filtered = cv2.bilateralFilter(luminosity.astype(np.float32), d=9, sigmaColor=75, sigmaSpace=75)

            # Clip values and convert back to uint8
            smart_filtered_img = np.clip(luminosity_filtered, 0, 255).astype(np.uint8)

        else:
            raise ValueError("Invalid mode for colored image. Choose 'RGB' or 'luminosity'.")

    elif len(I.shape) == 2 or (len(I.shape) == 3 and I.shape[2] == 1):  # Grayscale image
        # Apply bilateral filter directly to the grayscale image
        smart_filtered_img = cv2.bilateralFilter(I, d=9, sigmaColor=75, sigmaSpace=75).astype(np.uint8)

    else:
        raise ValueError("Unsupported image format. Input should be RGB or grayscale.")

    return smart_filtered_img

import numpy as np
import cv2

def unsharp_masking(I, k, mode):
    if I is None:
        raise ValueError("Input image is None. Provide a valid image for processing.")

    if len(I.shape) == 2:  # Grayscale image
        image_type = "grayscale"
    elif len(I.shape) == 3 and I.shape[2] == 3:  # RGB image
        image_type = "RGB"
    else:
        raise ValueError("Unsupported image format. Provide a grayscale or RGB image.")

    if mode not in ["RGB", "Luminosity", "grayscale"]:
        raise ValueError("Invalid mode. Choose from 'RGB', 'Luminosity', or 'grayscale'.")

    input_image = I.astype(np.float64)
    mean_filter = np.ones((5, 5), np.float32) / 25  # Increased kernel size

    if mode == "grayscale":
        if image_type != "grayscale":
            raise ValueError("Input image is not grayscale. Use 'Luminosity' or 'RGB' mode for color images.")

        blurred_image = cv2.filter2D(input_image, -1, mean_filter)
        mask = input_image - blurred_image
        sharpened_image = np.clip(input_image + k * mask, 0, 255)

    elif mode == "RGB":
        if image_type != "RGB":
            raise ValueError("Input image is not RGB. Use 'grayscale' mode for grayscale images.")

        sharpened_image = np.zeros_like(input_image)
        for channel in range(3):
            channel_data = input_image[:, :, channel]
            blurred_image = cv2.filter2D(channel_data, -1, mean_filter)
            mask = channel_data - blurred_image
            sharpened_channel = channel_data + k * mask
            sharpened_image[:, :, channel] = np.clip(sharpened_channel, 0, 255)
        sharpened_image = sharpened_image.astype(np.uint8)

    elif mode == "Luminosity":
        if image_type != "RGB":
            raise ValueError("Luminosity mode requires an RGB image.")

        # Convert RGB to YUV and extract the Y (luma) channel
        yuv_image = cv2.cvtColor(I, cv2.COLOR_RGB2YUV)
        luma_channel = yuv_image[:, :, 0].astype(np.float64)

        # Apply unsharp masking to the luma channel
        blurred_image = cv2.filter2D(luma_channel, -1, mean_filter)
        mask = luma_channel - blurred_image
        sharpened_luma = luma_channel + k * mask
        sharpened_image = np.clip(sharpened_luma, 0, 255).astype(np.uint8)

    return sharpened_image.astype(np.uint8)


import cv2
import numpy as np

def sobel_edge(image, process_mode='RGB'):
    if image is None:
        raise ValueError("Input image is None. Provide a valid image.")
    
    if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image
    else:
        grayscale_image = image

    # Define Sobel filters
    Mx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # Horizontal mask
    My = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])  # Vertical mask

    if process_mode == 'Luminosity':
        # Convert to grayscale for luminosity processing
        input_image = cv2.cvtColor(grayscale_image, cv2.COLOR_BGR2GRAY)
        Gx = cv2.filter2D(input_image, cv2.CV_64F, Mx)
        Gy = cv2.filter2D(input_image, cv2.CV_64F, My)
        sobel_magnitude = np.sqrt(Gx**2 + Gy**2)
        processed_image = cv2.convertScaleAbs(sobel_magnitude)
    elif process_mode == 'RGB':
        input_image = grayscale_image.astype(np.float64)
        processed_image = np.zeros_like(input_image, dtype=np.uint8)

        # Process each channel independently
        for channel in range(3):
            channel_data = input_image[:, :, channel]
            Gx = cv2.filter2D(channel_data, cv2.CV_64F, Mx)
            Gy = cv2.filter2D(channel_data, cv2.CV_64F, My)
            sobel_magnitude = np.sqrt(Gx**2 + Gy**2)
            processed_image[:, :, channel] = cv2.convertScaleAbs(sobel_magnitude)
    else:
        raise ValueError("Invalid process_mode. Use 'RGB' or 'Luminosity'.")

    return processed_image


def laplace_edge(image, threshold_value=50, process_mode='RGB'):
    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if process_mode == 'Luminosity':
        input_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        processed_image = np.zeros_like(input_image, dtype=np.uint8)
    else:
        input_image = image.astype(np.float64)
        processed_image = np.zeros_like(image, dtype=np.uint8)

    laplace_kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])

    if process_mode == 'Luminosity':
        laplace_filtered = cv2.filter2D(input_image, -1, laplace_kernel)
        laplace_result = cv2.convertScaleAbs(laplace_filtered)
        laplace_result[laplace_result < threshold_value] = 0
        processed_image = laplace_result
    else:
        for channel in range(3):
            channel_data = input_image[:, :, channel]
            laplace_filtered = cv2.filter2D(channel_data, -1, laplace_kernel)
            laplace_result = cv2.convertScaleAbs(laplace_filtered)
            laplace_result[laplace_result < threshold_value] = 0
            processed_image[:, :, channel] = laplace_result

    return processed_image


def compute_mse(image1, image2):
    diff = image1 - image2
    mse = np.mean(diff ** 2)
    return float(mse) 

def compute_psnr(image1, image2, data_range=255):
    mse = compute_mse(image1, image2)
    if mse == 0:  
        return float('inf')
    psnr_value = 10 * np.log10((data_range ** 2) / mse)
    return float(psnr_value)


def RMSEandPSNR(I, I2, mode):
    I = I.astype(np.float64)
    I2 = I2.astype(np.float64)

    # Resize if dimensions do not match
    if I.shape != I2.shape:
        I2 = cv2.resize(I2, (I.shape[1], I.shape[0]))

    if mode == "Luminosity":
        if I.ndim == 3 and I.shape[2] == 3:
            # Convert to grayscale using luminosity method
            I = 0.2989 * I[:, :, 0] + 0.5870 * I[:, :, 1] + 0.1140 * I[:, :, 2]
        if I2.ndim == 3 and I2.shape[2] == 3:
            I2 = 0.2989 * I2[:, :, 0] + 0.5870 * I2[:, :, 1] + 0.1140 * I2[:, :, 2]

        # Compute MSE, RMSE, and PSNR for grayscale images
        mse = compute_mse(I, I2)
        rmse = np.sqrt(mse)
        peaksnr = compute_psnr(I, I2, data_range=255)

        return {
            "RMSE": {"Combined": round(rmse, 2), "Y": round(rmse, 2)},  # Only "Y" and "Combined"
            "PSNR": {"Combined": round(peaksnr, 2), "Y": round(peaksnr, 2)}  # Only "Y" and "Combined"
        }

    elif mode == "RGB" and I.ndim == 3:
        if I.shape[2] != 3 or I2.shape[2] != 3:
            raise ValueError("For RGB mode, images must have 3 channels.")

        N = np.prod(I.shape[:2])
        rmse_channels = {}
        psnr_channels = {}

        for i, channel in enumerate(['R', 'G', 'B']):
            mse_channel = compute_mse(I[:, :, i], I2[:, :, i])
            rmse_channels[channel] = round(np.sqrt(mse_channel), 2)
            psnr_channels[channel] = round(compute_psnr(I[:, :, i], I2[:, :, i], data_range=255), 2)

        # Calculate the Y (luminance) channel
        Y1 = 0.2989 * I[:, :, 0] + 0.5870 * I[:, :, 1] + 0.1140 * I[:, :, 2]
        Y2 = 0.2989 * I2[:, :, 0] + 0.5870 * I2[:, :, 1] + 0.1140 * I2[:, :, 2]

        mse_Y = compute_mse(Y1, Y2)
        rmse_Y = round(np.sqrt(mse_Y), 2)
        psnr_Y = round(compute_psnr(Y1, Y2, data_range=255), 2)

        # Combined metrics for the RGB channels
        diff_combined = I - I2
        mse_combined = np.sum(diff_combined ** 2) / (3 * N)
        rmse_combined = round(np.sqrt(mse_combined), 2)
        peaksnr_combined = round(compute_psnr(I, I2, data_range=255), 2)

        return {
            "RMSE": {
                "R": rmse_channels['R'],
                "G": rmse_channels['G'],
                "B": rmse_channels['B'],
                "Y": rmse_Y,
                "Combined": rmse_combined
            },
            "PSNR": {
                "R": psnr_channels['R'],
                "G": psnr_channels['G'],
                "B": psnr_channels['B'],
                "Y": psnr_Y,
                "Combined": peaksnr_combined
            }
        }
    else:
        raise ValueError("Invalid mode or dimensions of input images. Ensure mode matches image format.")

def image_stats(I, mode="RGB"):
    I = I.astype(np.float64)  # Convert image to double precision for accurate computation

    def round_stats(stats):
        """Helper function to round and format stats for better readability."""
        if isinstance(stats, dict):
            return {key: round(float(value), 2) for key, value in stats.items()}
        return {key: round(float(value), 2) for key, value in stats.items()}

    if mode == "Luminosity":
        # Convert RGB to grayscale using luminosity formula
        if I.ndim == 3:
            I = 0.2989 * I[:, :, 0] + 0.5870 * I[:, :, 1] + 0.1140 * I[:, :, 2]

        # Compute statistics for the grayscale/luminosity image
        min_val = np.min(I)
        max_val = np.max(I)
        mean_val = np.mean(I)
        std_dev = np.std(I)
        variance_val = np.var(I)
        snr = mean_val / std_dev if std_dev > 0 else np.inf

        return round_stats({
            "Min": min_val,
            "Max": max_val,
            "Mean": mean_val,
            "Standard Deviation": std_dev,
            "Variance": variance_val,
            "SNR": snr,
        })

    elif mode == "RGB" and I.ndim == 3:
        stats = {"R": {}, "G": {}, "B": {}}
        for i, channel in enumerate(["R", "G", "B"]):
            channel_data = I[:, :, i]

            # Compute statistics for each channel
            stats[channel] = {
                "Min": np.min(channel_data),
                "Max": np.max(channel_data),
                "Mean": np.mean(channel_data),
                "Standard Deviation": np.std(channel_data),
                "Variance": np.var(channel_data),
                "SNR": np.mean(channel_data) / np.std(channel_data) if np.std(channel_data) > 0 else np.inf,
            }

        return {channel: round_stats(channel_stats) for channel, channel_stats in stats.items()}

    elif I.ndim == 2:  # Handle grayscale images directly
        min_val = np.min(I)
        max_val = np.max(I)
        mean_val = np.mean(I)
        std_dev = np.std(I)
        variance_val = np.var(I)
        snr = mean_val / std_dev if std_dev > 0 else np.inf

        return round_stats({
            "Min": min_val,
            "Max": max_val,
            "Mean": mean_val,
            "Standard Deviation": std_dev,
            "Variance": variance_val,
            "SNR": snr,
        })

    else:
        raise ValueError("Invalid mode or dimensions of the input image.")



if __name__ == '__main__':
    app.run(debug=True)
