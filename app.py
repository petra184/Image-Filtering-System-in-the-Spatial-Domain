from flask import Flask, render_template, request, jsonify, send_file
from PIL import Image
import numpy as np
import io
import cv2
import base64
from io import BytesIO
from PIL import Image
import random

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/results')
def results():
    return render_template('results.html')

@app.route('/image_processing', methods=['POST'])
def image_processing():
    # Retrieve the image data from the JSON request
    data = request.get_json()
    if 'image' not in data:
        return jsonify({"error": "No image data provided"}), 400

    # Decode the base64 image
    image_data = data['image']
    processingChannel = data['processingChannel'],
    noiseModeling = data['noiseModeling'],
    filter_to_use = data['filter']
    
    image_data = image_data.split(",")[1]  # Remove the data:image/...;base64, prefix
    image_bytes = io.BytesIO(base64.b64decode(image_data))
    image = Image.open(image_bytes)

    # Convert PIL image to NumPy array
    image_np = np.array(image)
    if (filter_to_use == 'Gaussian Noise'):
        c = random.uniform(0.2, 0.8)
        noisy_image_np = gaussian_noise(image_np, c=c)
    else:
        c = random.uniform(0.2, 0.8)
        noisy_image_np = gaussian_noise(image_np, c=c)
    

    # Convert the NumPy array back to an image
    noisy_image = Image.fromarray(noisy_image_np)

    # Encode the image back to base64
    buffered = io.BytesIO()
    noisy_image = noisy_image.convert("RGB")  # Convert RGBA to RGB
    noisy_image.save(buffered, format="JPEG")
    noisy_image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # Return the processed image as a JSON response
    return jsonify({"noisy_image": f"data:image/jpeg;base64,{noisy_image_base64}"}), 200

def gaussian_noise(image, c):
    """
    Add Gaussian noise to an image.

    Parameters:
        image (numpy.ndarray): Input image (grayscale or RGB).
        c (float): Scaling factor for the noise.

    Returns:
        numpy.ndarray: Noisy image.
    """
    image = image.astype(np.float64)

    if len(image.shape) == 3 and image.shape[2] == 3:
        noisy_image = np.zeros_like(image)
        for channel_idx in range(image.shape[2]):
            channel = image[:, :, channel_idx]
            
            channel_std = np.std(channel)
            noise_std = c * channel_std
            noise = noise_std * np.random.randn(*channel.shape)
            noisy_channel = np.clip(channel + noise, 0, 255)
            noisy_image[:, :, channel_idx] = noisy_channel
    
    else:
        image_std = np.std(image)
        noise_std = c * image_std
        noise = noise_std * np.random.randn(*image.shape)
        noisy_image = np.clip(image + noise, 0, 255)
    
    noisy_image = noisy_image.astype(np.uint8)
   
    return noisy_image

def data():
    #stavi gaussian coefficient
    #stavi za pepper noise kolka je korupcija
    return 0

if __name__ == '__main__':
    app.run(debug=True)
