from flask import Flask, render_template, request, jsonify, send_file
from PIL import Image
import numpy as np
import io
import cv2
import base64
import random
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte

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

    # Load the image using PIL
    pil_image = Image.open(image_bytes)
    image_np = np.array(pil_image)  # Convert PIL to NumPy array
    
    # Convert NumPy array to OpenCV format (BGR)
    if image_np.ndim == 2:  # Grayscale image
        image_cv = image_np
    else:
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Apply noise or other OpenCV-based processing
    if noiseModeling == 'gaussian':
        c = random.uniform(0.2, 0.8)
        noisy_image_cv = gaussian_noise(image_cv, c=c)
    else:
        c = random.uniform(0.1, 0.2)
        noisy_image_cv = impulse_noise(image_cv, corruption_rate=c)

    # Convert processed OpenCV image back to RGB (for PIL compatibility)
    noisy_image_rgb = cv2.cvtColor(noisy_image_cv, cv2.COLOR_BGR2RGB)
    # Convert back to PIL for final processing
    noisy_image_pil = Image.fromarray(noisy_image_rgb)

    # Encode the image back to base64
    buffered = io.BytesIO()
    noisy_image_pil.save(buffered, format="JPEG")
    noisy_image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return jsonify({
        "noisy_image": f"data:image/jpeg;base64,{noisy_image_base64}",
        "c_value": c
        }), 200

def gaussian_noise(image, c):
    image = image.astype(np.float64)
    noise = np.random.randn(*image.shape) * c * np.std(image, axis=(0, 1), keepdims=True)
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy_image

def impulse_noise(image, corruption_rate):
    if len(image.shape) == 2:  # Grayscale image
        rows, cols = image.shape
        channels = 1  # Treat grayscale as having one channel
    elif len(image.shape) == 3:  # Color image
        rows, cols, channels = image.shape
    else:
        raise ValueError("Input image must be either grayscale or color (RGB).")

    total_pixels = rows * cols

    # Calculate the number of corrupted pixels
    num_corrupted_pixels = round(corruption_rate * total_pixels)
    
    # Generate random indices for the pixels to be corrupted
    corrupted_indices = np.random.choice(total_pixels, num_corrupted_pixels, replace=False)
    
    # Create a copy of the image to add noise
    corrupted_img = np.copy(image)
    
    # Generate random noise values
    random_values = np.random.randint(0, 256, (num_corrupted_pixels, channels))
    
    # Flatten the image and add noise
    img_linear = corrupted_img.reshape(-1, channels)
    img_linear[corrupted_indices] = random_values
    
    # Reshape the image back to its original dimensions
    corrupted_img = img_linear.reshape((rows, cols, channels)) if channels > 1 else img_linear.reshape((rows, cols))

    return corrupted_img

if __name__ == '__main__':
    app.run(debug=True)
