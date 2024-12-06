# Image Filtering System in the Spatial Domain

## 📄 Short Summary of the Project
This project implements an **Image Filtering System** that allows users to upload an image, apply Gaussian noise to it, and view the original and processed (blurred) images. The system demonstrates how image filtering techniques work by combining a Python backend for computations and a JavaScript-based frontend for interactivity. Users can select processing options, such as noise modeling and filter types, and view the results on a visually appealing interface.

---

## 🚀 How to Run the Project

### Prerequisites
- Install [**Python**](https://www.python.org/downloads/) on your system.
- Have a **code editor** like [**VS Code**](https://code.visualstudio.com/) installed.
- Ensure you have the **Live Server** extension in VS Code.

---

### ⚙️ Steps to Run the Project

1. **Clone or Download the Project**
   - Save the project files (`index.html`, `results.html`, `script.js`, `calculations.py`) to a folder on your system.

2. **Set Up the Backend (Python)**
   - Open a terminal in the project folder.
   - Install the required dependencies by running:
     ```bash
     pip install flask numpy
     ```
   - Start the Python server:
     ```bash
     python calculations.py
     ```
   - The server will run at `http://127.0.0.1:5000`.

3. **Set Up the Frontend**
   - Open the project folder in **VS Code**.
   - Right-click on `index.html` and select **Open with Live Server**.
   - This will start the frontend at a URL like `http://127.0.0.1:5500`.

4. **Run the Project**
   - Open your browser and navigate to the Live Server URL for the frontend (e.g., `http://127.0.0.1:5500/index.html`).
   - Upload an image, select the desired options, and click **Apply Filter**.
   - The processed image will be displayed on the results page.

---

## 📝 Notes
- Ensure both the **Python server** and the **Live Server** are running simultaneously for the system to work.
- Supported image formats: `.jpeg` and `.png`.
- The backend dynamically processes the uploaded image and returns the result to the frontend for display.

---

### 💻 Project Structure
```plaintext
├── index.html          # Main interface for user interaction
├── results.html        # Displays the processed results
├── script.js           # Frontend logic for file handling and interactivity
├── calculations.py     # Backend logic for image processing
├── styles.css          # (Optional) Custom styling for the interface
