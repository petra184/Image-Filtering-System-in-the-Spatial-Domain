import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as immse

def RMSEandPSNR(I, I2, mode="RGB"):
   
    # Convert images to double precision
    I = I.astype(np.float64)
    I2 = I2.astype(np.float64)

    # Luminosity channel mode
    if mode == "luminosity":
        if I.ndim == 3:  # Convert RGB to grayscale (luminosity approximation)
            I = 0.2989 * I[:, :, 0] + 0.5870 * I[:, :, 1] + 0.1140 * I[:, :, 2]
            I2 = 0.2989 * I2[:, :, 0] + 0.5870 * I2[:, :, 1] + 0.1140 * I2[:, :, 2]

        # Calculate RMSE and PSNR
        mse = immse(I, I2)
        rmse = np.sqrt(mse)
        peaksnr = psnr(I, I2, data_range=255)
        return {"RMSE": rmse, "PSNR": peaksnr}

    # RGB channel mode
    elif mode == "RGB" and I.ndim == 3:
        N = np.prod(I.shape[:2])  # Total number of pixels per channel
        rmse_channels = {}
        psnr_channels = {}

        # Calculate RMSE and PSNR for each channel
        for i, channel in enumerate(['R', 'G', 'B']):
            diff = I[:, :, i] - I2[:, :, i]
            mse_channel = immse(I[:, :, i], I2[:, :, i])
            rmse_channels[channel] = np.sqrt(mse_channel)
            psnr_channels[channel] = psnr(I[:, :, i], I2[:, :, i], data_range=255)

        # Combined RMSE across all channels
        diff_combined = I - I2
        mse_combined = np.sum(diff_combined**2) / (3 * N)
        rmse_combined = np.sqrt(mse_combined)
        peaksnr_combined = psnr(I, I2, data_range=255)

        return {
            "RMSE": {"R": rmse_channels['R'], "G": rmse_channels['G'], "B": rmse_channels['B'], "Combined": rmse_combined},
            "PSNR": {"R": psnr_channels['R'], "G": psnr_channels['G'], "B": psnr_channels['B'], "Combined": peaksnr_combined}
        }

    else:
        raise ValueError("Invalid mode or dimensions of input images. Ensure mode matches image format.")
