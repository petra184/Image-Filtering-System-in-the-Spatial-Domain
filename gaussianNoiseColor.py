import numpy as np

def gaussian_noise(I, noise_coeff, mode="RGB"):
 
    I = I.astype(np.float64)  # Convert to double precision for calculations

    if mode == "luminosity":
        # Convert to luminosity channel if RGB
        if I.ndim == 3:
            I = 0.2989 * I[:, :, 0] + 0.5870 * I[:, :, 1] + 0.1140 * I[:, :, 2]

        # Calculate standard deviation of the channel
        std_dev = np.std(I)

        # Generate Gaussian noise
        noise = noise_coeff * std_dev * np.random.randn(*I.shape)

        # Add noise and clip to [0, 255]
        noisy_image = I + noise
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

        return noisy_image

    elif mode == "RGB" and I.ndim == 3:
        # Separate RGB channels
        R, G, B = I[:, :, 0], I[:, :, 1], I[:, :, 2]

        # Calculate standard deviation for each channel
        R_std, G_std, B_std = np.std(R), np.std(G), np.std(B)

        # Generate Gaussian noise for each channel
        R_noise = noise_coeff * R_std * np.random.randn(*R.shape)
        G_noise = noise_coeff * G_std * np.random.randn(*G.shape)
        B_noise = noise_coeff * B_std * np.random.randn(*B.shape)

        # Add noise and clip values to [0, 255]
        R_noisy = np.clip(R + R_noise, 0, 255).astype(np.uint8)
        G_noisy = np.clip(G + G_noise, 0, 255).astype(np.uint8)
        B_noisy = np.clip(B + B_noise, 0, 255).astype(np.uint8)

        # Combine channels back into a noisy image
        noisy_image = np.stack([R_noisy, G_noisy, B_noisy], axis=2)

        return noisy_image

    else:
        raise ValueError("Invalid mode or dimensions of the input image.")

# Example Usage
if __name__ == "__main__":
    # Test image (3D array for RGB)
    I = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

    # Add Gaussian noise to RGB channels
    noisy_rgb = gaussian_noise(I, noise_coeff=0.1, mode="RGB")
    print("Noisy RGB Image Generated.")

    # Add Gaussian noise to luminosity channel
    noisy_luminosity = gaussian_noise(I, noise_coeff=0.1, mode="luminosity")
    print("Noisy Luminosity Image Generated.")
