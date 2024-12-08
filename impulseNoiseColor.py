import numpy as np

def impulse_noise(image, noise_type, corruption_rate, mode="RGB"):
    corrupted_image = np.copy(image)

    if mode == "luminosity":
        # Convert to luminosity channel if RGB
        if corrupted_image.ndim == 3:
            corrupted_image = 0.2989 * corrupted_image[:, :, 0] + \
                              0.5870 * corrupted_image[:, :, 1] + \
                              0.1140 * corrupted_image[:, :, 2]

        # Get dimensions
        rows, cols = corrupted_image.shape

        # Total number of pixels to corrupt
        num_pixels_to_corrupt = round(corruption_rate * rows * cols)

        # Randomly select pixel indices
        random_row_indices = np.random.randint(0, rows, size=num_pixels_to_corrupt)
        random_col_indices = np.random.randint(0, cols, size=num_pixels_to_corrupt)

        for i in range(num_pixels_to_corrupt):
            if noise_type == "random":
                # Assign a random value
                corrupted_image[random_row_indices[i], random_col_indices[i]] = np.random.randint(0, 256)
            elif noise_type == "salt-and-pepper":
                # Assign salt-and-pepper noise
                corrupted_image[random_row_indices[i], random_col_indices[i]] = 0 if np.random.rand() < 0.5 else 255
            else:
                raise ValueError("Invalid noise type. Choose either 'random' or 'salt-and-pepper'.")

        return np.clip(corrupted_image, 0, 255).astype(np.uint8)

    elif mode == "RGB" and corrupted_image.ndim == 3:
        # Separate RGB channels
        R, G, B = corrupted_image[:, :, 0], corrupted_image[:, :, 1], corrupted_image[:, :, 2]

        # Get dimensions
        rows, cols = R.shape

        # Total number of pixels to corrupt
        num_pixels_to_corrupt = round(corruption_rate * rows * cols)

        # Randomly select pixel indices
        random_row_indices = np.random.randint(0, rows, size=num_pixels_to_corrupt)
        random_col_indices = np.random.randint(0, cols, size=num_pixels_to_corrupt)

        for i in range(num_pixels_to_corrupt):
            if noise_type == "random":
                # Assign random values to each channel
                R[random_row_indices[i], random_col_indices[i]] = np.random.randint(0, 256)
                G[random_row_indices[i], random_col_indices[i]] = np.random.randint(0, 256)
                B[random_row_indices[i], random_col_indices[i]] = np.random.randint(0, 256)
            elif noise_type == "salt-and-pepper":
                # Assign salt-and-pepper noise to each channel
                R[random_row_indices[i], random_col_indices[i]] = 0 if np.random.rand() < 0.5 else 255
                G[random_row_indices[i], random_col_indices[i]] = 0 if np.random.rand() < 0.5 else 255
                B[random_row_indices[i], random_col_indices[i]] = 0 if np.random.rand() < 0.5 else 255
            else:
                raise ValueError("Invalid noise type. Choose either 'random' or 'salt-and-pepper'.")

        # Combine channels back
        corrupted_image = np.stack([R, G, B], axis=2)

        return np.clip(corrupted_image, 0, 255).astype(np.uint8)

    else:
        raise ValueError("Invalid mode or dimensions of the input image.")

# Example Usage
if __name__ == "__main__":
    # Example RGB image
    image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

    # Apply random impulse noise to RGB channels
    noisy_rgb = impulse_noise(image, noise_type="random", corruption_rate=0.1, mode="RGB")
    print("Noisy RGB Image Generated.")

    # Apply salt-and-pepper noise to luminosity channel
    noisy_luminosity = impulse_noise(image, noise_type="salt-and-pepper", corruption_rate=0.1, mode="luminosity")
    print("Noisy Luminosity Image Generated.")
