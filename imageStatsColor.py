import numpy as np

def image_stats(I, mode="RGB"):
    """
    Calculate image statistics for RGB channels or the luminosity channel.
    
    Parameters:
        I (ndarray): Input image (grayscale or RGB).
        mode (str): Mode of operation - "RGB" for individual channel statistics or 
                    "luminosity" for grayscale approximation. Automatically handles grayscale images.
    
    Returns:
        dict: Statistics including min, max, mean, standard deviation, variance, and SNR.
    """
    I = I.astype(np.float64)  # Convert image to double precision for accurate computation

    if mode == "luminosity":
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

        return {
            "Min": min_val,
            "Max": max_val,
            "Mean": mean_val,
            "Standard Deviation": std_dev,
            "Variance": variance_val,
            "SNR": snr,
        }

    elif mode == "RGB" and I.ndim == 3:
        stats = {"R": {}, "G": {}, "B": {}}
        for i, channel in enumerate(["R", "G", "B"]):
            channel_data = I[:, :, i]

            # Compute statistics for each channel
            stats[channel]["Min"] = np.min(channel_data)
            stats[channel]["Max"] = np.max(channel_data)
            stats[channel]["Mean"] = np.mean(channel_data)
            stats[channel]["Standard Deviation"] = np.std(channel_data)
            stats[channel]["Variance"] = np.var(channel_data)
            stats[channel]["SNR"] = (
                stats[channel]["Mean"] / stats[channel]["Standard Deviation"]
                if stats[channel]["Standard Deviation"] > 0
                else np.inf
            )

        return stats

    elif I.ndim == 2:  # Handle grayscale images directly
        min_val = np.min(I)
        max_val = np.max(I)
        mean_val = np.mean(I)
        std_dev = np.std(I)
        variance_val = np.var(I)
        snr = mean_val / std_dev if std_dev > 0 else np.inf

        return {
            "Min": min_val,
            "Max": max_val,
            "Mean": mean_val,
            "Standard Deviation": std_dev,
            "Variance": variance_val,
            "SNR": snr,
        }

    else:
        raise ValueError("Invalid mode or dimensions of the input image.")

# Example usage
if __name__ == "__main__":
    # Test image (3D array for RGB)
    I = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

    # Calculate statistics for RGB channels
    stats_rgb = image_stats(I, mode="RGB")
    print("RGB Channel Statistics:")
    for channel, stats in stats_rgb.items():
        print(f"{channel}: {stats}")

    # Calculate statistics for the luminosity channel
    stats_luminosity = image_stats(I, mode="luminosity")
    print("\nLuminosity Channel Statistics:")
    print(stats_luminosity)

    # Test grayscale image
    grayscale_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    stats_grayscale = image_stats(grayscale_image)
    print("\nGrayscale Image Statistics:")
    print(stats_grayscale)
