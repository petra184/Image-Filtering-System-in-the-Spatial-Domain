function noise_im = gaussian_noise(image, c)
    image = double(image);
    image_std = std(image(:));

    noise_std = c * image_std;
    noise = noise_std * randn(size(image));

    noise_im = image + noise;
    noise_im = max(0, min(255, noise_im));
    noise_im = uint8(noise_im);
end