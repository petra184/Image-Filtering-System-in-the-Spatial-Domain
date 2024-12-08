function [sharpened_image] = unsharpMaskingColor(input_image, k)
    % Read the image
    I = imread(input_image);
    
    % Check if the image is colored (RGB)
    if size(I, 3) ~= 3
        error('Input must be a colored image (RGB).');
    end

    % Convert the image to double for calculations
    input_image = double(I);

    % Pre-allocate the sharpened image
    sharpened_image = zeros(size(input_image));

    % Define the 3x3 mean filter
    mean_filter = (1/9) * ones(3, 3);

    % Process each color channel independently
    for channel = 1:3
        % Extract the channel
        channel_data = input_image(:, :, channel);

        % Apply mean filter using convolution to handle boundary effects
        blurred_image = conv2(channel_data, mean_filter, 'same');

        % Calculate the mask
        mask = channel_data - blurred_image;

        % Enhance the image using the sharpening factor k
        sharpened_channel = channel_data + k * mask;

        % Clip values to [0, 255] range
        sharpened_channel = max(0, min(255, sharpened_channel));

        % Store the result in the corresponding channel
        sharpened_image(:, :, channel) = sharpened_channel;
    end

    % Convert back to uint8 for display
    sharpened_image = uint8(sharpened_image);

    % Display the original and sharpened images
    figure, imshow(uint8(I)), title('Original Image');
    figure, imshow(sharpened_image), title(['Sharpened Image (k = ', num2str(k), ')']);
end
