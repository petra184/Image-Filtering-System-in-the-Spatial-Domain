function [sobel_image, laplace_image] = sobelAndLaplaceColor(input_image, thresholdValue)
    % Read and validate the input image
    I = imread(input_image);

    % Check if the image is colored (RGB)
    if size(I, 3) ~= 3
        error('Input must be a colored image (RGB).');
    end

    % Convert to double for calculations
    input_image = double(I);

    % Sobel Operator Masks
    Mx = [-1 0 1; -2 0 2; -1 0 1]; % Horizontal mask
    My = [-1 -2 -1; 0 0 0; 1 2 1]; % Vertical mask

    % Laplace Operator Mask
    laplace_kernel = [0 -1 0; -1 4 -1; 0 -1 0];

    % Pre-allocate matrices for Sobel and Laplace results
    sobel_image = zeros(size(input_image), 'uint8');
    laplace_image = zeros(size(input_image), 'uint8');

    % Edge Detection Process for Each Channel
    for channel = 1:3
        % Extract the channel
        channel_data = input_image(:, :, channel);

        % Get image dimensions
        [rows, cols] = size(channel_data);

        % Pre-allocate the filtered images
        sobel_filtered = zeros(rows, cols);
        laplace_filtered = zeros(rows, cols);

        % Sobel Filtering
        for i = 2:rows-1
            for j = 2:cols-1
                % Extract the 3x3 region
                region = channel_data(i-1:i+1, j-1:j+1);

                % Gradient approximations
                Gx = sum(sum(Mx .* region));
                Gy = sum(sum(My .* region));

                % Calculate the magnitude
                sobel_filtered(i, j) = sqrt(Gx.^2 + Gy.^2);
            end
        end

        % Laplace Filtering
        for i = 2:rows-1
            for j = 2:cols-1
                % Extract the 3x3 region
                region = channel_data(i-1:i+1, j-1:j+1);

                % Laplacian approximation
                laplace_filtered(i, j) = sum(sum(laplace_kernel .* region));
            end
        end

        % Normalize results
        sobel_image(:, :, channel) = uint8(255 * mat2gray(sobel_filtered));
        laplace_image(:, :, channel) = uint8(255 * mat2gray(laplace_filtered));
    end

    % Apply threshold for Laplacian if given
    if nargin < 2
        thresholdValue = 100; % Default threshold
    end
    laplace_image(laplace_image < thresholdValue) = 0;

    % Convert logical image back to uint8
    laplace_image = uint8(laplace_image);

    % Display results
    figure, imshow(I); title('Input Image');
    figure, imshow(sobel_image); title('Sobel Edge Detection');
    figure, imshow(laplace_image); title('Laplace Edge Detection');
end
