function meanFilteredImg = meanFilterColored(I)
    % Check if the input is a colored image (RGB)
    if size(I, 3) == 3
        % Separate the image into RGB channels
        R = I(:, :, 1);
        G = I(:, :, 2);
        B = I(:, :, 3);
        
        % Create the arithmetic mean filter kernel (3x3 window)
        meanFilterKernel = (1/9) * ones(3, 3);
        
        % Apply the mean filter to each channel
        R_filtered = conv2(double(R), meanFilterKernel, 'same');
        G_filtered = conv2(double(G), meanFilterKernel, 'same');
        B_filtered = conv2(double(B), meanFilterKernel, 'same');
        
        % Combine the filtered channels back into a colored image
        meanFilteredImg = cat(3, uint8(R_filtered), uint8(G_filtered), uint8(B_filtered));
    else
        % If the image is grayscale, just apply the mean filter
        meanFilterKernel = (1/9) * ones(3, 3);
        meanFilteredImg = conv2(double(I), meanFilterKernel, 'same');
        meanFilteredImg = uint8(meanFilteredImg); % Convert to uint8 for display
    end
    
    % Display the original and mean-filtered images
    figure;
    subplot(1, 2, 1), imshow(I), title('Original Image');
    subplot(1, 2, 2), imshow(meanFilteredImg), title('Mean Filtered Image');
end
