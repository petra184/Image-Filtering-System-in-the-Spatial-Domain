function [meanFilteredImg, spatialFilteredImg] = meanAndSpatialFilter(I)
    % Convert the image to grayscale if it's not already
    if size(I, 3) == 3
        % Separate the color channels
        R = I(:, :, 1);
        G = I(:, :, 2);
        B = I(:, :, 3);
    else
        % If the image is grayscale, treat it as such
        R = I;
        G = I;
        B = I;
    end

    % Create the arithmetic mean filter kernel (3x3 window)
    meanFilterKernel = (1/9) * ones(3, 3);

    % Apply the arithmetic mean filter to each channel
    meanFilteredR = conv2(double(R), meanFilterKernel, 'same');
    meanFilteredG = conv2(double(G), meanFilterKernel, 'same');
    meanFilteredB = conv2(double(B), meanFilterKernel, 'same');
    meanFilteredImg = cat(3, uint8(meanFilteredR), uint8(meanFilteredG), uint8(meanFilteredB));

    % Create the spatial filter kernel (3x3 window)
    spatialFilterKernel = (1/16) * [1 2 1; 2 4 2; 1 2 1];

    % Apply the spatial filter to each channel
    spatialFilteredR = conv2(double(R), spatialFilterKernel, 'same');
    spatialFilteredG = conv2(double(G), spatialFilterKernel, 'same');
    spatialFilteredB = conv2(double(B), spatialFilterKernel, 'same');
    spatialFilteredImg = cat(3, uint8(spatialFilteredR), uint8(spatialFilteredG), uint8(spatialFilteredB));

    % Display the original and filtered images
    figure;
    subplot(1, 3, 1), imshow(I), title('Original Image');
    subplot(1, 3, 2), imshow(meanFilteredImg), title('Mean Filtered Image');
    subplot(1, 3, 3), imshow(spatialFilteredImg), title('Spatial Filtered Image');
end

% Read and display an example color image
colorImage = imread('Pepper.tif');
figure; imshow(colorImage); title('Original Color Image');

% Measure mean and standard deviation of each color channel
R = colorImage(:, :, 1);
G = colorImage(:, :, 2);
B = colorImage(:, :, 3);
meanR = mean(R(:));
stdR = std(double(R(:)));
meanG = mean(G(:));
stdG = std(double(G(:)));
meanB = mean(B(:));
stdB = std(double(B(:)));
disp(['Mean of red channel: ', num2str(meanR)]);
disp(['Standard deviation of red channel: ', num2str(stdR)]);
disp(['Mean of green channel: ', num2str(meanG)]);
disp(['Standard deviation of green channel: ', num2str(stdG)]);
disp(['Mean of blue channel: ', num2str(meanB)]);
disp(['Standard deviation of blue channel: ', num2str(stdB)]);

% Add Gaussian noise with σn equal to 0.2σ and 0.3σ 
noiseCoeff1 = 0.2;
noiseCoeff2 = 0.3;

noisyImage1 = gaussianNoise(colorImage, noiseCoeff1);
noisyImage2 = gaussianNoise(colorImage, noiseCoeff2);

% Display the original and noisy images
figure;
subplot(2, 2, 1), imshow(colorImage), title('Original Clean Image');
subplot(2, 2, 2), imshow(noisyImage1), title('Noisy Image (0.2 * std)');
subplot(2, 2, 3), imshow(noisyImage2), title('Noisy Image (0.3 * std)');

% Apply mean and spatial filters
[meanFilteredImg1, spatialFilteredImg1] = meanAndSpatialFilter(noisyImage1);
[meanFilteredImg2, spatialFilteredImg2] = meanAndSpatialFilter(noisyImage2);

% Evaluate RMSE and PSNR
% Compare clean image with noisy and filtered
[rNoisy1, rCheckNoisy1, psnrNoisy1, psnrCheckNoisy1] = RMSEandPSNR(colorImage, noisyImage1);
[rFilteredMean1, rCheckFilteredMean1, psnrFilteredMean1, psnrCheckFilteredMean1] = RMSEandPSNR(colorImage, meanFilteredImg1);
[rFilteredSmart1, rCheckFilteredSmart1, psnrFilteredSmart1, psnrCheckFilteredSmart1] = RMSEandPSNR(colorImage, spatialFilteredImg1);

[rNoisy2, rCheckNoisy2, psnrNoisy2, psnrCheckNoisy2] = RMSEandPSNR(colorImage, noisyImage2);
[rFilteredMean2, rCheckFilteredMean2, psnrFilteredMean2, psnrCheckFilteredMean2] = RMSEandPSNR(colorImage, meanFilteredImg2);
[rFilteredSmart2, rCheckFilteredSmart2, psnrFilteredSmart2, psnrCheckFilteredSmart2] = RMSEandPSNR(colorImage, spatialFilteredImg2);

% Display the RMSE and PSNR values
disp('Evaluation for Noisy Image (0.2 * std):');
disp(['RMSE with clean image: ', num2str(rNoisy1), ' (Check: ', num2str(rCheckNoisy1), ')']);
disp(['PSNR with clean image: ', num2str(psnrNoisy1), ' (Check: ', num2str(psnrCheckNoisy1), ')']);
disp(['RMSE of mean filtered image: ', num2str(rFilteredMean1), ' (Check: ', num2str(rCheckFilteredMean1), ')']);
disp(['PSNR of mean filtered image: ', num2str(psnrFilteredMean1), ' (Check: ', num2str(psnrCheckFilteredMean1), ')']);
disp(['RMSE of spatial filtered image: ', num2str(rFilteredSmart1), ' (Check: ', num2str(rCheckFilteredSmart1), ')']);
disp(['PSNR of spatial filtered image: ', num2str(psnrFilteredSmart1), ' (Check: ', num2str(psnrCheckFilteredSmart1), ')']);

disp('Evaluation for Noisy Image (0.3 * std):');
disp(['RMSE with clean image: ', num2str(rNoisy2), ' (Check: ', num2str(rCheckNoisy2), ')']);
disp(['PSNR with clean image: ', num2str(psnrNoisy2), ' (Check: ', num2str(psnrCheckNoisy2), ')']);
disp(['RMSE of mean filtered image: ', num2str(rFilteredMean2), ' (Check: ', num2str(rCheckFilteredMean2), ')']);
disp(['PSNR of mean filtered image: ', num2str(psnrFilteredMean2), ' (Check: ', num2str(rCheckFilteredMean2), ')']);
disp(['RMSE of spatial filtered image: ', num2str(rFilteredSmart2), ' (Check: ', num2str(rCheckFilteredSmart2), ')']);
disp(['PSNR of spatial filtered image: ', num2str(psnrFilteredSmart2), ' (Check: ', num2str(psnrCheckFilteredSmart2), ')']);
