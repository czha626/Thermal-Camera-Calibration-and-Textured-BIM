%%% Author: Cheng ZHANG
%%% Date: 3/19/2023

clear;clc;close all

image1 = imread('DJI_0190.jpg');
image1_gray = rgb2gray(image1);

image2 = imread('DJI_0192.jpg');
image2_gray = rgb2gray(image2);

% Convert the images to double for arithmetic operations
image1_double = double(image1);
image2_double = double(image2);

% Find pixels that are non-black in both images
non_black_in_both = (image1_gray > 0) & (image2_gray > 0);

% Histogram distribution of images
[counts1,binLocations1] = imhist(image1_gray(non_black_in_both));
[counts2,binLocations2] = imhist(image2_gray(non_black_in_both));

figure;
% set(gcf,'position',[400,300,320,180]);
plot(counts1/max(counts1),'Color',[0.85 0.33 0.1]);hold on
xlim([50,200]);
set(gca,'fontname','times new roman','FontSize',10);
ylabel('Counts','fontname','times new roman','FontSize',10)
xlabel('Intensity','fontname','times new roman','FontSize',10)
[pks1,locs1] = findpeaks(counts1/max(counts1),binLocations1, 'MinPeakHeight', 0.2);

figure;
% set(gcf,'position',[400,300,320,180]);
plot(counts2,'Color',[0 0.45 0.74]);hold on
xlim([50,200]);
set(gca,'fontname','times new roman','FontSize',10);
ylabel('Counts','fontname','times new roman','FontSize',10)
xlabel('Intensity','fontname','times new roman','FontSize',10)
[pks2,locs2] = findpeaks(counts2/max(counts2),binLocations2, 'MinPeakHeight', 0.2);

%% Linear fitting between the pixel value of these two images
linearFit = polyfit(locs2, locs1, 1);

slope = linearFit(1,1);
intercept = linearFit(1,2);

% Apply the linear relationship to non-zero pixel values of image 2
modified_image2 = image2_gray;
non_zero_pixels = modified_image2(modified_image2 ~= 0);
modified_non_zero_pixels = slope * non_zero_pixels + intercept;
modified_image2(modified_image2 ~= 0) = modified_non_zero_pixels;

%% Stitch image 1 and the corrected image 2
image1_gray_new = image1_gray;
image2_gray_new = modified_image2;

% Get x and y coordinates of non-black pixels
[y, x] = find(non_black_in_both);
meanX = round(mean(x));

% Initialize the merged image with zeros
merged_image = zeros(size(image1_gray));

% Apply merging logic
merged_image(:,1:meanX) = image1_gray_new(:,1:meanX);
merged_image(:,meanX+1:end) = image2_gray_new(:,meanX+1:end);

% Convert the merged image back to uint8 for saving/display
merged_image_uint8 = uint8(merged_image);

figure
imshow(merged_image_uint8)
