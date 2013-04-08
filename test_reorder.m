clc;
clear all;
close all;

imgpath = 'discs3_small.bmp';
img=double(imread(imgpath))/255;
target=double(imread('target.bmp'))/255;
[rows cols] = size(img);
S = [randi(64, 3, 1) randi(64, 3, 1)]; 

%%
outImage = drawcircle(img, S, 3);
figure; imshow(outImage);
OS = reorder_samples(S);
%%
outImage = drawcircle(img, OS(1:2), 1);
figure; imshow(outImage);
