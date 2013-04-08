%--------------------------------------------------------------------------
% Jump Diffusion MCMC Algorithm
%--------------------------------------------------------------------------
% TuannD
% 04/04/13
% @see: Computer Vision slides
%--------------------------------------------------------------------------
clc;
clear all;
close all;
rng('shuffle');

fprintf('JUMP DIFFUSION MCMC ALGORITHM\n');
fprintf('Copyright-TuanND-04/2013\n');
%% PARAMETERS
lambda = 3;
Nsteps = 20; %Number of sampling steps
K_MAX = 20; %Max (Number of objects)
init_k = 5;
M_BURN_IN = 3;
STEP_BURN_IN = 2;

%% Step 0: Prepare data
% load image/target
fprintf('Loading image....\n');
imgpath = 'discs8.bmp';
img=double(imread(imgpath))/255;
target=double(imread('target.bmp'))/255;
[rows cols] = size(img);
fprintf(['Loading image ', imgpath,' and target is done\n']);

%% Test gibss sampling
Oxy = gibss_sampling(img, target, 8);
%%
outImg = drawcircle(img, Oxy, 8);
figure(3);imshow(outImg);