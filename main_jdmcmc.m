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
lambda = 20;
Nsteps = 20; %Number of sampling steps
K_MAX = 25; %Max (Number of objects)
init_k = 19;
M_BURN_IN = 3;
STEP_BURN_IN = 2;

%% Step 0: Prepare data
% load image/target
fprintf('Loading image....\n');
imgpath = 'discs20.bmp';
[pathstr, name, ext] = fileparts(imgpath);
img=double(imread(imgpath))/255;
target=double(imread('target.bmp'))/255;
[rows cols] = size(img);
fprintf(['Loading image ', imgpath,' and target is done\n']);
%% Step 1: Initialization
%Initialize locations of k hypothesized objects and the maximum order Kmax.
allStartTime = tic;
num_obj = zeros(Nsteps, 1);
num_obj(1) = init_k;
Oxy = cell(Nsteps, 1);%Objects position
Oxy{1} = [randi(rows, num_obj(1) , 1) randi(cols, num_obj(1), 1)];%k hypothesized object locations
showImg = drawcircle(img, (Oxy{1})', num_obj(1));
figure(1); imshow(showImg);
obj_fn = zeros(Nsteps, 1);
obj_fn(1) = likelihood(img, target, Oxy{1}, num_obj(1)) * poisspdf(num_obj(1), lambda); %aposterior for evaluation
fprintf('Iteration[%02u]--Discs:[%02u]--OBJ_FN:[%d]--Duration[%5.5f]\n', 1, num_obj(1), obj_fn(1), toc(allStartTime));
Imframe(1:rows,1:cols,1)=showImg; 
Imframe(1:rows,1:cols,2)=showImg; 
Imframe(1:rows,1:cols,3)=showImg;
videoseg(1)=im2frame(Imframe);          % make the first frame
%% Step 2: Iteration
%for i=1:N
for i = 2:Nsteps
    iterStartTime = tic;
    % Draw a sample a~U(0,1)
    a = rand(1);
    % If a<0.33 and k>1                         (jump by -1)
    fprintf('Iteration[%02u]--a:[%1.2f]',i, a);
    if a < 0.33 && num_obj(i-1) > 1
        % k=k-1;
        fprintf('--Jump-1');
        num_obj(i) = num_obj(i-1) - 1;%              
        % else if a<0.66 and k<Kmax             (jump by +1)
    elseif a < 0.66 && num_obj(i-1) < K_MAX
        % k=k+1;
        fprintf('--Jump+1');
        num_obj(i) = num_obj(i-1) + 1;        
        % else                                   (no jump)
    else
        fprintf('--Jump+0');
        num_obj(i) = num_obj(i-1);
        % End
    end
    fprintf('--Discs:[%02u]',num_obj(i));
    % MCMC Gibbs sampling
    [Oxy{i} VS] = gibss_sampling(img, target, num_obj(i));
    % Accept or reject by Metropolis Sampling
    obj_fn(i) = likelihood(img, target, Oxy{i}, num_obj(i)) * poisspdf(num_obj(i), lambda);    
    pa_jump = min(obj_fn(i)/obj_fn(i-1), 1);%acceptance probability for jump/no jump (note that poisspdf is the same if no jump)
    fprintf('\n\t\tOBJ_FN:[%d]--AcceptRate:[%2.2f]',obj_fn(i), pa_jump);
    u0 = rand(1);
    if pa_jump > u0 %Accept jump with prob. of p_jump
        %keep the jump
        fprintf('--Accept');        
        num_frame = length(VS);
        cur_num = length(videoseg);
        videoseg(cur_num + 1:cur_num + num_frame) = VS;
    else %Reject jump
        fprintf('--Reject');
        %duplicate previous step
        num_obj(i) = num_obj(i-1);
        Oxy{i} = Oxy{i-1};
        obj_fn(i) = obj_fn(i-1);
    end
    fprintf('--Duration:[%3.3f]\n', toc(iterStartTime));
end
%% Step 3
% Select samples after M iterations (burn-in);
% Obtain a set of samples with certain step size. 
fprintf('Burning to get Experimental Result...\n');
BI_AOxy = Oxy(M_BURN_IN + 1:STEP_BURN_IN:Nsteps);
bi_num_obj = num_obj(M_BURN_IN + 1:STEP_BURN_IN:Nsteps);
%% Step 4
% Compute the mean estimate of the object number k*
final_num_obj = round(mean(bi_num_obj));
fprintf('Number of objects: %02u\n', final_num_obj);
% For samples with k*, re-order all object locations and compute the mean location for each object. 
K_BI_AOxy = BI_AOxy(bi_num_obj == final_num_obj);
num_sp = length(K_BI_AOxy);
S = zeros(final_num_obj, 2, num_sp); 
for i = 1:num_sp
    S(:,:,i) = K_BI_AOxy{i};
end
OS = reorder_samples(S);
final_Oxy = round(mean(OS,3));
fprintf('Object locations:\n');
reshape(final_Oxy, 2, [])'
%% Show the result
fprintf('See figure(2) for final result');
final_img = drawcircle(img, final_Oxy, final_num_obj);
figure(2);imshow(final_img);
cur_num_frame = length(videoseg);
movie2avi(videoseg(1:(cur_num_frame)),['JD_MCMC_',name,'.avi'],'FPS',10,'COMPRESSION','None');
%% Plot
figure; plot(num_obj);title('Number of objects vs. iteration');
figure; plot(obj_fn); title('Object function vs. iteration');
%% Highest obj_fn
fprintf('The result with best objective function...\n');
[max_obj idx] = max(obj_fn);
max_oxy = Oxy{idx};
max_img = drawcircle(img, max_oxy, num_obj(idx));
figure; imshow(max_img);

