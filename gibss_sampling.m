function [Mxy videoseg] = gibss_sampling(img, tg, M)
% GIBSS_SAMPLING do gibs sampling
% @see Computer Vision slides
% TuanND
% 04/06
% Input:
%   img: image on which have objects need to be dectected
%   target: target knowledge
%   k: number of objects

fprintf('\n\t\tGibss:[Step/Discs-100/25]');
%PARAMETER
T = 50;%Number of MCMC steps
M_BURN_IN = 50;
N_BURN_IN = 2;
K_MAX = 50;
[rows cols] = size(img);
%1. Initialize {zi: i = 1, ..., M}
AOxy = zeros(M, 2, T);
AOxy(:,:,1) = [randi(rows, M, 1) randi(cols, M, 1)];%All object position
Cur_Oxy = AOxy(:,:,1);
showImg = drawcircle(img, Cur_Oxy, M);
figure(1); imshow(showImg);
L1 = likelihood(img, tg, Cur_Oxy, M);

Imframe(1:rows,1:cols,1)=showImg; 
Imframe(1:rows,1:cols,2)=showImg; 
Imframe(1:rows,1:cols,3)=showImg;
videoseg(1) = im2frame(Imframe);          % make the first frame
for t = 2:T
    for i = 1:M        
        fprintf('\b\b\b\b\b\b\b');            
        fprintf('%03u/%02u]', t, i);
        Oxy = Cur_Oxy(2*i-1:2*i);%init position of ith object
        for j = 1:K_MAX            
            %Sampling ith variable            
            Dxy = Oxy + round(randn(1,2)*20);
            Dxy=clip(Dxy,1,rows);% make sure the position in the image
            New_Cur_Oxy = Cur_Oxy;
            New_Cur_Oxy(2*i-1:2*i) = Dxy;
            L2=likelihood(img,tg,New_Cur_Oxy,M);% evaluate the likelihood
            v=min(1,L2/L1);                     % compute the acceptance ratio
            u=rand;                             % draw a sample uniformly in [0 1]
            if v>u                
                Oxy = Dxy;% accept the move                        
                Cur_Oxy = New_Cur_Oxy;
                L1 = L2;                
%                 showImg = drawcircle(img, Cur_Oxy, M);
%                 figure(1); imshow(showImg);
            else                       
            end            
        end
        AOxy(:,:, t) = Cur_Oxy;       
        showImg = drawcircle(img, Cur_Oxy, M);
        figure(1); imshow(showImg);
        Imframe(1:rows,1:cols,1)=showImg; 
        Imframe(1:rows,1:cols,2)=showImg; 
        Imframe(1:rows,1:cols,3)=showImg;
        videoseg((t-2)*M + i + 1) = im2frame(Imframe);          
    end
end
fprintf('...Burning...');
%Burn-in
% S = AOxy(:,:, M_BURN_IN+1:N_BURN_IN:T);%do burn-in, drop fist M samples, keep N-steps samples
% OS = reorder_samples(S);
% Mxy = round(mean(OS, 3));
Mxy = Cur_Oxy;
showImg = drawcircle(img, Mxy, M);figure(1);imshow(showImg);
Imframe(1:rows,1:cols,1)=showImg; 
Imframe(1:rows,1:cols,2)=showImg; 
Imframe(1:rows,1:cols,3)=showImg;
videoseg((T-1) * M + 2) = im2frame(Imframe);          
fprintf('Done Gibss');
end