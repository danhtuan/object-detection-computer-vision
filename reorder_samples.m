function OS = reorder_samples(S)
num_sp = size(S, 3);%number of samples
OS = S;
for i = 1:num_sp
    Si = S(:,:,i);
    Si = reshape(Si, 2, []);
    D = Si(1, :).^2 + Si(2,:).^2;
    [B IX] = sort(D,'ascend');
    Si = Si(:, IX);
    Si = reshape(Si, [], 2);
    OS(:,:,i) = Si;
end
end