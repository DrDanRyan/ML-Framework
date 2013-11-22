%%
close all
clear all
R = 1;
D = 2;
X = 5;
layer = NDimChebyshevHiddenLayer(1, 1, R, D, X);

%%
[z1, z2] = meshgrid(-4:.1:4, -4:.1:4);
z = permute([reshape(z1, 1, []); reshape(z2, 1, [])], [3 2 4 1]);
tanhz = tanh(z);

%%
layer.params{3} = rand(1, 1, R, D, X);
% layer.params{3}(:,:,:,:,2) = layer.params{3}(:,:,:,:,1)/2 + .2*rand(1,1,R,D,1) - .1;
% layer.params{3}(:,:,:,:,4) = layer.params{3}(:,:,:,:,5)/2 + .2*rand(1,1,R,D,1) - .1;
% layer.params{3}(:,:,:,:,3) = 0;
layer.params{4}(1,1,:) = ones(1, R)/R;
%layer.params{4}(1,1,:) = [8/15, 4/15, 2/15, 1/15];
[y, chebRank1, cheb1D] = layer.compute_Chebyshev_interpolants(tanhz);
z3 = double(gather(reshape(y, 81, 81)));

surf(z1, z2, z3)
figure()
contour(z1, z2, z3)

% figure()
% layer.params{3} = layer.params{3} + .2*rand(1, 1, R, D, X) - .1;
% [y, chebRank1, cheb1D] = layer.compute_Chebyshev_interpolants(tanhz);
% z3 = double(gather(reshape(y, 81, 81)));
% surf(z1, z2, z3)
% figure()
% cheb1D = double(gather(reshape(cheb1D, 81, 81, R, D)));
% plot(-4:.1:4, cheb1D(1,:,1,1), 'r')
% hold on
% plot(-4:.1:4, cheb1D(:,1,1,2), 'r--')
% plot(-4:.1:4, cheb1D(1,:,2,1), 'b')
% plot(-4:.1:4, cheb1D(:,1,2,2), 'b--')
% hold off