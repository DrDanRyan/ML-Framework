function [x, w, D1] = compute_cheb_constants(N, gpuState)
% Computes the Chebyshev points, (Lagrange) weights and differentiation matrix 
if nargin < 2
   gpuState = GPUState();
end

j = gpuState.linspace(0, N-1, N);

% Chebyshev points
x = cos(pi*j/(N-1))';

% Weights
w = gpuState.ones([N,1]);
w(2:2:end) = -1;
w(1) = .5;
w(end) = w(end)*.5;

% First Differentiation Matrix
numerator = bsxfun(@rdivide, w', w);
denominator = bsxfun(@minus, x, x');
D1 = numerator./denominator;
D1(D1 == Inf) = NaN;
column_sums = nansum(D1);
D1(isnan(D1)) = -column_sums;

end

