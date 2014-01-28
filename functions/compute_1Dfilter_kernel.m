function kernel = compute_1Dfilter_kernel(filterType, filterSize, gpuState)
if nargin < 3
   gpuState = GPUState();
end

switch filterType
   case 'box'
      kernel = gpuState.ones(filterSize, 1)/filterSize;
   case 'Gaussian'
      x = gpuState.linspace(-2, 2, filterSize)';
      kernel = exp(-x.*x/2);
      kernel = kernel/sum(kernel);
end

end

