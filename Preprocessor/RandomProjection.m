classdef RandomProjection < Preprocessor
   % Projects data onto a random lower dimensional subspace using a
   % Gaussian distribution to generate random vectors that are then
   % normalized to unit length (in L2 norm). Resulting vectors are NOT
   % orthogonal.
   
   properties
      P % "projection" matrix
   end
   
   methods
      function obj = RandomProjection(inputDim, outputDim, gpuState)
         if nargin < 3
            gpuState = GPUState();
         end
         obj.P = gpuState.randn(outputDim, inputDim);
         columnNorms = sqrt(sum(obj.P.^2, 1));
         obj.P = bsxfun(@rdivide, obj.P, columnNorms);
      end
      
      function data = transform(obj, data)
         data = obj.P*data;
      end
   end
   
end

